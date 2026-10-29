import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest

from app.main import create_app


@asynccontextmanager
async def client_for(settings):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            yield app, c


@pytest.mark.parametrize("mode", ["async_batch", "threadpool", "processpool"])
async def test_all_modes(settings, image_bytes, mode):
    settings.serve_mode = mode
    async with client_for(settings) as (app, client):
        assert (await client.get("/readyz")).status_code == 200
        metadata = (await client.get("/v1/models")).json()["models"][0]
        assert metadata["serve_mode"] == mode
        responses = await asyncio.gather(
            *[
                client.post(
                    "/v1/models/dummy/predict",
                    files={"file": ("a.png", image_bytes)},
                    data={"top_k": "3"},
                    headers={"X-Request-ID": f"request-{i}"},
                )
                for i in range(8)
            ]
        )
        for i, response in enumerate(responses):
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["request_id"] == response.headers["X-Request-ID"] == f"request-{i}"
            assert len(body["predictions"]) == 3
            assert all(value >= 0 for value in body["timing"].values())
            if mode != "async_batch":
                assert body["batch_size"] == 1
        if mode == "async_batch":
            assert max(r.json()["batch_size"] for r in responses) > 1
        if mode == "processpool":
            assert len(set(app.state.service.worker_pids)) == settings.process_workers
        metrics = (await client.get("/metrics")).text
        assert 'status="200"' in metrics and "inference_batch_size_bucket" in metrics
        app.state.service.begin_shutdown()
        assert (await client.get("/readyz")).status_code == 503
        assert (await client.get("/healthz")).status_code == 200
        rejected = await client.post("/v1/models/dummy/predict", files={"file": image_bytes})
        assert rejected.status_code == 503
        assert rejected.headers["Retry-After"] == "1"


@pytest.mark.parametrize(
    ("model", "content", "top_k", "headers", "status"),
    [
        ("missing", b"abc", "5", {}, 400),
        ("dummy", b"bad", "5", {}, 400),
        ("dummy", b"", "5", {}, 400),
        ("dummy", b"x", "0", {}, 400),
        ("dummy", b"x", "11", {}, 400),
        ("dummy", b"x", "abc", {}, 400),
        ("dummy", b"x", "5", {"X-Timeout-Ms": "-1"}, 400),
        ("dummy", b"x", "5", {"X-Timeout-Ms": "abc"}, 400),
    ],
)
async def test_errors(settings, model, content, top_k, headers, status):
    async with client_for(settings) as (_, client):
        r = await client.post(
            f"/v1/models/{model}/predict",
            files={"file": content},
            data={"top_k": top_k},
            headers=headers,
        )
        assert r.status_code == status
        assert set(r.json()) == {"code", "message", "request_id"}


async def test_size_limits(settings):
    settings.max_upload_bytes = 100
    async with client_for(settings) as (_, client):
        for content in (b"x" * 101, b"x" * 70000):
            r = await client.post("/v1/models/dummy/predict", files={"file": content})
            assert r.status_code == 413


async def test_rate_limit(settings, image_bytes):
    settings.rate_limit_qps = 0.01
    settings.rate_limit_burst = 1
    async with client_for(settings) as (_, client):
        first = await client.post("/v1/models/dummy/predict", files={"file": image_bytes})
        second = await client.post("/v1/models/dummy/predict", files={"file": image_bytes})
        assert first.status_code == 200
        assert second.status_code == 429
        assert second.headers["Retry-After"] == "1"
        assert (await client.get("/healthz")).status_code == 200


async def test_timeout_and_health_responsiveness(settings, image_bytes):
    settings.dummy_delay_ms = 200
    async with client_for(settings) as (_, client):
        prediction = asyncio.create_task(
            client.post(
                "/v1/models/dummy/predict",
                files={"file": image_bytes},
                headers={"X-Timeout-Ms": "30"},
            )
        )
        await asyncio.sleep(0.01)
        health = await asyncio.wait_for(client.get("/healthz"), 0.1)
        assert health.status_code == 200
        assert (await prediction).status_code == 504


async def test_missing_form_and_invalid_request_id(settings):
    async with client_for(settings) as (_, client):
        r = await client.post("/v1/models/dummy/predict", headers={"X-Request-ID": "a" * 200})
        assert r.status_code == 400
        assert len(r.headers["X-Request-ID"]) == 32


async def test_chunked_body_limit_and_slow_upload(settings):
    settings.max_upload_bytes = 10

    async def oversized():
        yield b"x" * 40000
        yield b"x" * 40000

    async def slow():
        await asyncio.sleep(0.1)
        yield b"x"

    async with client_for(settings) as (_, client):
        r = await client.post("/v1/models/dummy/predict", content=oversized())
        assert r.status_code == 413
        r = await client.post(
            "/v1/models/dummy/predict", content=slow(), headers={"X-Timeout-Ms": "10"}
        )
        assert r.status_code == 504


async def test_real_image_format_and_pixel_limit(settings):
    import io

    from PIL import Image

    settings.max_image_pixels = 224 * 224
    async with client_for(settings) as (_, client):
        for size, fmt in [((225, 225), "PNG"), ((16, 16), "GIF")]:
            stream = io.BytesIO()
            Image.new("RGB", size).save(stream, fmt)
            r = await client.post(
                "/v1/models/dummy/predict",
                files={"file": ("fake.jpg", stream.getvalue(), "image/jpeg")},
            )
            assert r.status_code == 400
