import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings, available_cpus
from app.middleware.rate_limit import RateLimiter
from client.sdk import InferenceClient, retry_after


@pytest.mark.parametrize("status", [400, 413, 500])
async def test_sdk_does_not_retry_input_or_server_errors(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={})

    async with InferenceClient(transport=httpx.MockTransport(handler), backoff=0) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.predict(b"x")
    assert len(calls) == 1


@pytest.mark.parametrize("status", [429, 503, 504])
async def test_sdk_retries_with_stable_request_id(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status if len(calls) < 3 else 200, json={"ok": True})

    async with InferenceClient(transport=httpx.MockTransport(handler), backoff=0) as client:
        assert await client.predict(b"x") == {"ok": True}
    assert len(calls) == 3
    assert len({r.headers["X-Request-ID"] for r in calls}) == 1


async def test_sdk_honors_retry_after_budget():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, headers={"Retry-After": "100"})

    async with InferenceClient(transport=httpx.MockTransport(handler), total_timeout=0.1) as client:
        with pytest.raises(TimeoutError):
            await client.predict(b"x")
    assert len(calls) == 1


async def test_connection_error_retried_but_read_timeout_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("offline")
        raise httpx.ReadTimeout("slow")

    async with InferenceClient(transport=httpx.MockTransport(handler), backoff=0) as client:
        with pytest.raises(httpx.ReadTimeout):
            await client.predict(b"x")
    assert len(calls) == 2


def test_cpu_quota(tmp_path):
    (tmp_path / "cpu.max").write_text("150000 100000")
    assert available_cpus(tmp_path) == 1
    (tmp_path / "cpu.max").write_text("max 100000")
    assert available_cpus(tmp_path) >= 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("torch_threads", "-1"),
        ("torch_threads", "bad"),
        ("queue_maxsize", 0),
        ("warmup_batches", "0,1"),
    ],
)
def test_config_validation(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_rate_buckets_refill_and_bounded_keys(settings):
    now = [0.0]
    settings.key_rate_limit_qps, settings.key_rate_limit_burst = 1, 1
    settings.rate_limit_max_keys = 2
    limiter = RateLimiter(settings, clock=lambda: now[0])
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")
    now[0] = 1
    assert limiter.allow("a")
    assert limiter.allow("c")
    assert len(limiter.keys) == 2


def test_retry_after_parsing():
    assert retry_after("3") == 3
    assert retry_after("invalid") == 0
    assert retry_after("Wed, 01 Jan 2020 00:00:00 GMT") == 0
