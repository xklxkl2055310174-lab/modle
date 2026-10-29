import asyncio
import random
import time
import uuid
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx


def retry_after(value: str | None) -> float:
    if not value:
        return 0
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0


class InferenceClient:
    def __init__(
        self,
        base_url="http://localhost:8000",
        *,
        total_timeout=10.0,
        max_retries=3,
        backoff=0.1,
        api_key=None,
        transport=None,
    ):
        if total_timeout <= 0 or max_retries < 0 or backoff < 0:
            raise ValueError("超时须为正数，重试次数和退避须非负")
        self.total_timeout, self.max_retries, self.backoff = total_timeout, max_retries, backoff
        self.client = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            headers={"X-API-Key": api_key} if api_key else {},
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        await self.client.aclose()

    async def predict(self, image: bytes, *, model="resnet18", top_k=5, request_id=None):
        """调用方事件循环执行；总预算包含请求、Retry-After 与随机退避。"""
        deadline = time.monotonic() + self.total_timeout
        request_id = request_id or uuid.uuid4().hex
        async with asyncio.timeout(self.total_timeout):
            for attempt in range(self.max_retries + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("客户端总超时预算耗尽")
                delay = 0
                try:
                    response = await self.client.post(
                        f"/v1/models/{model}/predict",
                        files={"file": ("image.png", image)},
                        data={"top_k": str(top_k)},
                        timeout=remaining,
                        headers={
                            "X-Request-ID": request_id,
                            "X-Timeout-Ms": str(max(1, int(remaining * 1000))),
                        },
                    )
                    if response.status_code not in {429, 503, 504} or attempt == self.max_retries:
                        response.raise_for_status()
                        return response.json()
                    delay = retry_after(response.headers.get("Retry-After"))
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if attempt == self.max_retries:
                        raise
                delay = max(delay, self.backoff * 2**attempt * random.uniform(0.5, 1.5))
                if delay >= deadline - time.monotonic():
                    raise TimeoutError("重试等待超过总超时预算")
                await asyncio.sleep(delay)
        raise TimeoutError("客户端总超时预算耗尽")
