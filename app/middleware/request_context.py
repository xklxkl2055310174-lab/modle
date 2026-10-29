import asyncio
import re
import time
import uuid

from starlette.datastructures import Headers
from starlette.requests import Request

from app.api.errors import ServiceError, error_response
from app.middleware.rate_limit import RateLimiter


class RequestContextMiddleware:
    def __init__(self, app, settings, metrics):
        self.app, self.settings, self.metrics = app, settings, metrics
        self.limiter = RateLimiter(settings)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.monotonic()
        headers = Headers(scope=scope)
        candidate = headers.get("x-request-id", "")
        request_id = (
            candidate if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", candidate) else uuid.uuid4().hex
        )
        state = scope.setdefault("state", {})
        state.update(
            request_id=request_id,
            started=started,
            batch_size=0,
            timing={"queue_ms": None, "preprocess_ms": None, "inference_ms": None},
        )
        request = Request(scope)
        status, response_started = 500, False
        self.metrics.inflight.inc()

        async def tracked_send(message):
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            predict = scope["path"].startswith("/v1/models/") and scope["path"].endswith("/predict")
            if predict:
                timeout = self.settings.request_timeout_ms
                if "x-timeout-ms" in headers:
                    try:
                        specified = int(headers["x-timeout-ms"])
                        if specified <= 0:
                            raise ValueError
                        timeout = min(timeout, specified)
                    except ValueError as exc:
                        raise ServiceError(
                            400, "invalid_timeout", "X-Timeout-Ms 必须为正整数"
                        ) from exc
                state["deadline"] = started + timeout / 1000
                service = getattr(scope["app"].state, "service", None)
                if service is None or not service.accepting:
                    raise ServiceError(503, "not_ready", "服务未就绪或正在停机")
                key = headers.get("x-api-key") or (scope.get("client") or ("unknown",))[0]
                if not self.limiter.allow(key):
                    raise ServiceError(429, "rate_limit", "请求频率超限")
                # 在 multipart 解析之前限制整个请求体，chunked 上传同样受约束。
                limit = self.settings.max_upload_bytes + 64 * 1024
                body = bytearray()
                try:
                    async with asyncio.timeout_at(state["deadline"]):
                        while True:
                            message = await receive()
                            if message["type"] == "http.disconnect":
                                return
                            chunk = message.get("body", b"")
                            if len(body) + len(chunk) > limit:
                                raise ServiceError(413, "file_too_large", "上传内容超过大小限制")
                            body.extend(chunk)
                            if not message.get("more_body", False):
                                break
                except TimeoutError as exc:
                    raise ServiceError(504, "timeout", "上传超过截止时间") from exc
                delivered = False

                async def replay():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": bytes(body), "more_body": False}
                    return await receive()

                await self.app(scope, replay, tracked_send)
            else:
                await self.app(scope, receive, tracked_send)
        except ServiceError as exc:
            state["error_code"] = exc.failure.code
            await error_response(request, exc)(scope, receive, tracked_send)
        except Exception:
            if response_started:
                raise
            await error_response(request, ServiceError(500, "internal_error", "服务内部错误"))(
                scope,
                receive,
                tracked_send,
            )
        finally:
            elapsed = time.monotonic() - started
            route = getattr(scope.get("route"), "path", "unmatched")
            if scope["path"].startswith("/v1/models/") and scope["path"].endswith("/predict"):
                route = "/v1/models/{model}/predict"
            self.metrics.inflight.dec()
            self.metrics.requests.labels(route, str(status)).inc()
            self.metrics.request_seconds.labels(route).observe(elapsed)
            if state.get("error_code") in {"rate_limit", "queue_full", "timeout"}:
                self.metrics.rejections.labels(state["error_code"]).inc()
            logger = getattr(scope["app"].state, "access_logger", None)
            if logger:
                logger.access(
                    request_id=request_id,
                    route=route,
                    status=status,
                    serve_mode=self.settings.serve_mode,
                    total_ms=elapsed * 1000,
                    batch_size=state["batch_size"],
                    **state["timing"],
                )
