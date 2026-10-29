import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from app.api.errors import ServiceError, error_response
from app.api.routes import router
from app.config import Settings
from app.middleware.request_context import RequestContextMiddleware
from app.observability.logging import AccessLogger
from app.observability.metrics import Metrics
from app.serving.batcher import BatchService
from app.serving.processpool import ProcessPoolService
from app.serving.threadpool import ThreadPoolService


def create_app(settings: Settings | None = None, runner=None) -> FastAPI:
    settings = settings or Settings()
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app):
        # 生命周期在事件循环运行；加载、预热和资源采样均委派给工作线程。
        app.state.access_logger = AccessLogger(settings)
        service_type = {
            "async_batch": BatchService,
            "threadpool": ThreadPoolService,
            "processpool": ProcessPoolService,
        }[settings.serve_mode]
        service = service_type(settings, metrics, runner)
        app.state.service = service
        sampler = None
        try:
            await service.start()

            async def sample():
                while True:
                    await asyncio.to_thread(metrics.sample_resources)
                    await asyncio.sleep(settings.resource_interval_s)

            sampler = asyncio.create_task(sample())
            yield
        finally:
            service.begin_shutdown()
            await service.stop()
            if sampler:
                sampler.cancel()
                with suppress(asyncio.CancelledError):
                    await sampler
            await asyncio.to_thread(app.state.access_logger.close)

    app = FastAPI(title="小模型推理服务", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.metrics = metrics
    app.add_middleware(RequestContextMiddleware, settings=settings, metrics=metrics)
    app.include_router(router)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        request.state.error_code = exc.failure.code
        return error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc):
        return error_response(request, ServiceError(400, "invalid_input", "请检查图片和表单参数"))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc):
        return error_response(request, ServiceError(exc.status_code, "http_error", str(exc.detail)))

    return app


app = create_app()
