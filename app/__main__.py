import uvicorn

from app.main import create_app


class GracefulServer(uvicorn.Server):
    def handle_exit(self, sig, frame):
        # 主线程信号回调立即摘除 readiness；随后由 Uvicorn 等待 HTTP 和 lifespan。
        service = getattr(self.config.app.state, "service", None)
        if service:
            service.begin_shutdown()
        super().handle_exit(sig, frame)


def main():
    application = create_app()
    settings = application.state.settings
    GracefulServer(
        uvicorn.Config(
            application,
            host=settings.host,
            port=settings.port,
            workers=1,
            access_log=False,
            log_level=settings.log_level.lower(),
            timeout_graceful_shutdown=settings.shutdown_timeout_s,
        )
    ).run()


if __name__ == "__main__":
    main()
