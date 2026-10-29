from app.config import Settings
from app.models.base import ModelRunner


def create_runner(settings: Settings) -> ModelRunner:
    """仅在推理线程或 spawn 子进程初始化器中加载模型，不在事件循环加载。"""
    if settings.model_name == "dummy":
        from app.models.dummy import DummyRunner

        return DummyRunner(settings)
    from app.models.resnet import ResNetRunner

    return ResNetRunner(settings)
