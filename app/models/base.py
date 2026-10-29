import io
import math
import warnings
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from app.config import Settings


def decode_image(data: bytes, max_pixels: int) -> Image.Image:
    """推理工作线程/进程调用，解码和验证不占用 ASGI 事件循环。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("仅支持 JPEG/PNG")
                if image.width * image.height > max_pixels:
                    raise ValueError("图片像素数超限")
                image.load()
                return image.convert("RGB")
    except (
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("图片损坏或像素数超限") from exc


class ModelRunner(ABC):
    device = "cpu"
    num_classes = 1000

    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    def preprocess(self, data: bytes) -> Any: ...

    @abstractmethod
    def infer_batch(self, inputs: list[Any]) -> Any: ...

    @abstractmethod
    def postprocess(self, output: Any, top_k: int) -> list[dict]: ...

    def warmup(self) -> None:
        """推理工作线程/进程执行；每个配置的批大小独立预热。"""
        stream = io.BytesIO()
        side = min(224, math.isqrt(self.settings.max_image_pixels))
        Image.new("RGB", (side, side), (100, 120, 140)).save(stream, format="PNG")
        sample = self.preprocess(stream.getvalue())
        for size in self.settings.batch_sizes():
            for _ in range(self.settings.warmup_runs):
                self.infer_batch([sample] * size)

    def clear_cache(self) -> None:
        return None
