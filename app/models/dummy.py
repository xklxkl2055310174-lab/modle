import time

from PIL import ImageStat

from app.models.base import ModelRunner, decode_image


class DummyRunner(ModelRunner):
    """离线协议测试用假模型，无权重、无 torch；性能不能代表 ResNet18。"""

    num_classes = 10

    def preprocess(self, data: bytes) -> list[float]:
        image = decode_image(data, self.settings.max_image_pixels)
        return ImageStat.Stat(image).mean

    def infer_batch(self, inputs: list) -> list[list[float]]:
        time.sleep(self.settings.dummy_delay_ms / 1000)
        rows = []
        for values in inputs:
            scores = [1 + ((sum(values) + i * 17) % 101) for i in range(self.num_classes)]
            rows.append([x / sum(scores) for x in scores])
        return rows

    def postprocess(self, output: list[float], top_k: int) -> list[dict]:
        indices = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:top_k]
        return [{"label": f"dummy_{i}", "score": output[i]} for i in indices]
