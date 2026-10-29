from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18

from app.models.base import ModelRunner, decode_image


class ResNetRunner(ModelRunner):
    def __init__(self, settings):
        super().__init__(settings)
        threads = settings.thread_count()
        if threads:
            torch.set_num_threads(threads)
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if settings.device == "auto"
            else settings.device
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda 但 CUDA 不可用")
        weights = Path(settings.weights_path)
        if not weights.is_file():
            raise FileNotFoundError("权重不存在，请先运行 python -m scripts.download_weights")
        self.model = resnet18(weights=None)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.eval().to(self.device)
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.labels = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]

    def preprocess(self, data: bytes):
        return self.transform(decode_image(data, self.settings.max_image_pixels))

    def infer_batch(self, inputs):
        """专用线程/子进程执行；复制到 CPU 时同步 GPU，计时包含设备拷贝。"""
        with torch.inference_mode():
            batch = torch.stack(inputs).to(self.device)
            return self.model(batch).softmax(dim=1).cpu()

    def postprocess(self, output, top_k):
        scores, indices = output.topk(top_k)
        return [
            {"label": self.labels[i], "score": float(score)}
            for score, i in zip(scores.tolist(), indices.tolist(), strict=True)
        ]

    def clear_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
