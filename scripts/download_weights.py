import argparse
import hashlib
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights


def main():
    parser = argparse.ArgumentParser(description="显式下载官方 ImageNet 权重；服务启动绝不联网")
    parser.add_argument("--output", default="weights/resnet18-f37072fd.pth")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = ResNet18_Weights.IMAGENET1K_V1.url
    expected = Path(url).stem.rsplit("-", 1)[-1]
    if path.exists():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if not actual.startswith(expected):
            raise RuntimeError("已存在权重校验失败，请检查文件后手动移走再下载")
        print(f"权重校验通过：{path}")
        return
    temporary = path.with_suffix(".download")
    try:
        torch.hub.download_url_to_file(url, str(temporary), hash_prefix=expected)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"权重已下载并验证 SHA256：{path}")


if __name__ == "__main__":
    main()
