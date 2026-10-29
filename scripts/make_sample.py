import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description="生成不涉及外部版权的合成压测图，不用于准确率评估")
    parser.add_argument("--output", default="reports/sample.png")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), (32, 48, 72))
    draw = ImageDraw.Draw(image)
    draw.ellipse((48, 48, 208, 208), fill=(220, 160, 70))
    draw.rectangle((90, 90, 166, 166), fill=(80, 160, 210))
    image.save(output)
    print(output)


if __name__ == "__main__":
    main()
