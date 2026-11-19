import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def opposite_response(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    positive = positive.astype(np.float32)
    negative = negative.astype(np.float32)
    return (positive - negative) / np.maximum(positive + negative, 1)


def save_gray(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels).save(path)


def signed_preview(response: np.ndarray) -> np.ndarray:
    return np.rint(127.5 + 127.5 * np.clip(response / 0.25, -1, 1)).astype(np.uint8)


def compare_vendor(
    folder: Path,
    mapped_max: np.ndarray,
    response_13: np.ndarray,
    response_24: np.ndarray,
    valid: np.ndarray,
) -> None:
    max_files = sorted(folder.glob("最大值融合_*.bmp"))
    normal_files = sorted(folder.glob("法向量_*.bmp"))
    if len(max_files) != 1 or len(normal_files) != 1:
        raise ValueError("比對需要各一張最大值融合與法向量 BMP")

    reference_max = load_gray(max_files[0])
    with Image.open(normal_files[0]) as image:
        normal = np.asarray(image.convert("RGB"))
    if mapped_max.shape != reference_max.shape or normal.shape[:2] != mapped_max.shape:
        raise ValueError("廠商輸出和原圖尺寸不同")

    mismatches = np.count_nonzero(mapped_max != reference_max)
    sample = np.s_[::37, ::37]
    good = valid[sample]
    values = np.stack(
        [
            response_13[sample][good],
            response_24[sample][good],
            normal[sample][..., 0][good],
            normal[sample][..., 1][good],
        ]
    )
    varies = values.shape[1] >= 2 and np.all(np.ptp(values, axis=1) > 0)
    print(f"{folder}  max 差異 {mismatches:,}/{mapped_max.size:,}  相關樣本 {good.sum():,}")
    if varies:
        print("Pearson r（列 1−3、2−4，欄 R、G）")
        print(np.round(np.corrcoef(values)[:2, 2:], 4))


def main() -> None:
    parser = argparse.ArgumentParser(description="四向 AOI 影像的對向差分與最大值融合")
    parser.add_argument("capture", type=Path, help="包含四張依時間命名的 Image_*.bmp")
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare-vendor", action="store_true")
    args = parser.parse_args()

    frames = sorted(
        path
        for path in args.capture.glob("Image_*.bmp")
        if re.fullmatch(r"Image_\d{17}\.bmp", path.name)
    )
    if len(frames) != 4:
        parser.error(f"需要四張 Image_<17 位時間戳>.bmp，找到 {len(frames)} 張")

    images = [load_gray(path) for path in frames]
    if len({image.shape for image in images}) != 1:
        parser.error("四張影像的尺寸必須相同")

    i1, i2, i3, i4 = images
    response_13 = opposite_response(i1, i3)
    response_24 = opposite_response(i2, i4)
    peak = np.maximum.reduce(images)
    valid = (np.minimum.reduce(images) >= 4) & (peak < 250)
    mapped_max = np.floor(48.158 + 0.71357 * peak).astype(np.uint8)

    if args.compare_vendor:
        try:
            compare_vendor(args.capture, mapped_max, response_13, response_24, valid)
        except ValueError as error:
            parser.error(str(error))

    args.output.mkdir(parents=True, exist_ok=True)
    save_gray(args.output / "max.png", peak)
    save_gray(args.output / "max_vendor.png", mapped_max)
    save_gray(args.output / "valid.png", valid.astype(np.uint8) * 255)
    for name, response in [("direction_13", response_13), ("direction_24", response_24)]:
        save_gray(args.output / f"{name}.tiff", response)
        save_gray(args.output / f"{name}.png", signed_preview(response))

    magnitude = np.hypot(response_13, response_24)
    save_gray(args.output / "direction_magnitude.tiff", magnitude)
    magnitude_preview = np.rint(255 * np.clip(magnitude / 0.25, 0, 1)).astype(np.uint8)
    save_gray(args.output / "direction_magnitude.png", magnitude_preview)


if __name__ == "__main__":
    main()
