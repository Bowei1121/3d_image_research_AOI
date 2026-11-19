import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from four_light import load_gray


COLOR_PROFILES = {
    "A": (1.45652, -77.35846),
    "B": (1.31579, -24.710585),
}
# Fitted on dent 1–4 only. The tongue capture was held out.
DIRECTION_MATRIX = np.array(
    [
        [0.37687, 0.01469, -0.00296],
        [-0.00218, 0.38373, 0.01765],
        [-0.01174, -0.09946, -0.03762],
    ],
    dtype=np.float64,
)
NORMAL_GAIN = np.array([942.93704, 778.15884, 6037.40606])
NORMAL_OFFSET = np.array([89.11223, 97.43618, -5690.49935])
CURVATURE_GAIN = np.array([-916.36691, -744.40786, -2151.85159])


def bitmap(folder: Path, prefix: str) -> Path:
    paths = list(folder.glob(f"{prefix}_*.bmp"))
    if len(paths) != 1:
        raise ValueError(f"{folder} 需要恰好一張 {prefix} BMP")
    return paths[0]


def quantize(value: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(value, 0, 255)).astype(np.uint8)


def reconstruct(frames: np.ndarray) -> tuple:
    i1, i2, i3, i4 = np.moveaxis(frames.astype(np.float64), -1, 0)
    total = i1 + i2 + i3 + i4
    dx, dy = i1 - i3, i2 - i4
    mean = total / 4
    gray = quantize(1.37494 * mean - 35.91784)
    reflectance = quantize(
        np.sqrt((1.57206 * mean) ** 2 + (0.42146 * dx / 2) ** 2 + (0.44876 * dy / 2) ** 2)
        - 107.28464
    )
    directional = np.stack((2 * dx, 2 * dy, i1 + i3 - i2 - i4), axis=-1)
    directional /= np.maximum(total[..., None], 1)
    direction = directional @ DIRECTION_MATRIX.T
    direction[..., 2] += 1
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    normal = quantize(direction * NORMAL_GAIN + NORMAL_OFFSET)
    curvature = quantize(direction @ CURVATURE_GAIN + 2310.48150)
    return gray, reflectance, normal, curvature


def packed_input(frames: np.ndarray) -> np.ndarray:
    return frames.astype(np.uint32) @ np.array([1, 256, 65536, 16777216], dtype=np.uint32)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def inspect(folder: Path, step: int) -> tuple:
    paths = sorted(folder.glob("Image_*.bmp"))
    if len(paths) != 4:
        raise ValueError(f"{folder} 需要恰好四張 Image_*.bmp")

    sample = np.s_[::step, ::step]
    frames = np.stack([load_gray(path)[sample] for path in paths], axis=-1)
    gray = load_gray(bitmap(folder, "灰度融合"))
    reflectance = load_gray(bitmap(folder, "反射率"))[sample]
    normal = read_rgb(bitmap(folder, "法向量"))[sample]
    curvature = load_gray(bitmap(folder, "曲率"))
    height = load_gray(bitmap(folder, "相对高度"))[sample]
    color = read_rgb(bitmap(folder, "彩色融合"))
    print(f"{folder}  網格樣本 {frames[..., 0].size:,}")

    for name, (gain, bias) in COLOR_PROFILES.items():
        lut = np.clip(np.floor(gain * np.arange(256) + bias), 0, 255).astype(np.uint8)
        r_mismatch = np.count_nonzero(lut[curvature] != color[..., 0])
        g_mismatch = np.count_nonzero(lut[gray] != color[..., 1])
        print(f"  彩色 {name}  全圖 R、G 差異 {r_mismatch:,} / {g_mismatch:,}（每通道 {gray.size:,}）")

    blue_at_height_ceiling = color[sample][..., 2][height == 255]
    if blue_at_height_ceiling.size:
        print(
            f"  高度 255 的位置，彩色 B 範圍 "
            f"{blue_at_height_ceiling.min()}–{blue_at_height_ceiling.max()}"
        )

    gray = gray[sample]
    curvature = curvature[sample]
    references = (gray, reflectance, normal, curvature)
    for name, prediction, reference in zip(
        ("灰度", "反射率", "法向 RGB", "曲率"), reconstruct(frames), references
    ):
        mask = reference > 0 if name in ("灰度", "反射率") else np.ones(reference.shape[:2], dtype=bool)
        error = np.abs(prediction[mask].astype(np.int16) - reference[mask])
        if error.size:
            print(f"  {name}  MAE {error.mean(axis=0).round(3)}  ≤1 DN {np.mean(error <= 1, axis=0).round(4)}")

    outputs = np.column_stack(
        [gray.ravel(), reflectance.ravel(), normal.reshape(-1, 3), curvature.ravel(), height.ravel()]
    )
    return packed_input(frames).ravel(), outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="檢查四光廠商輸出和跨組重現性")
    parser.add_argument("captures", type=Path, nargs="+", help="最後一組用於未參與擬合的交叉比對")
    parser.add_argument("--step", type=int, default=5, help="固定網格的像素間距")
    args = parser.parse_args()
    if len(args.captures) < 2 or args.step < 1:
        parser.error("需至少兩組影像，且 step 必須大於零")

    try:
        *training, (held_keys, held_outputs) = [inspect(folder, args.step) for folder in args.captures]
    except ValueError as error:
        parser.error(str(error))

    known_keys, first = np.unique(np.concatenate([keys for keys, _ in training]), return_index=True)
    known_outputs = np.concatenate([outputs for _, outputs in training])[first]
    positions = np.minimum(np.searchsorted(known_keys, held_keys), len(known_keys) - 1)
    matched = held_keys == known_keys[positions]
    differences = held_outputs[matched] != known_outputs[positions[matched]]
    print("跨組同一四燈亮度組合（依序為灰度、反射率、法向 RGB、曲率、相對高度）")
    print(f"  重疊 {matched.sum():,}/{held_keys.size:,}  各欄差異 {differences.sum(axis=0)}")


if __name__ == "__main__":
    main()
