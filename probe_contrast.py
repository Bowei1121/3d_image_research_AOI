import argparse
from pathlib import Path

import numpy as np

from four_light import save_gray
from probe_fusion import CASES, smooth_directions
from probe_height import directions, load_frames, preview, to_u8


def centered_fields(fields: np.ndarray, background: np.ndarray) -> tuple:
    values = fields[:, background]
    center = np.median(values, axis=1)[:, None, None]
    centered = fields - center
    scale = 1.4826 * np.median(np.abs(centered[:, background]), axis=1)[:, None, None]
    return centered, np.maximum(scale, 1e-5)


def magnitude(fields: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(valid, np.linalg.norm(fields, axis=0), 0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="比較雙軸頻帶融合、對比校正與梯度")
    parser.add_argument("root", type=Path, help="含壓傷與横纹子目錄的資料根目錄")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    overview = []
    for name, folder, box in CASES:
        frames = load_frames(args.root / folder, box)
        p, q, valid, _ = directions(frames)
        background = np.zeros_like(valid)
        background[72:-110, 72:-72] = valid[72:-110, 72:-72]
        if not background.any():
            parser.error(f"{name} 沒有可用的背景像素")

        fine_sigma = 2 if name == "tongue" else 12
        smoothed = np.stack(smooth_directions(p, q, valid, fine_sigma))
        broad = np.stack(smooth_directions(p, q, valid, 64))
        band, band_scale = centered_fields(smoothed - broad, background)

        common = magnitude(band / band_scale.mean(), valid)
        adjusted = magnitude(band / band_scale, valid)

        gradient_valid = np.zeros_like(valid)
        gradient_valid[1:-1, 1:-1] = (
            valid[1:-1, 1:-1] & valid[:-2, 1:-1] & valid[2:, 1:-1]
            & valid[1:-1, :-2] & valid[1:-1, 2:]
        )
        if not (background & gradient_valid).any():
            parser.error(f"{name} 沒有可用的梯度背景像素")
        gradients = np.stack([derivative for channel in smoothed for derivative in np.gradient(channel)])
        gradient, gradient_scale = centered_fields(gradients, background & gradient_valid)
        gradient_score = magnitude(gradient / gradient_scale, gradient_valid)

        target = args.output / name
        target.mkdir(parents=True, exist_ok=True)
        save_gray(target / "bandpass_common.tiff", common)
        save_gray(target / "bandpass_axis_mad.tiff", adjusted)
        save_gray(target / "gradient.tiff", gradient_score)
        save_gray(target / "valid.png", valid.astype(np.uint8) * 255)
        save_gray(target / "gradient_valid.png", gradient_valid.astype(np.uint8) * 255)
        local, local_scale = centered_fields(
            np.stack((p, q)) - np.stack(smooth_directions(p, q, valid, 32)), background
        )
        previous = magnitude(local / local_scale.mean(), valid)
        panels = [
            (f"{name} | Input 1", frames[0]),
            ("Local 0/32 | Common scale", to_u8(previous * (255 / 6))),
            (f"Bandpass {fine_sigma}/64 | Common scale", to_u8(common * (255 / 6))),
            (f"Bandpass {fine_sigma}/64 | Per-axis scale", to_u8(adjusted * (255 / 6))),
            (f"Gradient after sigma {fine_sigma}", to_u8(gradient_score * (255 / 6))),
        ]
        preview(target / "comparison.png", panels, columns=5, max_width=480)
        overview.extend(panels)
        print(f"{name}  頻帶兩軸 MAD 尺度 {band_scale.ravel().round(4)}")

    preview(args.output / "overview.png", overview, columns=5, max_width=300)


if __name__ == "__main__":
    main()
