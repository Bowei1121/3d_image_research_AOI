import argparse
from pathlib import Path

import numpy as np

from four_light import save_gray, signed_preview
from probe_height import DENT_ROIS, directions, load_frames, preview, to_u8


CASES = [
    (f"dent-{name}", Path("壓傷") / name, box) for name, box in DENT_ROIS.items()
] + [("tongue", Path("横纹"), (1830, 650, 3770, 1390))]


def smooth_directions(
    p: np.ndarray, q: np.ndarray, valid: np.ndarray, sigma: float
) -> tuple:
    margin = int(np.ceil(3 * sigma))
    fields = np.stack((p * valid, q * valid, valid.astype(np.float32)))
    padded = np.pad(fields, ((0, 0), (margin, margin), (margin, margin)), mode="reflect")
    height, width = padded.shape[-2:]
    kx = 2 * np.pi * np.fft.rfftfreq(width)[None, :]
    ky = 2 * np.pi * np.fft.fftfreq(height)[:, None]
    kernel = np.exp(-0.5 * sigma**2 * (kx**2 + ky**2))
    blurred = np.fft.irfft2(np.fft.rfft2(padded) * kernel, s=(height, width))
    p_sum, q_sum, weight = blurred[:, margin:-margin, margin:-margin]
    denominator = np.maximum(weight, 1e-6)
    return p_sum / denominator, q_sum / denominator


def local_response(
    p: np.ndarray, q: np.ndarray, valid: np.ndarray, sigma: float
) -> np.ndarray:
    p_background, q_background = smooth_directions(p, q, valid, sigma)
    corrected = np.hypot(p - p_background, q - q_background)
    return np.where(valid, corrected, 0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="比較雙軸方向幅值與去除局部背景的幅值")
    parser.add_argument("root", type=Path, help="含壓傷與横纹子目錄的資料根目錄")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sigma", type=float, default=32, help="背景 Gaussian 的 sigma，單位 px")
    args = parser.parse_args()
    if not np.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("sigma 必須是正數")

    overview = []
    for name, folder, box in CASES:
        frames = load_frames(args.root / folder, box)
        p, q, valid, _ = directions(frames)
        magnitude = np.where(valid, np.hypot(p, q), 0)
        corrected = local_response(p, q, valid, args.sigma)
        target = args.output / name
        target.mkdir(parents=True, exist_ok=True)
        save_gray(target / "magnitude.tiff", magnitude)
        save_gray(target / "local_magnitude.tiff", corrected)
        save_gray(target / "valid.png", valid.astype(np.uint8) * 255)
        panels = [
            (f"{name} | Input 1", frames[0]),
            ("Direction 1-3", signed_preview(p)),
            ("Direction 2-4", signed_preview(q)),
            ("Magnitude | 0-0.25", to_u8(magnitude * (255 / 0.25))),
            ("Local magnitude | 0-0.20", to_u8(corrected * (255 / 0.20))),
        ]
        preview(target / "comparison.png", panels, columns=5, max_width=480)
        overview.extend(panels)
        print(f"{name}  ROI {corrected.shape[1]} x {corrected.shape[0]}  有效像素 {valid.mean():.1%}")

    preview(args.output / "overview.png", overview, columns=5, max_width=300)


if __name__ == "__main__":
    main()
