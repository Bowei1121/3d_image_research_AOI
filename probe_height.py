import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from four_light import opposite_response, signed_preview
from verify_vendor import bitmap


def crop(path: Path, box: tuple, mode: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.crop(box).convert(mode))


def load_capture(folder: Path, box: tuple) -> tuple:
    paths = sorted(folder.glob("Image_*.bmp"))
    if len(paths) != 4:
        raise ValueError(f"{folder} 需要四張 Image_*.bmp")

    frames = np.stack([crop(path, box, "L") for path in paths])
    height = crop(bitmap(folder, "相对高度"), box, "L")
    color = crop(bitmap(folder, "彩色融合"), box, "RGB")
    return frames, height, color


def integration_terms(p: np.ndarray, q: np.ndarray, sigma: float) -> np.ndarray:
    height, width = p.shape
    kx = 2 * np.pi * np.fft.rfftfreq(width)[None, :]
    ky = 2 * np.pi * np.fft.fftfreq(height)[:, None]
    radius_squared = kx**2 + ky**2
    highpass = -np.expm1(-0.5 * radius_squared * sigma**2) if sigma else 1
    radius_squared[0, 0] = 1

    terms = []
    for slope in (p, q):
        fft = np.fft.rfft2(slope.astype(np.float64))
        for frequency in (kx, ky):
            field = -1j * frequency * fft * highpass / radius_squared
            terms.append(np.fft.irfft2(field, s=p.shape))
    return np.stack(terms, axis=-1)


def to_u8(values: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(values, 0, 255)).astype(np.uint8)


def preview(path: Path, panels: list) -> None:
    height, width = panels[0][1].shape[:2]
    panel_width = min(width, 480)
    panel_height = round(height * panel_width / width)
    tile_height = panel_height + 19
    canvas = Image.new("RGB", (panel_width * 3, tile_height * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, pixels) in enumerate(panels):
        x, y = i % 3 * panel_width, i // 3 * tile_height
        draw.text((x + 6, y + 3), label, fill="black")
        visual = Image.fromarray(pixels).convert("RGB")
        canvas.paste(visual.resize((panel_width, panel_height), Image.Resampling.LANCZOS), (x, y + 19))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="比對局部高度近似、廠商高度與彩色 B")
    parser.add_argument("root", type=Path, help="含壓傷與横纹子目錄的資料根目錄")
    parser.add_argument("preview", type=Path)
    args = parser.parse_args()

    captures = (
        load_capture(args.root / "壓傷" / "1", (2300, 1320, 3570, 2170)),
        load_capture(args.root / "横纹", (1830, 650, 3770, 1390)),
    )
    prepared = []
    for frames, height, color in captures:
        i1, i2, i3, i4 = frames
        p, q = opposite_response(i1, i3), opposite_response(i2, i4)
        valid = (frames.min(axis=0) >= 4) & (frames.max(axis=0) < 250)
        mask = np.zeros_like(valid)
        mask[::3, ::3] = True
        mask[:32] = mask[-32:] = mask[:, :32] = mask[:, -32:] = False
        mask &= valid
        blue = color[..., 2]
        at_230 = blue[height == 230]
        print(f"有效取樣 {mask.sum():,}，高度/B Pearson r = {np.corrcoef(height[mask], blue[mask])[0, 1]:.3f}")
        if at_230.size:
            print(f"高度 = 230 時，彩色 B = {at_230.min()}–{at_230.max()}")
        prepared.append((frames, height, color, p, q, valid, mask))

    train, held = prepared
    preview_height = None
    for sigma in (0, 3, 4, 8):
        training_fields = integration_terms(train[3], train[4], sigma)[train[6]]
        x = np.column_stack((np.ones(len(training_fields)), training_fields))
        expected = train[1][train[6]]
        gain = np.linalg.lstsq(x, expected, rcond=None)[0]
        held_fields = integration_terms(held[3], held[4], sigma)
        prediction = gain[0] + held_fields @ gain[1:]
        train_r = np.corrcoef(x @ gain, expected)[0, 1]
        held_r = np.corrcoef(prediction[held[6]], held[1][held[6]])[0, 1]
        print(f"高通 sigma={sigma}，壓傷擬合 r={train_r:.3f}，橫紋沿用係數 r={held_r:.3f}")
        if sigma == 4:
            preview_height = to_u8(prediction)

    frames, height, vendor_color, _, q, valid, _ = held
    q_display = np.where(valid, q, 0) * (255 / 0.25)
    direction_color = to_u8(np.stack((q_display, frames.mean(axis=0), -q_display), axis=-1))
    preview(
        args.preview,
        [
            ("Input 1", frames[0]),
            ("Direction 2-4", signed_preview(q)),
            ("Local height candidate", preview_height),
            ("Vendor relative height", height),
            ("Direction RGB", direction_color),
            ("Vendor RGB", vendor_color),
        ],
    )


if __name__ == "__main__":
    main()
