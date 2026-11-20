import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from four_light import map_vendor_max, opposite_response, signed_preview
from verify_vendor import bitmap


DENT_ROIS = {
    "1": (2300, 1320, 3570, 2170),
    "2": (2040, 1490, 3310, 2340),
    "3": (2020, 1570, 3290, 2420),
    "4": (2180, 1375, 3450, 2225),
}
LOCAL_SIGMA = 4


def crop(path: Path, box: tuple, mode: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.crop(box).convert(mode))


def load_frames(folder: Path, box: tuple) -> np.ndarray:
    paths = sorted(folder.glob("Image_*.bmp"))
    if len(paths) != 4:
        raise ValueError(f"{folder} 需要四張 Image_*.bmp")

    return np.stack([crop(path, box, "L") for path in paths])


def load_capture(folder: Path, box: tuple) -> tuple:
    frames = load_frames(folder, box)
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


def preview(path: Path, panels: list, columns: int = 3, max_width: int = 480) -> None:
    height, width = panels[0][1].shape[:2]
    panel_width = min(width, max_width)
    panel_height = round(height * panel_width / width)
    tile_height = panel_height + 19
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (panel_width * columns, tile_height * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, pixels) in enumerate(panels):
        x, y = i % columns * panel_width, i // columns * tile_height
        draw.text((x + 6, y + 3), label, fill="black")
        visual = Image.fromarray(pixels).convert("RGB")
        canvas.paste(visual.resize((panel_width, panel_height), Image.Resampling.LANCZOS), (x, y + 19))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def directions(frames: np.ndarray) -> tuple:
    i1, i2, i3, i4 = frames
    p, q = opposite_response(i1, i3), opposite_response(i2, i4)
    valid = (frames.min(axis=0) >= 4) & (frames.max(axis=0) < 250)
    mask = np.zeros_like(valid)
    mask[32:-32, 32:-32] = True
    sample = np.zeros_like(valid)
    sample[::3, ::3] = True
    return p, q, valid, mask & sample & valid


def direction_rgb(frames: np.ndarray, response: np.ndarray, valid: np.ndarray) -> np.ndarray:
    display = np.where(valid, response, 0) * (255 / 0.25)
    return to_u8(np.stack((display, frames.mean(axis=0), -display), axis=-1))


def compare_dents(root: Path, output: Path) -> None:
    overview = []
    for name, box in DENT_ROIS.items():
        frames, height, color = load_capture(root / "壓傷" / name, box)
        p, q, valid, mask = directions(frames)
        fields = integration_terms(p, q, LOCAL_SIGMA)
        if name == "1":
            known = fields[mask]
            design = np.column_stack((np.ones(len(known)), known))
            gain = np.linalg.lstsq(design, height[mask], rcond=None)[0]

        prediction = gain[0] + fields @ gain[1:]
        r = np.corrcoef(prediction[mask], height[mask])[0, 1]
        valid_fraction = valid[32:-32, 32:-32].mean()
        print(f"壓傷 {name}  高度 r={r:.3f}  有效點 {mask.sum():,}  ROI 內部有效比例 {valid_fraction:.1%}")
        panels = [
            (f"Dent {name} | Input 1", frames[0]),
            ("Direction 1-3", signed_preview(p)),
            ("Direction 2-4", signed_preview(q)),
            ("Max fusion", map_vendor_max(frames.max(axis=0))),
            ("Local height candidate", to_u8(prediction)),
            ("Vendor relative height", height),
            ("Direction RGB 1-3", direction_rgb(frames, p, valid)),
            ("Direction RGB 2-4", direction_rgb(frames, q, valid)),
            ("Vendor RGB", color),
        ]
        preview(output / f"dent-{name}.png", panels)
        overview.extend(panels[i] for i in (0, 1, 2, 4, 5, 6))

    preview(output / "overview.png", overview, columns=6, max_width=275)


def main() -> None:
    parser = argparse.ArgumentParser(description="比對局部高度近似、廠商高度與彩色 B")
    parser.add_argument("root", type=Path, help="含壓傷與横纹子目錄的資料根目錄")
    parser.add_argument("preview", type=Path, help="對照 PNG 路徑。使用 --dents 時給輸出目錄")
    parser.add_argument("--dents", action="store_true", help="用壓傷 1 擬合，再輸出壓傷 1–4 對照圖")
    args = parser.parse_args()
    if args.dents:
        compare_dents(args.root, args.preview)
        return

    captures = (
        load_capture(args.root / "壓傷" / "1", DENT_ROIS["1"]),
        load_capture(args.root / "横纹", (1830, 650, 3770, 1390)),
    )
    prepared = []
    for frames, height, color in captures:
        p, q, valid, mask = directions(frames)
        blue = color[..., 2]
        at_230 = blue[height == 230]
        print(f"有效取樣 {mask.sum():,}，高度/B Pearson r = {np.corrcoef(height[mask], blue[mask])[0, 1]:.3f}")
        if at_230.size:
            print(f"高度 = 230 時，彩色 B = {at_230.min()}–{at_230.max()}")
        prepared.append((frames, height, color, p, q, valid, mask))

    train, held = prepared
    preview_height = None
    for sigma in (0, 3, LOCAL_SIGMA, 8):
        training_fields = integration_terms(train[3], train[4], sigma)[train[6]]
        x = np.column_stack((np.ones(len(training_fields)), training_fields))
        expected = train[1][train[6]]
        gain = np.linalg.lstsq(x, expected, rcond=None)[0]
        held_fields = integration_terms(held[3], held[4], sigma)
        prediction = gain[0] + held_fields @ gain[1:]
        train_r = np.corrcoef(x @ gain, expected)[0, 1]
        held_r = np.corrcoef(prediction[held[6]], held[1][held[6]])[0, 1]
        print(f"高通 sigma={sigma}，壓傷擬合 r={train_r:.3f}，橫紋沿用係數 r={held_r:.3f}")
        if sigma == LOCAL_SIGMA:
            preview_height = to_u8(prediction)

    frames, height, vendor_color, _, q, valid, _ = held
    preview(
        args.preview,
        [
            ("Input 1", frames[0]),
            ("Direction 2-4", signed_preview(q)),
            ("Local height candidate", preview_height),
            ("Vendor relative height", height),
            ("Direction RGB", direction_rgb(frames, q, valid)),
            ("Vendor RGB", vendor_color),
        ],
    )


if __name__ == "__main__":
    main()
