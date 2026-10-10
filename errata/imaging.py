from __future__ import annotations

import random

# Frigate feeds the detector ~1.35x the object's larger side, with black
# padding when the region extends past the frame. Training on the same domain
# as inference is required or the model scores poorly on clean frames.
CROP_SCALE = 1.35
MIN_SIDE = 160


def region_geometry(box, fw: int, fh: int, min_side: int = MIN_SIDE,
                    scale: float = CROP_SCALE, rng=random):
    """Return (x0, y0, side, box_in_crop) for a square training crop.

    ``box`` is normalized [x, y, w, h]. When ``box`` is None a random square is
    chosen (used for background/false-positive crops). ``box_in_crop`` is the
    box re-expressed in the square crop's normalized coordinates, or None.
    """
    if box is None:
        choices = [s for s in (320, 480, 640) if s <= min(fw, fh)] or [min(fw, fh)]
        side = max(rng.choice(choices) // 4 * 4, 4)
        x0 = rng.randint(0, max(0, fw - side))
        y0 = rng.randint(0, max(0, fh - side))
        return x0, y0, side, None

    px, py, pw, ph = box[0] * fw, box[1] * fh, box[2] * fw, box[3] * fh
    side = int(scale * max(pw, ph))
    side = max(side, min_side, 4)
    side = min(side, 2 * max(fw, fh))
    cx, cy = px + pw / 2.0, py + ph / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    bx = min(max((px - x0) / side, 0.0), 1.0)
    by = min(max((py - y0) / side, 0.0), 1.0)
    bw = min(max(pw / side, 0.0), 1.0)
    bh = min(max(ph / side, 0.0), 1.0)
    return x0, y0, side, (bx, by, bw, bh)


def region_crop_cv2(img, box, imgsz: int, min_side: int = MIN_SIDE):
    """Crop+resize with cv2 (used by the trainer, fast path)."""
    import cv2
    import numpy as np

    fh, fw = img.shape[:2]
    x0, y0, side, box_c = region_geometry(box, fw, fh, min_side=min_side)
    canvas = np.zeros((side, side, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    dx0, dy0 = sx0 - x0, sy0 - y0
    cw = min(fw - sx0, side - dx0)
    ch = min(fh - sy0, side - dy0)
    if cw > 0 and ch > 0:
        canvas[dy0:dy0 + ch, dx0:dx0 + cw] = img[sy0:sy0 + ch, sx0:sx0 + cw]
    crop = cv2.resize(canvas, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    return crop, box_c


def region_crop_pil(img, box, imgsz: int, min_side: int = MIN_SIDE, scale: float = CROP_SCALE):
    """Crop+resize with PIL (used by the web preview, no cv2 dependency).

    Geometry matches region_crop_cv2; only the interpolation library differs.
    ``scale`` controls how much context is included (1.35 = training crop, a
    larger value shows more surroundings for the review preview).
    """
    from PIL import Image

    fw, fh = img.size
    x0, y0, side, box_c = region_geometry(box, fw, fh, min_side=min_side, scale=scale)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    sx0, sy0 = max(0, x0), max(0, y0)
    dx0, dy0 = sx0 - x0, sy0 - y0
    cw = min(fw - sx0, side - dx0)
    ch = min(fh - sy0, side - dy0)
    if cw > 0 and ch > 0:
        piece = img.crop((sx0, sy0, sx0 + cw, sy0 + ch))
        canvas.paste(piece, (dx0, dy0))
    return canvas.resize((imgsz, imgsz), Image.BILINEAR), box_c


def preview_geometry(box, fw: int, fh: int, min_side: int = MIN_SIDE,
                     scale: float = CROP_SCALE) -> tuple[int, int, int]:
    """Square crop geometry clamped fully inside the frame (no black padding).

    Returns ``(x0, y0, side)`` in pixels. Shared by the review preview and the
    GenAI panel so frame boxes and crop boxes agree.
    """
    x0, y0, side, _ = region_geometry(box, fw, fh, min_side=min_side, scale=scale)
    side = max(4, min(side, fw, fh))
    x0 = min(max(int(x0), 0), max(0, fw - side))
    y0 = min(max(int(y0), 0), max(0, fh - side))
    return x0, y0, side


def box_in_crop(box, x0: int, y0: int, side: int, fw: int, fh: int):
    """Re-express a normalized full-frame box in the square crop's coordinates."""
    if box is None:
        return None
    try:
        px, py, pw, ph = box[0] * fw, box[1] * fh, box[2] * fw, box[3] * fh
    except (TypeError, ValueError, IndexError):
        return None
    bx = min(max((px - x0) / side, 0.0), 1.0)
    by = min(max((py - y0) / side, 0.0), 1.0)
    bw = min(max(pw / side, 0.0), 1.0)
    bh = min(max(ph / side, 0.0), 1.0)
    return (bx, by, bw, bh)


def region_crop_pil_fit(img, box, imgsz: int, min_side: int = MIN_SIDE, scale: float = CROP_SCALE):
    """Like region_crop_pil but clamped fully inside the frame -- no black padding.

    Used by the review preview so wide/short cameras don't show black bars: the
    square is shrunk to the frame and shifted inward instead of being padded.
    Training still uses the padded geometry (region_crop_cv2).
    """
    from PIL import Image

    fw, fh = img.size
    x0, y0, side = preview_geometry(box, fw, fh, min_side=min_side, scale=scale)
    crop = img.crop((x0, y0, x0 + side, y0 + side)).resize((imgsz, imgsz), Image.BILINEAR)
    return crop, box_in_crop(box, x0, y0, side, fw, fh)
