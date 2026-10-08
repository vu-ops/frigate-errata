from __future__ import annotations

import json

MIN_SIDE = 0.005
MAX_ASPECT = 20.0


def parse_box(box) -> tuple[float, float, float, float] | None:
    if not box:
        return None
    try:
        values = json.loads(box) if isinstance(box, str) else box
        x, y, w, h = (float(v) for v in values)
    except (ValueError, TypeError):
        return None
    return x, y, w, h


def is_oversized_box(box, max_area: float) -> bool:
    """True when a box covers more than `max_area` of the frame (as a fraction).

    Unlike the degenerate slivers caught by is_plausible_box, an oversized box is
    geometrically legal — it is typically a merged detection (two adjacent cars
    under IR at night) or a whole-frame hallucination. Those should be flagged
    for human review, never auto-confirmed or used as training targets.
    """
    parsed = parse_box(box)
    if parsed is None:
        return False
    x, y, w, h = parsed
    if w <= 0 or h <= 0:
        return False
    return w * h > max_area


def is_plausible_box(box, min_side: float = MIN_SIDE, max_aspect: float = MAX_ASPECT) -> bool:
    """Reject degenerate detector boxes (e.g. a full-width sliver at the frame edge).

    A box that is both very thin and extremely elongated cannot be a real object;
    it is a detector artifact. Real detections in the corpus bottom out around a
    min side of 0.009 and an aspect ratio of 11, while artifacts sit at a min side
    below 0.004 and an aspect ratio above 50, so these thresholds separate them
    cleanly without discarding distant small objects or tall narrow ones. A box
    that hugs a frame edge while spanning nearly the full width (top-third /
    whole-frame hallucinations) is the other artifact family; legitimate large
    detections never anchor exactly on the frame edge across the full width.
    """
    parsed = parse_box(box)
    if parsed is None:
        return False
    x, y, w, h = parsed
    if w <= 0 or h <= 0:
        return False
    if min(w, h) < min_side:
        return False
    if max(w, h) / min(w, h) > max_aspect:
        return False
    if w >= 0.9 and h > 0.05 and (x <= 0.02 or x + w >= 0.98) and (y <= 0.02 or y + h >= 0.98):
        return False
    return True
