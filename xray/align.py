"""One-off straightening of the coin photos: `python main.py --align`.

The 10 NIS coin has parallel straight stripes on the left of the core, on both faces.
Each template photo is rotated so its stripes are exactly vertical and on the left,
which puts all three into one shared "straight" frame, the size of the reference
template and centred on the coin:

  reference.png -> reference_aligned.png   the match target
  xray.png      -> xray_aligned.png        the overlay (also scaled to the reference size)
  back.png      -> back_aligned.png        straightened, then mirrored left-right (as seen
                                           through the coin from the front): stripes on the right

So the reference_aligned -> frame transform applies to the other two as-is, and nothing
is aligned at runtime. Re-run after recapturing any of the three photos.

Stripe angle: every whole-degree rotation is tried, scoring how sharply the columns of
the left half of the core show vertical edges (the stripes line up with the columns only
when they are vertical); then the best one is refined in 0.1 degree steps. Only the left
half is scored, so the 180-degree flip (stripes on the right) loses. Per-image manual
angles in `xray_alignment.manual_deg` override the detection.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .matcher import load_image


def coin_circle(mask: np.ndarray) -> tuple[float, float, float]:
    """Centre and radius of the coin from its mask (centroid, radius from the area)."""
    ys, xs = np.nonzero(mask)
    return xs.mean(), ys.mean(), math.sqrt(len(xs) / math.pi)


def rotation(cx: float, cy: float, angle: float, scale: float, out_size: tuple[int, int]) -> np.ndarray:
    """2x3: rotate `angle` degrees counter-clockwise (on screen) and scale about (cx, cy),
    then move that point to the centre of an out_size (w, h) image."""
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[:, 2] += [(out_size[0] - 1) / 2 - cx, (out_size[1] - 1) / 2 - cy]
    return M


class StripeScorer:
    """Stripe energy of one coin photo after a rotation: for the left and right halves of
    the stripe annulus, the variance across columns of the mean horizontal edge strength.
    Straight vertical stripes give sharp column peaks, so a high variance."""

    def __init__(self, img: np.ndarray, mask: np.ndarray, radius: tuple[float, float]):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.gray = cv2.GaussianBlur(g, (0, 0), 1.0)
        self.cx, self.cy, self.r = coin_circle(mask)
        h, w = mask.shape
        yy, xx = np.mgrid[:h, :w]
        rr = np.hypot(xx - self.cx, yy - self.cy) / self.r
        ring = (rr > radius[0]) & (rr < radius[1])
        dx = (xx - self.cx) / self.r
        self.halves = {"left": ring & (dx < -0.1), "right": ring & (dx > 0.1)}  # skip the centre line

    def energy(self, angle: float) -> dict[str, float]:
        h, w = self.gray.shape
        M = cv2.getRotationMatrix2D((self.cx, self.cy), angle, 1)
        gx = np.abs(cv2.Sobel(cv2.warpAffine(self.gray, M, (w, h)), cv2.CV_32F, 1, 0))
        out = {}
        for side, m in self.halves.items():
            n = m.sum(0)
            col = (gx * m).sum(0) / np.maximum(n, 1)
            out[side] = float(col[n > 15].var())
        return out

    def find_angle(self) -> tuple[float, float]:
        """Counter-clockwise rotation (degrees, -180..180) that makes the stripes vertical
        and on the left, plus the left/right stripe energy ratio there (>1 = on the left)."""
        coarse = max(range(360), key=lambda a: self.energy(a)["left"])
        best = max(np.arange(coarse - 1, coarse + 1.01, 0.1), key=lambda a: self.energy(a)["left"])
        e = self.energy(best)
        return (best + 180) % 360 - 180, e["left"] / max(e["right"], 1e-9)


def straighten(img: np.ndarray, mask: np.ndarray, angle: float, out_size: tuple[int, int],
               out_r: float, mirror: bool = False) -> np.ndarray:
    """BGRA image of out_size: the coin rotated by `angle`, scaled to radius out_r,
    centred, alpha = coin mask. mirror flips it left-right afterwards."""
    cx, cy, r = coin_circle(mask)
    M = rotation(cx, cy, angle, out_r / r, out_size)
    out = np.dstack([cv2.warpAffine(img, M, out_size, flags=cv2.INTER_LINEAR),
                     cv2.warpAffine(mask, M, out_size, flags=cv2.INTER_LINEAR)])
    return cv2.flip(out, 1) if mirror else out


def with_guides(bgra: np.ndarray, n: int = 9) -> np.ndarray:
    """Coin on black with thin vertical guide lines, to judge whether the stripes are straight."""
    img = (bgra[:, :, :3].astype(np.float32) * (bgra[:, :, 3:] / 255.0)).astype(np.uint8)
    h, w = img.shape[:2]
    for i in range(1, n + 1):
        x = round(i * w / (n + 1))
        cv2.line(img, (x, 0), (x, h - 1), (0, 255, 0), 1)
    return img


def preview(ref: np.ndarray, xray: np.ndarray, back: np.ndarray) -> np.ndarray:
    """reference | xray | back (mirrored), all with guide lines | reference/xray 50-50 blend."""
    from .render import corner_label
    blend = cv2.addWeighted(ref[:, :, :3], 0.5, xray[:, :, :3], 0.5, 0)
    return np.hstack([corner_label(with_guides(ref), "reference"), corner_label(with_guides(xray), "xray"),
                      corner_label(with_guides(back), "back (mirrored)"), corner_label(blend, "ref/xray blend")])


def run(cfg: dict) -> None:
    assets, acfg = cfg["assets"], cfg["xray_alignment"]
    max_dim = assets["template_max_dim"]
    sources = {"reference": assets["reference"], "xray": assets["front"], "back": assets["back"]}
    for name, path in sources.items():
        if not Path(path).exists():
            raise SystemExit(f"{path} missing. Capture it: python main.py --capture-template")

    # Output frame: the reference template's size as the Matcher loads it, coin centred.
    ref_img, ref_mask = load_image(sources["reference"], max_dim)
    out_size = (ref_img.shape[1], ref_img.shape[0])
    out_r = coin_circle(ref_mask)[2]

    manual = acfg.get("manual_deg") or {}
    results = {}
    for name, path in sources.items():
        img, mask = load_image(path, max_dim)
        if manual.get(name) is not None:
            angle = float(manual[name])
            print(f"[align] {name}: rotate {angle:+.1f} deg (manual, xray_alignment.manual_deg)")
        else:
            angle, ratio = StripeScorer(img, mask, acfg["stripe_radius"]).find_angle()
            print(f"[align] {name}: stripes vertical after rotating {angle:+.1f} deg counter-clockwise "
                  f"(auto; stripe energy left/right {ratio:.1f}x)")
            if ratio < 1.3:
                print(f"[align]   WARNING: stripes not clearly on one side, check the preview "
                      f"and set xray_alignment.manual_deg.{name} if it is wrong")
        results[name] = straighten(img, mask, angle, out_size, out_r, mirror=(name == "back"))

    for name, key in (("reference", "reference_aligned"), ("xray", "xray_aligned"), ("back", "back_aligned")):
        cv2.imwrite(assets[key], results[name])
        print(f"[align] wrote {assets[key]}")

    dbg = Path(cfg["debug_dir"])
    dbg.mkdir(parents=True, exist_ok=True)
    out = dbg / "aligned_preview.png"
    cv2.imwrite(str(out), preview(results["reference"], results["xray"], results["back"]))
    print(f"[align] check it: {out} (stripes parallel to the green lines, the \"10\" overlapping in the blend)")
