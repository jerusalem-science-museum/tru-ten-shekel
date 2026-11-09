"""One-off alignment of the x-ray image to the reference: `python main.py --align`.

Finds where xray.png (the modified coin) sits on reference.png (a normal coin) and
writes assets/xray_aligned.png: xray.png warped into reference.png's pixel
coordinates, transparent outside the coin. The exhibit uses that file as the overlay,
so no alignment happens at runtime. Re-run after recapturing either image.

Both crops are centred on the coin, so only the rotation is unknown. SIFT is tried
first (works if both are the same coin variant). If it fails, a brute-force rotation
search compares the coarse shapes of the core. `xray_alignment.mode: manual` uses the
configured rotation instead.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .matcher import FeatureMatch, Matcher, Template


def to3x3(M: np.ndarray) -> np.ndarray:
    return np.vstack([M, [0, 0, 1]])


def centred(ov: Template, ref: Template, rotation_ccw: float) -> np.ndarray:
    """Both templates are centred square crops of the coin: centre to centre, size to
    size, rotated `rotation_ccw` degrees counter-clockwise (as seen on screen)."""
    oh, ow = ov.image.shape[:2]
    rh, rw = ref.image.shape[:2]
    M = cv2.getRotationMatrix2D(((ow - 1) / 2, (oh - 1) / 2), rotation_ccw, min(rw, rh) / min(ow, oh))
    M[:, 2] += [(rw - ow) / 2, (rh - oh) / 2]
    return M


def ccw_angle(M: np.ndarray) -> float:
    return -math.degrees(math.atan2(M[1, 0], M[0, 0]))


def try_sift(m: Matcher, ov: Template, acfg: dict) -> tuple[FeatureMatch, bool]:
    """SIFT-align the x-ray template to the reference. Returns (match, plausible?)."""
    ref = m.front
    fm = m.match_features(ov, ref.keypoints, ref.descriptors, acfg["ransac_threshold"])
    if fm.M is None:
        return fm, False
    # Plausibility: both are centred crops of the coin, so the x-ray centre must land
    # near the reference centre and the scale must be close to the size ratio.
    oh, ow = ov.image.shape[:2]
    rh, rw = ref.image.shape[:2]
    centre = fm.M @ np.float64([(ow - 1) / 2, (oh - 1) / 2, 1])
    off = np.hypot(*(centre - [(rw - 1) / 2, (rh - 1) / 2])) / (min(rw, rh) / 2)
    scale_err = abs(math.hypot(fm.M[0, 0], fm.M[1, 0]) / (min(rw, rh) / min(ow, oh)) - 1)
    return fm, fm.inliers >= acfg["min_inliers"] and off < 0.1 and scale_err < 0.1


def rotation_search(ov: Template, ref: Template, radius_frac: float, size: int = 256) -> np.ndarray:
    """Correlation of blurred edge strength inside the core, for every whole-degree
    counter-clockwise rotation 0..359. Works on coarse shapes (the "10", text lines),
    so it survives a different coin variant, wear and the hole, where SIFT doesn't."""
    def edges(img):
        g = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2)
        return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

    r_edges, o_edges = edges(ref.image), edges(ov.image)
    yy, xx = np.mgrid[:size, :size]
    inside = np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2) < radius_frac * size / 2
    c = ((size - 1) / 2, (size - 1) / 2)
    scores = np.empty(360)
    for a in range(360):
        w = cv2.warpAffine(o_edges, cv2.getRotationMatrix2D(c, a, 1), (size, size))
        scores[a] = np.corrcoef(w[inside], r_edges[inside])[0, 1]
    return scores


def compute(m: Matcher, ov: Template, acfg: dict) -> tuple[np.ndarray, str]:
    """2x3 transform xray.png -> reference.png pixels, plus a description."""
    if acfg["mode"] == "manual":
        return centred(ov, m.front, acfg["rotation_deg"]), f"manual: rotation {acfg['rotation_deg']} deg"

    fm, ok = try_sift(m, ov, acfg)
    print(f"[align] SIFT: {fm.inliers} inliers / {len(fm.good)} good -> {'ok' if ok else 'failed'}")
    if ok and fm.M is not None:
        return fm.M, f"SIFT, {fm.inliers} inliers, rotation {ccw_angle(fm.M):.1f} deg"

    scores = rotation_search(ov, m.front, acfg["search_radius"])
    best = int(np.argmax(scores))
    far = [scores[a] for a in range(360) if min(abs(a - best), 360 - abs(a - best)) > 20]
    print(f"[align] rotation search: best {best} deg (score {scores[best]:.3f}), "
          f"best other angle {max(far):.3f}, median {np.median(scores):.3f}")
    return centred(ov, m.front, best), f"rotation search, {best} deg"


def preview(ref: Template, warped: np.ndarray) -> np.ndarray:
    """reference | aligned x-ray | 50/50 blend."""
    from .render import corner_label
    blend = cv2.addWeighted(ref.image, 0.5, warped, 0.5, 0)
    return np.hstack([corner_label(ref.image, "reference"), corner_label(warped, "xray aligned"),
                      corner_label(blend, "blend")])


def run(cfg: dict) -> None:
    assets = cfg["assets"]
    m = Matcher(cfg, require_overlay=False)
    if not m.has_reference:
        raise SystemExit("Capture a reference first: python main.py --capture-template (key 3)")
    ov = m.make_template("xray", assets["front"])
    M, info = compute(m, ov, cfg["xray_alignment"])
    print(f"[align] {info}")

    h, w = m.front.image.shape[:2]
    img = cv2.warpAffine(ov.image, M, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(ov.mask, M, (w, h), flags=cv2.INTER_LINEAR)
    out = Path(assets["xray_aligned"])
    cv2.imwrite(str(out), np.dstack([img, mask]))
    print(f"[align] wrote {out}")

    dbg = Path(cfg["debug_dir"])
    dbg.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dbg / "xray_on_reference.png"), preview(m.front, img))
    print(f"[align] check it: {dbg / 'xray_on_reference.png'}")
