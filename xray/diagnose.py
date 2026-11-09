"""Matching diagnostics: `python main.py --diagnose [frame.png ...]`.

Writes images to debug/ and prints statistics that explain WHY matching works or
not:
  - keypoints per template, split into outer ring / inner core
  - a breakdown of every template keypoint under the final transform:
      inlier     matched correctly
      ambiguous  the correct point WAS the best or 2nd-best candidate, but the ratio
                 test threw it away because another spot looked almost the same
                 (repetitive patterns: the stripes, the rim lettering)
      wrong      the correct point wasn't among the 2 best candidates (the spot looks
                 different: another coin variant, wear, the hole, lighting/reflections)
  - for each saved frame: the normal runtime match, its debug view and the x-ray.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from . import render
from .matcher import FeatureMatch, Matcher, Template

RING_FRAC = 0.72   # approx. inner/outer radius ratio of the bimetallic 10 NIS coin


def region_of(pt, size) -> str:
    h, w = size
    r = math.hypot(pt[0] - (w - 1) / 2, pt[1] - (h - 1) / 2) / (min(w, h) / 2)
    return "ring" if r > RING_FRAC else "core"


def breakdown(m: Matcher, t: Template, kp, des, M: np.ndarray, thr: float) -> dict[str, dict[str, int]]:
    """Classify every template keypoint by what happened to it, given the true transform M."""
    out = {r: {"inlier": 0, "ambiguous": 0, "wrong": 0} for r in ("ring", "core")}
    if t.descriptors is None or des is None:
        return out
    ratio = m.cfg["ratio_test"]
    pts = np.float32([k.pt for k in kp])
    for pair in m.bf.knnMatch(t.descriptors, des, k=2):
        if len(pair) < 2:
            continue
        src = t.keypoints[pair[0].queryIdx].pt
        expect = M @ np.float64([src[0], src[1], 1])
        hit = [np.hypot(*(pts[p.trainIdx] - expect)) < thr for p in pair]
        passed = pair[0].distance < ratio * pair[1].distance
        if passed and hit[0]:
            kind = "inlier"
        elif hit[0] or hit[1]:
            kind = "ambiguous"
        else:
            kind = "wrong"
        out[region_of(src, t.image.shape[:2])][kind] += 1
    return out


def print_breakdown(title: str, b: dict[str, dict[str, int]]) -> None:
    print(f"  {title}")
    for region, d in b.items():
        n = sum(d.values()) or 1
        print(f"    {region:4s}: " + "  ".join(f"{k} {v:4d} ({v / n:4.0%})" for k, v in d.items()))


def draw_inliers(t: Template, img: np.ndarray, kp, fm: FeatureMatch) -> np.ndarray:
    inl = [g for g, ok in zip(fm.good, fm.inlier_mask) if ok] if fm.inlier_mask is not None else []
    out = cv2.drawMatches(t.image, t.keypoints, img, kp, fm.good, None,
                          matchColor=(0, 0, 255), singlePointColor=(80, 80, 80),
                          flags=cv2.DrawMatchesFlags_DEFAULT)
    return cv2.drawMatches(t.image, t.keypoints, img, kp, inl, out,
                           matchColor=(0, 255, 0),
                           flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def label(img: np.ndarray, text: str) -> np.ndarray:
    return render.corner_label(img, text)


def diagnose_templates(m: Matcher) -> None:
    ref = m.front
    print("\n== Templates ==")
    for t in (ref, m.templates["back"]):
        regions = [region_of(k.pt, t.image.shape[:2]) for k in t.keypoints]
        print(f"  {t.name:6s} {t.image.shape[1]}x{t.image.shape[0]}: {len(t.keypoints)} keypoints "
              f"(ring {regions.count('ring')}, core {regions.count('core')})")

    if not m.has_reference:
        print("\nNo reference.png yet: capture one with  python main.py --capture-template (key 3)")
        return
    back = m.templates["back"]
    fm = m.match_features(ref, back.keypoints, back.descriptors, m.cfg["ransac_threshold"])
    print(f"\n  baseline, reference vs back (unrelated images): {fm.inliers} inliers / {len(fm.good)} good")


def diagnose_frame(m: Matcher, cfg: dict, path: str, out: Path) -> None:
    frame = cv2.imread(path)
    if frame is None:
        print(f"\nCould not read {path}")
        return
    stem = Path(path).stem
    print(f"\n== Frame {path} ==")
    res = m.match(frame)
    print(f"  {res.summary()}")
    cv2.imwrite(str(out / f"{stem}_debug.png"), render.debug_view(frame, res, m.templates))

    s = min(1.0, m.cfg["frame_max_dim"] / max(frame.shape[:2]))
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
    kp, des = m.sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
    for name, t in m.templates.items():
        fm = m.match_features(t, kp, des, m.cfg["ransac_threshold"])
        cv2.imwrite(str(out / f"{stem}_matches_{name}.png"),
                    label(draw_inliers(t, small, kp, fm), f"{name}: {fm.inliers} inliers / {len(fm.good)} good"))
        sc = res.scores[name]
        if sc.M is not None:
            print_breakdown(f"{name}: what happened to each template keypoint:",
                            breakdown(m, t, kp, des, sc.M * s, m.cfg["ransac_threshold"]))
    if res.side == "front" and m.overlay is not None:
        cv2.imwrite(str(out / f"{stem}_xray.png"),
                    render.composite_xray(frame, m, res.M, cfg["tint"]))


def run(cfg: dict, frames: list[str]) -> None:
    out = Path(cfg["debug_dir"])
    out.mkdir(parents=True, exist_ok=True)
    m = Matcher(cfg, require_overlay=False)
    diagnose_templates(m)
    for f in frames:
        diagnose_frame(m, cfg, f, out)
    print(f"\nImages written to {out.resolve()}")
