"""SIFT matching of a captured frame against the coin templates.

Matching targets:
  front = assets/reference_aligned.png (a normal coin, palm side, straightened by
          `python main.py --align`). Falls back to reference.png, then xray.png, if missing.
  back  = assets/back.png as captured (only used to detect a flipped coin).

The overlay is assets/xray_aligned.png: xray.png (the modified coin) straightened into
the same frame as reference_aligned.png by --align (see align.py). So the reference ->
frame transform applies to the overlay (and to back_aligned.png, blended over the whole
coin at overlay.back_weight) as-is.

All transforms are 2x3 similarity transforms (rotation + uniform scale + translation).
The coin lies flat under a top-down camera, so this is enough and more robust than a
full homography.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Template:
    name: str
    image: np.ndarray        # BGR
    mask: np.ndarray         # uint8, 255 inside the coin
    keypoints: list
    descriptors: np.ndarray | None


@dataclass
class FeatureMatch:
    """Result of matching template features to another image's features."""
    good: list = field(default_factory=list)      # cv2.DMatch passing the ratio test
    M: np.ndarray | None = None                   # 2x3, template -> other image
    inlier_mask: np.ndarray | None = None         # bool per entry of `good`

    @property
    def inliers(self) -> int:
        return 0 if self.inlier_mask is None else int(self.inlier_mask.sum())


@dataclass
class SideScore:
    good_matches: int = 0
    inliers: int = 0
    M: np.ndarray | None = None              # 2x3, template -> full-res frame
    inlier_pts: np.ndarray | None = None     # Nx2, full-res frame coords


@dataclass
class MatchResult:
    side: str | None                          # "front", "back" or None
    scores: dict[str, SideScore] = field(default_factory=dict)

    @property
    def M(self) -> np.ndarray | None:
        return self.scores[self.side].M if self.side else None

    @property
    def angle_deg(self) -> float | None:
        M = self.M
        return None if M is None else math.degrees(math.atan2(M[1, 0], M[0, 0]))

    @property
    def scale(self) -> float | None:
        M = self.M
        return None if M is None else math.hypot(M[0, 0], M[1, 0])

    def summary(self) -> str:
        parts = [f"{n}: {s.inliers} inliers / {s.good_matches} good" for n, s in self.scores.items()]
        head = f"side={self.side}"
        if self.side:
            head += f" angle={self.angle_deg:.1f}deg scale={self.scale:.3f}"
        return head + " | " + ", ".join(parts)


def load_image(path: str, max_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (BGR image, coin mask). Uses the PNG alpha channel as mask if present,
    otherwise assumes the coin fills the image and uses the inscribed circle."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read template image: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        mask = np.where(img[:, :, 3] > 127, 255, 0).astype(np.uint8)
        img = img[:, :, :3].copy()
    else:
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)

    s = max_dim / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, mask


class Matcher:
    def __init__(self, cfg: dict, require_overlay: bool = True):
        self.cfg = cfg["matching"]
        self.full_cfg = cfg
        assets = cfg["assets"]
        self.max_dim = assets["template_max_dim"]
        self.sift = cv2.SIFT_create(nfeatures=self.cfg["sift_features"])
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

        self.has_reference = Path(assets["reference"]).exists()
        # The straightened reference made by --align, in the same frame as xray_aligned.
        self.aligned = self.has_reference and Path(assets["reference_aligned"]).exists()
        if self.aligned:
            ref_path = assets["reference_aligned"]
        elif self.has_reference:
            print(f"[matcher] WARNING: {assets['reference_aligned']} missing, matching against "
                  f"reference.png as captured. Run once: python main.py --align")
            ref_path = assets["reference"]
        else:
            print(f"[matcher] WARNING: {assets['reference']} missing, matching against xray.png directly. "
                  f"Capture one with: python main.py --capture-template (key 3)")
            ref_path = assets["front"]

        self.templates: dict[str, Template] = {
            "front": self.make_template("front", ref_path),
            # As captured, only detects a flip. Core only (matching.back_core_only): the ring
            # lettering differs between mint years, so it would not match the visitor's coin.
            "back": self.make_template("back", assets["back"],
                                       core_only=self.cfg.get("back_core_only", True)),
        }
        self.overlay = self._load_overlay(assets, require_overlay)
        # Mirrored back, blended over the whole coin at this weight. 0/null = off.
        self.back_weight = float(cfg.get("overlay", {}).get("back_weight") or 0.0)
        self.back_overlay = self._load_back_overlay(assets, require_overlay)

    def _load_back_overlay(self, assets: dict, required: bool) -> Template | None:
        """back_aligned.png (made by --align, mirrored, same frame as the overlay), or
        None if there is no aligned reference to place it with. Loaded even at weight 0 so
        the weight can be raised live (o/O in --debug); only required when the weight > 0."""
        if self.overlay is None or not self.aligned:
            return None
        path = assets["back_aligned"]
        if not Path(path).exists():
            if not self.back_weight:
                return None
            if required:
                raise SystemExit(f"{path} missing. Run once: python main.py --align "
                                 f"(or set overlay.back_weight: 0)")
            print(f"[matcher] WARNING: {path} missing, x-ray without the back. Run: python main.py --align")
            return None
        img, mask = load_image(path, self.max_dim)
        if img.shape[:2] != self.front.image.shape[:2]:
            print(f"[matcher] WARNING: back overlay size {img.shape[1]}x{img.shape[0]} != reference size. "
                  f"Re-run: python main.py --align")
        return Template("back_xray", img, mask, [], None)

    def _load_overlay(self, assets: dict, required: bool) -> Template | None:
        """The x-ray image in the front template's pixel coordinates, so the frame
        transform applies to it directly. With a reference that is xray_aligned.png,
        produced once by `python main.py --align` together with reference_aligned.png."""
        if not self.has_reference:
            path = assets["front"]
        else:
            path = assets["xray_aligned"]
            if not self.aligned or not Path(path).exists():
                if required:
                    raise SystemExit(f"{path} or reference_aligned.png missing. Run once: python main.py --align")
                return None
            made = [Path(assets[k]).stat().st_mtime for k in ("reference_aligned", "xray_aligned", "back_aligned")
                    if Path(assets[k]).exists()]
            newest_src = max(Path(assets[k]).stat().st_mtime for k in ("reference", "front", "back"))
            if min(made) < newest_src:
                print("[matcher] WARNING: reference.png, xray.png or back.png changed after the aligned "
                      "images were made. Re-run: python main.py --align")
        img, mask = load_image(path, self.max_dim)
        if img.shape[:2] != self.front.image.shape[:2]:
            print(f"[matcher] WARNING: overlay size {img.shape[1]}x{img.shape[0]} != reference size. "
                  f"Re-run: python main.py --align")
        return Template("xray", img, self.core_mask(mask), [], None)

    def core_mask(self, coin_mask: np.ndarray) -> np.ndarray:
        """Restrict the overlay to the inner core, so the real coin's silver ring (whose
        lettering differs between coin variants) stays visible. Soft-edged circle around
        the coin mask's centre, `overlay.core_radius_frac` of the coin radius."""
        ocfg = self.full_cfg.get("overlay", {})
        frac = ocfg.get("core_radius_frac")
        if not frac:
            return coin_mask                                     # null/0 = whole coin
        ys, xs = np.nonzero(coin_mask)
        cx, cy = xs.mean(), ys.mean()
        coin_r = math.sqrt(len(xs) / math.pi)
        yy, xx = np.mgrid[:coin_mask.shape[0], :coin_mask.shape[1]]
        r = np.hypot(xx - cx, yy - cy) / coin_r
        feather = max(float(ocfg.get("feather_frac", 0.0)), 1e-6)
        alpha = np.clip((frac - r) / feather + 0.5, 0, 1)        # 1 inside, ramps to 0 across the edge
        return np.minimum(coin_mask, (alpha * 255).astype(np.uint8))

    def make_template(self, name: str, path: str, core_only: bool = False) -> Template:
        img, mask = load_image(path, self.max_dim)
        if core_only:
            mask = self.core_mask(mask)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, mask)
        print(f"[matcher] {name}: {len(kp)} keypoints from {path}")
        return Template(name, img, mask, kp, des)
    @property
    def front(self) -> Template:
        return self.templates["front"]

    # --- core feature matching ------------------------------------------------
    def match_features(self, t: Template, kp, des, ransac_threshold: float) -> FeatureMatch:
        fm = FeatureMatch()
        if t.descriptors is None or des is None or len(des) < 2:
            return fm
        for pair in self.bf.knnMatch(t.descriptors, des, k=2):
            if len(pair) == 2 and pair[0].distance < self.cfg["ratio_test"] * pair[1].distance:
                fm.good.append(pair[0])
        if len(fm.good) < 3:
            return fm
        src = np.float32([t.keypoints[m.queryIdx].pt for m in fm.good])
        dst = np.float32([kp[m.trainIdx].pt for m in fm.good])
        M, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold)
        if M is not None:
            fm.M, fm.inlier_mask = M, inl.ravel().astype(bool)
        return fm

    # --- runtime: frame -> templates ------------------------------------------
    def match(self, frame: np.ndarray) -> MatchResult:
        # Downscale for speed; transforms are scaled back to full resolution.
        s = min(1.0, self.cfg["frame_max_dim"] / max(frame.shape[:2]))
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)

        result = MatchResult(side=None)
        for name, t in self.templates.items():
            result.scores[name] = self._score(t, kp, des, s)

        # The back has its own (lower) threshold: it only decides "flip it" vs. "not found",
        # so a false positive there is cheap, and its core-only template has fewer features.
        best = max(result.scores, key=lambda n: result.scores[n].inliers)
        need = self.cfg["min_inliers"]
        if best == "back":
            need = self.cfg.get("min_inliers_back") or need
        if result.scores[best].inliers >= need:
            result.side = best
        return result

    def _score(self, t: Template, kp, des, s: float) -> SideScore:
        fm = self.match_features(t, kp, des, self.cfg["ransac_threshold"])
        score = SideScore(good_matches=len(fm.good))
        if fm.M is None:
            return score
        # Reject degenerate fits (e.g. RANSAC collapsing scattered matches to a point).
        lo, hi = self.cfg["scale_range"]
        if not lo <= math.hypot(fm.M[0, 0], fm.M[1, 0]) / s <= hi:
            return score
        dst = np.float32([kp[m.trainIdx].pt for m in fm.good])
        score.inliers = fm.inliers
        score.M = fm.M / s                      # small-frame coords -> full-res coords
        score.inlier_pts = dst[fm.inlier_mask] / s
        return score
