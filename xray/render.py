"""Drawing: x-ray compositing, scanline sweep, text panels and the debug view."""

from __future__ import annotations

import cv2
import numpy as np

from .matcher import Matcher, MatchResult, Template

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _warp(t: Template, M: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Template image and soft-edged alpha (HxWx1, 0..1) warped into the frame by M."""
    img = cv2.warpAffine(t.image, M, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(t.mask, M, size, flags=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)                 # soften the edge slightly
    return img, (mask.astype(np.float32) / 255.0)[:, :, None]


def _gray3(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def composite_xray(frame: np.ndarray, m: Matcher, M: np.ndarray, tint: dict) -> np.ndarray:
    """The x-ray picture of the coin, M = reference template -> frame.

    Inside the coin (front template mask): the real coin from the frame (its ring) with
    the modified coin's core (m.overlay) on top, mixed with the whole mirrored back
    (m.back_overlay) at weight m.back_weight, as if seen through the coin. Then the
    whole coin is inverted / tinted. Outside: the frame (grayscale, dimmed)."""
    h, w = frame.shape[:2]
    gray = tint.get("grayscale", True)
    conv = _gray3 if gray else (lambda x: x)

    background = conv(frame).astype(np.float32)
    _, coin_alpha = _warp(m.front, M, (w, h))

    # Front layer: real coin (ring) + x-ray core.
    xray, core_alpha = _warp(m.overlay, M, (w, h))
    coin = background * (1.0 - core_alpha) + conv(xray).astype(np.float32) * core_alpha

    # Mirrored back, mixed over the whole coin.
    if m.back_overlay is not None and m.back_weight:
        back, back_alpha = _warp(m.back_overlay, M, (w, h))
        a = m.back_weight * back_alpha
        coin = coin * (1.0 - a) + conv(back).astype(np.float32) * a

    if tint.get("invert"):
        coin = 255.0 - coin                                    # negative, like a radiograph (whole coin)

    if tint.get("enabled"):
        g = cv2.cvtColor(np.clip(coin, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        coin = g[:, :, None] * np.float32(tint["color_bgr"])[None, None, :]

    dim = float(tint.get("background_dim", 0.0)) if tint.get("enabled") else 0.0
    background *= 1.0 - dim

    out = coin * coin_alpha + background * (1.0 - coin_alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def sweep(raw: np.ndarray, xray: np.ndarray, progress: float, cfg: dict) -> np.ndarray:
    """Top-to-bottom scanline revealing `xray` over `raw`. progress in [0, 1]."""
    h = raw.shape[0]
    y = int(np.clip(progress, 0, 1) * h)
    out = raw.copy()
    out[:y] = xray[:y]

    color = np.float32(cfg["line_color_bgr"])
    glow = int(cfg["glow"])
    y0, y1 = max(0, y - glow), min(h, y + glow)
    if y1 > y0:
        dist = np.abs(np.arange(y0, y1) - y).astype(np.float32)
        a = (1.0 - dist / max(glow, 1))[:, None, None] ** 2 * 0.6
        band = out[y0:y1].astype(np.float32)
        out[y0:y1] = np.clip(band * (1 - a) + color * a, 0, 255).astype(np.uint8)
    cv2.line(out, (0, y), (out.shape[1], y), tuple(int(c) for c in color), int(cfg["line_thickness"]))
    return out


def text_panel(img: np.ndarray, lines: list[str], where: str = "bottom", scale: float | None = None) -> np.ndarray:
    """Draw centred text lines on a semi-transparent dark band."""
    out = img.copy()
    h, w = out.shape[:2]
    scale = scale or h / 900
    thick = max(1, int(round(scale * 2)))
    line_h = int(45 * scale)
    pad = int(25 * scale)
    band_h = pad * 2 + line_h * len(lines)
    top = h - band_h if where == "bottom" else 0

    overlay = out.copy()
    cv2.rectangle(overlay, (0, top), (w, top + band_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, FONT, scale, thick)
        org = ((w - tw) // 2, top + pad + line_h * i + th + (line_h - th) // 2)
        cv2.putText(out, line, org, FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def legend(img: np.ndarray, entries: list[tuple[str, str]]) -> np.ndarray:
    """Key legend (key, description) in a semi-transparent box at the top-right corner."""
    out = img.copy()
    scale, thick, line_h, pad, gap = 0.6, 1, 26, 12, 16

    def text_w(s: str) -> int:
        return cv2.getTextSize(s, FONT, scale, thick)[0][0]

    key_w = max(text_w(k) for k, _ in entries)
    width = key_w + gap + max(text_w(d) for _, d in entries) + 2 * pad
    height = line_h * len(entries) + 2 * pad
    x0, y0 = out.shape[1] - width - 10, 10

    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    for i, (key, desc) in enumerate(entries):
        y = y0 + pad + line_h * i + 18
        cv2.putText(out, key, (x0 + pad, y), FONT, scale, (0, 255, 255), thick, cv2.LINE_AA)
        cv2.putText(out, desc, (x0 + pad + key_w + gap, y), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def corner_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (15, 35), FONT, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (15, 35), FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def debug_view(frame: np.ndarray, result: MatchResult, templates: dict[str, Template]) -> np.ndarray:
    """Captured frame with each side's detected coin outline, inlier points and scores."""
    out = frame.copy()
    colors = {"front": (0, 255, 0), "back": (0, 0, 255)}

    for name, score in result.scores.items():
        color = colors.get(name, (255, 255, 0))
        if score.inlier_pts is not None:
            for x, y in score.inlier_pts.astype(int):
                cv2.circle(out, (int(x), int(y)), 4, color, -1)
        if score.M is not None:
            t = templates[name]
            h, w = t.image.shape[:2]
            # Coin outline + an "up" tick so the detected rotation is visible.
            ang = np.linspace(0, 2 * np.pi, 72)
            r = min(w, h) / 2
            circle = np.stack([w / 2 + r * np.cos(ang), h / 2 + r * np.sin(ang)], 1)
            up = np.float32([[w / 2, h / 2], [w / 2, h / 2 - r]])
            pts = cv2.transform(circle[None].astype(np.float32), score.M)[0].astype(np.int32)
            upt = cv2.transform(up[None], score.M)[0].astype(np.int32)
            thick = 3 if name == result.side else 1
            cv2.polylines(out, [pts], True, color, thick, cv2.LINE_AA)
            cv2.arrowedLine(out, tuple(upt[0]), tuple(upt[1]), color, thick, cv2.LINE_AA, tipLength=0.08)

    lines = [f"DEBUG  side: {result.side}"]
    if result.side:
        lines.append(f"angle: {result.angle_deg:.1f} deg   scale: {result.scale:.3f}")
    for name, s in result.scores.items():
        ratio = s.inliers / s.good_matches if s.good_matches else 0
        lines.append(f"{name}: inliers {s.inliers} / good {s.good_matches} ({ratio:.0%})")

    for i, line in enumerate(lines):
        org = (15, 35 + 32 * i)
        cv2.putText(out, line, org, FONT, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, org, FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out
