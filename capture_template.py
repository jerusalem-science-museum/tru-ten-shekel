"""Capture a coin template (assets/reference.png, xray.png or back.png) from the camera.

Finds the coin with a Hough circle detector, crops a square around it and saves a
PNG whose alpha channel masks everything outside the coin.

Usage:
    python capture_template.py [--source 0] [--config config.yaml]
    (or: python main.py --capture-template)

Select what you are capturing with 1/2/3 (starts on front):
    1  front     = the modified coin (x-ray overlay)         -> assets/xray.png
    2  back      = a normal coin, other side (flip detection) -> assets/back.png
    3  reference = a normal coin, palm side (match target)    -> assets/reference.png

Keys:
    1/2/3  select front / back / reference
    Space  save the detected coin to the selected template
    R      select the coin manually (drag a box around it, Enter to confirm)
    b/B g/G w/W f/F  lower / raise exposure, gain, white balance, focus
    P      save the camera values to config.yaml
    Q/Esc  quit
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from xray.setup import DEFAULT_CONFIG, load_config, open_camera

WIN = "Capture template"
SIDE_KEYS = {ord("1"): "front", ord("2"): "back", ord("3"): "reference"}


def find_coin(frame: np.ndarray, cfg: dict) -> tuple[float, float, float] | None:
    """Return (x, y, r) of the coin in full-res pixels, or None.

    Picks the LARGEST circle: the 10 NIS coin is bimetallic, so the inner core
    is a circle too and we want the outer rim."""
    s = min(1.0, cfg["detect_max_dim"] / max(frame.shape[:2]))
    small = frame if s == 1.0 else cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # Blur is given in full-res pixels, so it suppresses the same amount of noise and
    # fine rim detail (bevel, lettering) at any detection resolution.
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (0, 0), max(cfg["blur_sigma"] * s, 0.5))
    h = gray.shape[0]
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=h * 0.05,
        param1=300, param2=cfg["circularity"],
        minRadius=int(h * cfg["min_radius_frac"]), maxRadius=int(h * cfg["max_radius_frac"]),
    )
    if circles is None:
        return None
    x, y, r = max(circles[0], key=lambda c: c[2])
    return x / s, y / s, r / s


class CircleSmoother:
    """Median of the last few detections, so the circle stops jittering.

    Hough runs on each frame independently and noise moves its peak by a few pixels;
    the coin is still while capturing, so a median over recent frames is stable. A
    detection far from the current median (coin moved) restarts the history."""

    def __init__(self, cfg: dict):
        self.history: deque[tuple[float, float, float]] = deque(maxlen=int(cfg["smooth_frames"]))
        self.reset_frac = float(cfg["smooth_reset_frac"])

    def update(self, coin: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
        if coin is None:
            return None
        if self.history:
            x, y, r = self.current()
            if np.hypot(coin[0] - x, coin[1] - y) > self.reset_frac * r or abs(coin[2] - r) > self.reset_frac * r:
                self.history.clear()
        self.history.append(coin)
        return self.current()

    def current(self) -> tuple[float, float, float]:
        x, y, r = np.median(np.array(self.history), axis=0)
        return float(x), float(y), float(r)


def crop_coin(frame: np.ndarray, x: float, y: float, r: float, margin: float) -> np.ndarray | None:
    """Square BGRA crop around the circle, transparent outside it. None if it leaves the frame."""
    R = int(round(r * (1 + margin)))
    cx, cy = int(round(x)), int(round(y))
    x0, y0, x1, y1 = cx - R, cy - R, cx + R, cy + R
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
        return None
    crop = frame[y0:y1, x0:x1]
    alpha = np.zeros(crop.shape[:2], np.uint8)
    cv2.circle(alpha, (R, R), R, 255, -1, cv2.LINE_AA)
    return np.dstack([crop, alpha])


def preview_on_checker(bgra: np.ndarray) -> np.ndarray:
    h, w = bgra.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    checker = np.where(((xx // 20 + yy // 20) % 2)[:, :, None] == 0, 200, 120).astype(np.float32)
    a = bgra[:, :, 3:4].astype(np.float32) / 255
    return (bgra[:, :, :3] * a + checker * (1 - a)).astype(np.uint8)


def run(cfg: dict, source: int, side: str = "front") -> None:
    tcfg = cfg["capture_template"]
    cap, camera = open_camera(source, cfg)
    smoother = CircleSmoother(tcfg)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print("  1/2/3: front/back/reference   Space: save   R: manual select   "
          "b/g/w/f (shift = up): camera   P: save camera   Q: quit")
    print(f"Capturing '{side}' -> {cfg['assets'][side]}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            out_path = Path(cfg["assets"][side])
            coin = smoother.update(find_coin(frame, tcfg))
            view = frame.copy()
            if coin:
                x, y, r = coin
                cv2.circle(view, (int(x), int(y)), int(r * (1 + tcfg["margin"])), (0, 255, 0), 2, cv2.LINE_AA)
                status = f"coin r={r:.0f}px  [Space] save {side}  [R] manual  [Q] quit"
            else:
                status = f"no coin found  [R] manual  [Q] quit"
            cv2.putText(view, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cam = f"{camera.status()}  [b g w f] down, shift up  [P] save"
            cv2.putText(view, cam, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, cam, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            sides = "  ".join(f"[{chr(k)}] " + (f">{s}<" if s == side else s) for k, s in SIDE_KEYS.items())
            target = f"{sides}   -> {out_path.name}"
            cv2.putText(view, target, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, target, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WIN, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if camera.handle_key(key):
                continue
            if key in SIDE_KEYS:
                side = SIDE_KEYS[key]
                print(f"Capturing '{side}' -> {cfg['assets'][side]}")
                continue
            if key == ord(" "):
                margin = tcfg["margin"]
            elif key == ord("r"):
                bx, by, bw, bh = cv2.selectROI(WIN, frame, showCrosshair=True)
                coin = (bx + bw / 2, by + bh / 2, min(bw, bh) / 2) if bw and bh else None
                margin = 0.0  # the box the user drew is already the coin
            else:
                continue

            if coin is None:
                print("No coin to save.")
                continue
            crop = crop_coin(frame, *coin, margin=margin)
            if crop is None:
                print("Coin too close to the frame edge, move it inward.")
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), crop)
            print(f"Saved {out_path} ({crop.shape[1]}x{crop.shape[0]})")
            if side in ("reference", "front", "back"):
                print("Re-run the alignment: python main.py --align")
            cv2.imshow(WIN, preview_on_checker(crop))
            cv2.waitKey(1500)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a coin template image")
    parser.add_argument("--source", type=int, default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, args.source if args.source is not None else cfg["camera"]["index"])


if __name__ == "__main__":
    main()
