"""X-ray 10 NIS coin exhibit.

Usage:
    python main.py [--source 0] [--debug] [--config config.yaml]
    python main.py --capture-template               # capture + crop coin templates (1/2/3 = front/back/reference)
    python main.py --align                          # once: straighten the coin photos by their stripes
    python main.py --diagnose [frame.png ...]       # write matching diagnostics to debug/

Keys:
    Space  capture button (only works while the case is closed).
           In --debug: toggles the x-ray, no case needed (C is not used).
    C      toggle the simulated case microswitch (open/closed, not in --debug)
    D      toggle the diagnostic view (--debug only)
    S      save the last captured frame to debug/ (--debug only, for --diagnose)
    b / B  lower / raise camera exposure (--debug only)
    g / G  lower / raise camera gain (--debug only)
    w / W  lower / raise white balance temperature (--debug only)
    f / F  lower / raise focus (--debug only)
    o / O  lower / raise the back blend weight, overlay.back_weight (--debug only)
    P      save the camera values and the back weight to config.yaml (--debug only)
    Q/Esc  quit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from xray import render
from xray.hardware import DummyHardware
from xray.matcher import Matcher, MatchResult
from xray.setup import DEFAULT_CONFIG, CameraControl, load_config, open_camera, save_config_values

IDLE, SWEEP, XRAY, MESSAGE = "idle", "sweep", "xray", "message"

# Shown in --debug mode. Keep in sync with the key handling in main().
KEY_LEGEND = [
    ("Space", "x-ray on / off"),
    ("D", "debug view"),
    ("S", "save captured frame"),
    ("b / B", "exposure - / +"),
    ("g / G", "gain - / +"),
    ("w / W", "white balance - / +"),
    ("f / F", "focus - / +"),
    ("o / O", "back weight - / +"),
    ("P", "save camera values + back weight"),
    ("Q / Esc", "quit"),
]


class App:
    def __init__(self, cfg: dict, matcher: Matcher, hw: DummyHardware, debug: bool):
        self.cfg = cfg
        self.matcher = matcher
        self.hw = hw
        self.debug = debug
        self.debug_view = False
        self.camera: CameraControl | None = None  # set by main()

        self.state = IDLE
        self.state_since = time.monotonic()
        self.captured: np.ndarray | None = None
        self.xray_img: np.ndarray | None = None
        self.result: MatchResult | None = None
        self.message: list[str] = []
        self.fps = 0.0  # measured display rate, shown in --debug

    def _set_state(self, state: str) -> None:
        self.state = state
        self.state_since = time.monotonic()

    def case_closed(self) -> bool:
        """--debug ignores the case: Space alone toggles the x-ray."""
        return self.debug or self.hw.is_case_closed()

    # --- triggers ----------------------------------------------------------
    def on_capture_button(self, frame: np.ndarray) -> None:
        """Physical button handler. Kept separate so any other trigger can call
        capture_and_show_xray() directly."""
        if not self.case_closed():
            print("[app] button ignored: case is open")
            return
        self.capture_and_show_xray(frame)

    def capture_and_show_xray(self, frame: np.ndarray) -> None:
        self.captured = frame.copy()
        t0 = time.perf_counter()
        self.result = self.matcher.match(self.captured)
        print(f"[match] {self.result.summary()} ({(time.perf_counter() - t0) * 1000:.0f} ms)")

        if self.result.side == "front":
            self.show_xray()
        elif self.result.side == "back":
            self._show_message(self.cfg["text"]["flipped"])
        else:
            self._show_message(self.cfg["text"]["no_match"])

    def show_xray(self) -> None:
        self.xray_img = render.composite_xray(
            self.captured, self.matcher, self.result.M, self.cfg["tint"]
        )
        self._set_state(SWEEP)

    def adjust_back_weight(self, sign: int) -> None:
        """Debug: change overlay.back_weight live, re-rendering an x-ray on screen."""
        step = float(self.cfg["overlay"].get("back_weight_step", 0.05))
        w = round(min(1.0, max(0.0, self.matcher.back_weight + sign * step)), 4)
        if self.matcher.back_overlay is None:
            print("[app] no back_aligned.png loaded, back weight has no effect. Run: python main.py --align")
        self.matcher.back_weight = self.cfg["overlay"]["back_weight"] = w
        print(f"[app] back weight {w:g}  (P saves to config.yaml)")
        if self.state in (SWEEP, XRAY) and self.result is not None and self.result.M is not None:
            self.xray_img = render.composite_xray(self.captured, self.matcher, self.result.M, self.cfg["tint"])

    def save_back_weight(self) -> None:
        path = Path(self.cfg["config_path"])
        if save_config_values(path, "overlay", {"back_weight": self.matcher.back_weight}):
            print(f"[app] saved overlay.back_weight {self.matcher.back_weight:g} to {path}")

    def save_captured(self) -> None:
        if self.captured is None:
            print("[app] nothing captured yet")
            return
        out = Path(self.cfg["debug_dir"])
        out.mkdir(parents=True, exist_ok=True)
        path = out / time.strftime("capture_%Y%m%d_%H%M%S.png")
        cv2.imwrite(str(path), self.captured)
        print(f"[app] saved {path}  (analyse with: python main.py --diagnose {path})")

    def _show_message(self, lines: list[str]) -> None:
        self.message = lines
        self._set_state(MESSAGE)

    # --- per-frame update ----------------------------------------------------
    def update(self, frame: np.ndarray) -> None:
        elapsed = time.monotonic() - self.state_since
        timing = self.cfg["timing"]
        pressed = self.hw.consume_button_press()

        if self.state == IDLE:
            if pressed:
                self.on_capture_button(frame)
        elif pressed and self.debug:
            self._set_state(IDLE)  # --debug: Space toggles the x-ray (or dismisses a message) off
        elif self.state in (SWEEP, XRAY):
            if not self.case_closed():
                self._set_state(IDLE)
            elif self.state == SWEEP and elapsed >= timing["sweep_seconds"]:
                self._set_state(XRAY)
            elif (self.state == XRAY and not self.debug  # --debug: x-ray stays until Space
                  and elapsed >= timing["xray_hold_seconds"]):
                self._set_state(IDLE)
        elif self.state == MESSAGE:
            if elapsed >= timing["message_seconds"]:
                self._set_state(IDLE)

    def render(self, frame: np.ndarray) -> np.ndarray:
        elapsed = time.monotonic() - self.state_since
        show_debug = self.debug and self.debug_view and self.result is not None

        if self.state == IDLE:
            out = render.text_panel(frame, self.cfg["text"]["instructions"])
        elif self.state == SWEEP:
            progress = elapsed / self.cfg["timing"]["sweep_seconds"]
            out = render.sweep(self.captured, self.xray_img, progress, self.cfg["sweep"])
        elif self.state == XRAY:
            out = (render.debug_view(self.captured, self.result, self.matcher.templates)
                   if show_debug else self.xray_img)
        else:  # MESSAGE
            base = (render.debug_view(self.captured, self.result, self.matcher.templates)
                    if show_debug else frame)
            out = render.text_panel(base, self.message)

        if self.debug:
            label = (f"{self.fps:.0f} fps  state: {self.state}  {self.camera.status()}  "
                     f"back {self.matcher.back_weight:g}  [D] debug view: {'on' if self.debug_view else 'off'}")
            out = render.corner_label(out, label) if not show_debug else out
            out = render.legend(out, KEY_LEGEND)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="X-ray coin exhibit")
    parser.add_argument("--source", type=int, default=None, help="webcam index (default: config)")
    parser.add_argument("--debug", action="store_true", help="enable the diagnostic view (D key)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--capture-template", action="store_true",
                        help="capture and crop coin templates (1/2/3 = front/back/reference) "
                             "instead of running the exhibit")
    parser.add_argument("--align", action="store_true",
                        help="straighten reference/xray/back.png by their stripes once -> assets/*_aligned.png")
    parser.add_argument("--diagnose", nargs="*", metavar="FRAME",
                        help="write matching diagnostics to debug/ (optionally for saved frames)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    index = args.source if args.source is not None else cfg["camera"]["index"]

    if args.capture_template:
        import capture_template
        capture_template.run(cfg, index)
        return
    if args.align:
        from xray import align
        align.run(cfg)
        return
    if args.diagnose is not None:
        from xray import diagnose
        diagnose.run(cfg, args.diagnose)
        return

    matcher = Matcher(cfg)
    hw = DummyHardware()
    app = App(cfg, matcher, hw, debug=args.debug)

    cap, camera = open_camera(index, cfg)
    app.camera = camera

    win = cfg["display"]["window_name"]
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if cfg["display"]["fullscreen"]:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] frame grab failed")
                time.sleep(0.05)
                continue

            now = time.perf_counter()
            app.fps = 0.9 * app.fps + 0.1 / max(now - last, 1e-6)
            last = now

            app.update(frame)
            cv2.imshow(win, app.render(frame))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif args.debug and key == ord("p"):
                camera.save()
                app.save_back_weight()
            elif args.debug and camera.handle_key(key):
                pass
            elif args.debug and key in (ord("o"), ord("O")):
                app.adjust_back_weight(1 if key == ord("O") else -1)
            elif key == ord(" "):
                hw.sim_press_button()
            elif key == ord("c") and not args.debug:
                hw.sim_toggle_case()
            elif key == ord("d") and args.debug:
                app.debug_view = not app.debug_view
            elif key == ord("s") and args.debug:
                app.save_captured()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
