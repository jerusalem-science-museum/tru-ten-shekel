"""Shared helpers: config loading, camera opening and live camera tuning."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["config_path"] = str(path)  # CameraControl writes tuned values back here
    # Resolve asset paths relative to the config file.
    for key in ("reference", "front", "back", "xray_aligned"):
        cfg["assets"][key] = str(path.parent / cfg["assets"][key])
    cfg["debug_dir"] = str(path.parent / cfg.get("debug_dir", "debug"))
    return cfg


# Fixed camera parameters: config key -> (property, lower/raise keys). No auto modes:
# on DirectShow, setting a value puts that control in manual mode.
PARAMS = {
    "exposure": (cv2.CAP_PROP_EXPOSURE, "bB"),
    "gain": (cv2.CAP_PROP_GAIN, "gG"),
    "wb_temperature": (cv2.CAP_PROP_WB_TEMPERATURE, "wW"),
    "focus": (cv2.CAP_PROP_FOCUS, "fF"),
}


class CameraControl:
    """Applies the fixed camera parameters from config and lets you tune them live.

    Lowercase key lowers, uppercase (shift) raises: b/B exposure, g/G gain,
    w/W white balance, f/F focus. P writes the current values to config.yaml.
    After each change the value is read back, so you can see whether the camera
    actually accepted it."""

    def __init__(self, cap: cv2.VideoCapture, cfg: dict):
        self.cap = cap
        self.cam_cfg = cfg["camera"]
        self.config_path = Path(cfg["config_path"])
        self.steps = self.cam_cfg.get("adjust_steps", {})
        # Last read-back values, for status(). cap.get() is slow on DirectShow
        # (~20 ms each), so it must not be called every frame.
        self.current: dict[str, float] = {}

    def apply_config(self) -> None:
        """Set every parameter to its fixed value from config."""
        report = []
        for key, (prop, _) in PARAMS.items():
            value = self.cam_cfg[key]
            ok = self.cap.set(prop, value)
            actual = self.current[key] = self.cap.get(prop)
            report.append(f"{key}={actual:g}" + ("" if ok and actual == value else f" (asked {value:g}!)"))
        print("[camera] " + ", ".join(report))

    def value(self, key: str) -> float:
        return self.cap.get(PARAMS[key][0])

    def status(self) -> str:
        return "  ".join(f"{k} {self.current.get(k, float('nan')):g}" for k in PARAMS)

    def handle_key(self, key: int) -> bool:
        if key == ord("p"):
            self.save()
            return True
        for name, (prop, keys) in PARAMS.items():
            if key in (ord(keys[0]), ord(keys[1])):
                break
        else:
            return False

        step = float(self.steps.get(name, 1))
        before = self.value(name)
        target = before + (step if key == ord(keys[1]) else -step)
        ok = self.cap.set(prop, target)
        after = self.current[name] = self.value(name)
        if ok and after != before:
            self.cam_cfg[name] = after
            print(f"[camera] {name} {before:g} -> {after:g}  (P saves to config.yaml)")
        else:
            print(f"[camera] {name} NOT changed: requested {target:g}, camera reports {after:g} "
                  f"(set() returned {ok}). The camera may be at its limit or not support this.")
        return True

    def save(self) -> None:
        """Write the current values into the camera section of config.yaml, keeping comments."""
        text = self.config_path.read_text(encoding="utf-8")
        m = re.search(r"^camera:\n(?:[ \t].*\n|\n)*", text, re.M)
        if not m:
            print("[camera] no camera section in config.yaml, nothing saved")
            return
        section = m.group(0)
        for key in PARAMS:
            section = re.sub(rf"^(\s+{key}:[ \t]*)[^#\n]*?([ \t]*(#.*)?)$",
                             lambda mm, v=self.value(key): f"{mm.group(1)}{v:g}{mm.group(2)}",
                             section, count=1, flags=re.M)
        self.config_path.write_text(text[:m.start()] + section + text[m.end():], encoding="utf-8")
        print(f"[camera] saved to {self.config_path}: {self.status()}")


def open_camera(index: int, cfg: dict) -> tuple[cv2.VideoCapture, CameraControl]:
    cam_cfg = cfg["camera"]
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DirectShow: fast open + settings work on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    if cam_cfg.get("fourcc"):  # must come AFTER the resolution: set before it, DirectShow
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam_cfg["fourcc"]))  # stays YUY2 (5.6 fps)

    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # Some drivers report the FOURCC as a negative (signed) int, or -1 when unknown.
    raw = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc = "unknown" if raw in (0, -1) else "".join(
        chr(b) if 32 <= b < 127 else "?" for b in (raw & 0xFFFFFFFF).to_bytes(4, "little"))
    print(f"[camera] {index} opened at {int(w)}x{int(h)}, format {fcc!r}, "
          f"driver fps {cap.get(cv2.CAP_PROP_FPS):g}")

    control = CameraControl(cap, cfg)
    control.apply_config()  # fixed from the first frame, no auto warm-up
    return cap, control
