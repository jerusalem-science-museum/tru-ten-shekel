# X-Ray 10 NIS coin exhibit

Science-museum exhibit prototype. A visitor puts a 10 NIS coin palm-side up in a closed box
with a top-down camera and presses a button. The app captures one frame, locates the coin
(position, rotation, scale) with SIFT, and a scanline sweep reveals a photo of an in-house
"modified" coin (electronics/antennas inside) warped onto it, as if x-rayed.

## Run

Windows, Python 3.12, venv at `.venv`.

```
.venv\Scripts\python main.py --debug                     # exhibit, camera index from config
.venv\Scripts\python main.py --capture-template           # capture + crop templates: 1 front (xray.png), 2 back, 3 reference
.venv\Scripts\python main.py --align                     # once: xray.png -> assets/xray_aligned.png
.venv\Scripts\python main.py --diagnose [frame.png]      # matching diagnostics -> debug/
```

## Layout

- `main.py`: `App` state machine (`idle` → `sweep` → `xray`, or `message`) plus the key handling loop.
- `capture_template.py`: template capture tool (Hough circle → square crop, alpha outside the coin).
- `xray/setup.py`: config loading, camera opening, `CameraControl` (fixed exposure/gain/WB/focus, live keys + P to save).
- `xray/matcher.py`: SIFT + ratio test + `estimateAffinePartial2D` (RANSAC) against reference (front) and back templates. Loads the overlay (xray_aligned.png).
- `xray/align.py`: `--align`, one-off xray.png -> reference.png alignment, writes xray_aligned.png.
- `xray/diagnose.py`: `--diagnose`, match statistics and visualisations.
- `xray/render.py`: compositing, sweep, text panels, debug view, key legend.
- `xray/hardware.py`: `DummyHardware`, keyboard stand-in for the case microswitch and capture button.
- `config.yaml`: every tunable (timings, texts, thresholds, tint, camera). `OPEN_QUESTIONS.md`: deferred decisions.

## Design decisions (agreed with the user, don't undo without asking)

- **Snapshot only, no live tracking.** One capture per button press. The idle screen is the live camera plus instructions.
- **`reference.png` (normal coin) is the match target, `xray.png` is only the overlay.** Matching the modified coin directly worked poorly. xray.png is aligned to reference.png ONCE, offline, by `--align` (SIFT, else brute-force rotation search on the core, else manual `xray_alignment.rotation_deg`), baked into assets/xray_aligned.png. No alignment at startup (user's request); the reference->frame transform applies to xray_aligned.png directly. Without reference.png it falls back to matching xray.png directly.
- **`back.png` is only for detecting a flipped coin**, never shown as an x-ray. Front vs. back is decided by RANSAC inlier count.
- **xray.png is not mirrored** (user confirmed). The modified coin is a different variant ("NEW SHEKELS", different ring lettering) from the user's reference coin ("NEW SHEQALIM", "ISRAEL" on the ring), plus the hole and wear: 0 SIFT features correspond, so --align uses the rotation search (139 deg).
- **Similarity transform, not homography.** The coin lies flat under a top-down camera. `scale_range` rejects degenerate fits.
- **The trigger is separate from the action.** `on_capture_button()` checks the case and calls `capture_and_show_xray()` → `show_xray()`, so a future real button, lid switch, etc. can call these directly.
- **No auto-trigger.** Capture happens only on the button press, and only while the case is closed. Opening the case ends the x-ray.
- **The x-ray result is grayscale by default** (`tint.grayscale`, coin + background). Colour tint is off by default; enable it in config.
- **No camera auto modes.** Exposure, gain, WB temperature and focus are fixed values from config, applied at open (no warm-up, no auto option). Tuned live only in `--debug`/capture tool, P saves to config.yaml.
- **Brightness is fixed with camera exposure** (`CameraControl`), not software contrast. The user explicitly rejected software contrast because it can't recover clipped highlights.
- Templates are captured on the rig itself with the same lighting and exposure as runtime.

## Conventions

- New tunables go in `config.yaml`, not as constants in code.
- New keys: update the handling in `main()`, `KEY_LEGEND` in `main.py` (shown in `--debug`), and the README keys table.
- Camera opens with DirectShow (`cv2.CAP_DSHOW`). On DirectShow, exposure is roughly log2(seconds), and setting it switches to manual.
- On the rig's camera, FOURCC must be set AFTER width/height, otherwise it stays YUY2 at 5.6 fps (MJPG: 25 fps). Measured on the rig.
- `cap.get()` costs ~20 ms on DirectShow: never call it per frame. `CameraControl.status()` uses cached values.
- `cv2.putText` can't render Hebrew. Instruction language is an open question.
- Keep it simple: this is a prototype to demo to the user's boss.

## Testing

No camera is available to Claude. Verify matching and rendering with synthetic images: draw a fake
coin, warp it by a known rotation/scale into a 1920x1080 frame, run `Matcher.match`, check the
angle/scale, and render the debug view to a PNG in the scratchpad. Real-camera behaviour
(exposure support, Hough detection, SIFT on the shiny coin) must be confirmed by the user on the rig.
