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
.venv\Scripts\python main.py --align                     # once: straighten the 3 photos -> assets/*_aligned.png
.venv\Scripts\python main.py --diagnose [frame.png]      # matching diagnostics -> debug/
```

## Layout

- `main.py`: `App` state machine (`idle` → `sweep` → `xray`, or `message`) plus the key handling loop.
- `capture_template.py`: template capture tool (Hough circle → square crop, alpha outside the coin).
- `xray/setup.py`: config loading, camera opening, `CameraControl` (fixed exposure/gain/WB/focus, live keys + P to save).
- `xray/matcher.py`: SIFT + ratio test + `estimateAffinePartial2D` (RANSAC) against reference_aligned (front) and back.png templates. Loads the overlays (xray_aligned.png core, back_aligned.png).
- `xray/align.py`: `--align`, one-off stripe straightening of reference/xray/back.png, writes the `*_aligned.png` files.
- `xray/diagnose.py`: `--diagnose`, match statistics and visualisations.
- `xray/render.py`: compositing, sweep, text panels, debug view, key legend.
- `xray/hardware.py`: `DummyHardware`, keyboard stand-in for the case microswitch and capture button.
- `config.yaml`: every tunable (timings, texts, thresholds, tint, camera). `OPEN_QUESTIONS.md`: deferred decisions.

## Design decisions (agreed with the user, don't undo without asking)

- **Snapshot only, no live tracking.** One capture per button press. The idle screen is the live camera plus instructions.
- **`reference.png` (normal coin) is the match target, `xray.png` is only the overlay.** Matching the modified coin directly worked poorly. Alignment happens ONCE, offline, by `--align`: no alignment at startup (user's request). Without reference.png it falls back to matching xray.png directly; without reference_aligned.png it matches reference.png and the exhibit refuses to start until `--align` is run.
- **`--align` straightens all three photos by the coin's stripes.** The 10 NIS coin has parallel straight stripes on the left of the core on both faces. Each photo is rotated so they are exactly vertical and on the left (whole-degree search over 360 deg + 0.1 deg refinement, maximising column sharpness of horizontal edges in the left half of the core annulus, `xray_alignment.stripe_radius`; scoring only the left half resolves the 180 deg ambiguity). Per-image overrides in `xray_alignment.manual_deg`. Outputs, all reference-template size, coin centred, alpha = coin: `reference_aligned.png` (the match target), `xray_aligned.png` (the overlay), `back_aligned.png`. So the reference_aligned->frame transform applies to the other two directly, and the reported match angle is relative to the straightened reference. Core stripes, not ring: on the modified coin the ring's stripes are a few degrees off the core's.
- **Only the inner core is overlaid** (branch `inner-core-overlay`). The silver ring's lettering differs between coin variants, so the visitor's real ring stays visible. `Matcher.core_mask()` cuts the overlay mask to a soft circle of `overlay.core_radius_frac` (0.69, measured on reference.png) of the coin radius; null = whole coin.
- **`back.png` (as captured) is the back match template, only for detecting a flipped coin.** Front vs. back is decided by RANSAC inlier count. back.png is the modified coin's back (hole near the top).
- **`back_aligned.png` is mirrored left-right** after straightening (as seen through the coin from the front; the user confirmed horizontal mirror is the physical flip), so its stripes are on the right. Its hole lands at the top of the rod in xray_aligned (~20 px right of the rod centre); the user decided to leave it like that (don't rotate the back to line up the hole).
- **The back is blended into the x-ray as a mirrored weighted layer over the whole coin** (`render.composite_xray`). Front layer = the real coin from the frame (ring, inside the warped reference mask) + the xray_aligned core; then `coin = (1 - w) * front + w * back_aligned` (whole back incl. ring, inside its alpha), `w = overlay.back_weight` (0.4; 0/null = off, back_aligned.png then not needed). Mixed in grayscale when `tint.grayscale`. **`tint.invert` inverts the whole coin** (ring + core + back), background untouched (user approved a preview). The colour tint uses the same mixed coin gray. Tuned live with o/O in `--debug` (re-renders the x-ray on screen; P saves it with the camera values).
- **xray.png is not mirrored** (user confirmed). The modified coin is a different variant ("NEW SHEKELS", different ring lettering) from the user's reference coin ("NEW SHEQALIM", "ISRAEL" on the ring), plus the hole and wear: 0 SIFT features correspond, which is why --align uses the stripes instead of SIFT.
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
