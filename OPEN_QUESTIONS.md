# Open questions

Decisions deferred past the prototype.

- **Button & microswitch wiring** – how do the capture button and the case-closed microswitch reach the PC? (Arduino over serial, GPIO on a Pi, USB HID keyboard emulator…) Currently stubbed in `xray/hardware.py`.
- **Lid switch vs. button** – could closing the lid trigger the capture by itself, removing the button?
- **Instruction language** – Hebrew/English/Arabic? `cv2.putText` can't render Hebrew; would need Pillow or pre-rendered images.
- **Live tracking** – the overlay is computed once per capture. Only revisit if the box design changes.
- **Capture latency** – SIFT runs on the UI thread (~0.3–1 s freeze on capture). Fine for the prototype; move to a worker thread if it's noticeable.
