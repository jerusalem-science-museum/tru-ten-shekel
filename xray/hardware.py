"""Hardware inputs: case microswitch and capture button.

The prototype simulates both from the keyboard. Replace DummyHardware with a
class exposing the same two methods once the real wiring is decided
(see OPEN_QUESTIONS.md).
"""


class DummyHardware:
    def __init__(self, case_closed: bool = False):
        self._case_closed = case_closed
        self._button_pressed = False

    # --- interface used by the app -------------------------------------
    def is_case_closed(self) -> bool:
        return self._case_closed

    def consume_button_press(self) -> bool:
        """True once per press."""
        pressed = self._button_pressed
        self._button_pressed = False
        return pressed

    # --- keyboard simulation -------------------------------------------
    def sim_toggle_case(self) -> None:
        self._case_closed = not self._case_closed
        print(f"[hw] case {'CLOSED' if self._case_closed else 'OPEN'}")

    def sim_press_button(self) -> None:
        self._button_pressed = True
