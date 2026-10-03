import random
import threading
import time
from collections import deque
from enum import Enum
from typing import List, Optional, Sequence

import cv2 as cv
import numpy as np

from verification.move_detector import is_valid_move
from verification.swipe_detector import is_valid_swipe
from verification.thumbs_up_detector import is_thumbs_up

try:
    import winsound
    _WINSOUND_AVAILABLE = True
except ImportError:
    _WINSOUND_AVAILABLE = False


def _beep(freq: int, duration_ms: int) -> None:
    """Fire a non-blocking beep on Windows; no-op elsewhere."""
    if _WINSOUND_AVAILABLE:
        threading.Thread(
            target=winsound.Beep, args=(freq, duration_ms), daemon=True
        ).start()


class VerificationState(Enum):
    IDLE = "IDLE"
    CHALLENGE = "CHALLENGE"
    VERIFYING = "VERIFYING"
    SUCCESS = "SUCCESS"
    TRY_AGAIN = "TRY_AGAIN"
    FAILED = "FAILED"


class GestureVerifier:
    def __init__(
        self,
        static_labels: Optional[List[str]] = None,
        dynamic_labels: Optional[List[str]] = None,
        max_attempts: int = 3,
        time_limit: float = 6.0,
        gesture_hold_time: float = 0.6,
        prep_time: float = 2.0,
        success_banner_time: float = 3.0,
        try_again_banner_time: float = 2.0,
        failed_banner_time: float = 3.0,
        labels: Optional[List[str]] = None,
    ):
        if static_labels is None:
            static_labels = labels if labels is not None else ["Open", "Close", "Pointer", "OK"]
        self.static_labels: List[str] = static_labels
        self.dynamic_labels: List[str] = (
            dynamic_labels
            if dynamic_labels is not None
            else ["Stop", "Clockwise", "Counter Clockwise", "Move"]
        )
        self.max_attempts: int = max_attempts
        self.time_limit: float = time_limit
        self.gesture_hold_time: float = gesture_hold_time
        self.prep_time: float = prep_time
        self.success_banner_time: float = success_banner_time
        self.try_again_banner_time: float = try_again_banner_time
        self.failed_banner_time: float = failed_banner_time

        self.state: VerificationState = VerificationState.IDLE
        self.challenge_type: str = "STATIC"
        self.required_hand: str = "RIGHT"
        self.target_gesture_id: int = 0
        self.target_name: str = ""       # e.g. "THUMBS UP", "OPEN", "SWIPE" …
        self.detected_hand: Optional[str] = None
        self.current_attempt: int = 1
        self.state_start_time: float = time.time()
        self.matched_gesture_start: Optional[float] = None
        self.swipe_history: deque = deque(maxlen=32)
        self.palm_trail: deque = deque(maxlen=32)

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def start_verification(self) -> None:
        self.current_attempt = 1
        self._reset_dynamic_histories()
        self._pick_new_target()
        self.state = VerificationState.CHALLENGE
        self.state_start_time = time.time()
        self.matched_gesture_start = None

    def _next_challenge(self) -> None:
        self._reset_dynamic_histories()
        self._pick_new_target()
        self.state = VerificationState.CHALLENGE
        self.state_start_time = time.time()
        self.matched_gesture_start = None

    def _reset_dynamic_histories(self) -> None:
        self.swipe_history.clear()
        self.palm_trail.clear()
        self.matched_gesture_start = None

    def _return_to_idle(self) -> None:
        self.state = VerificationState.IDLE
        self.current_attempt = 1
        self._reset_dynamic_histories()

    # ------------------------------------------------------------------
    # Challenge selection
    # ------------------------------------------------------------------

    def _pick_new_target(self) -> None:
        """60% STATIC / 40% DYNAMIC.
        Within STATIC: 65% THUMBS UP, then ~8.75% each for OPEN/CLOSE/POINTER/OK.
        Within DYNAMIC: 70% SWIPE / 30% MOVE.
        """
        self.required_hand = random.choice(["LEFT", "RIGHT"])

        if bool(self.static_labels):
            self.challenge_type = "STATIC" if random.random() < 0.6 else "DYNAMIC"
        else:
            self.challenge_type = "DYNAMIC"

        if self.challenge_type == "STATIC":
            if random.random() < 0.65:
                # Landmark-based: no classifier ID needed
                self.target_gesture_id = -1
                self.target_name = "THUMBS UP"
            else:
                # Remaining 35% evenly split across 4 classifier gestures
                self.target_gesture_id = random.choice([0, 1, 2, 3])
                self.target_name = self.static_labels[self.target_gesture_id].upper()
        else:
            if random.random() < 0.7:
                self.target_gesture_id = 1
                self.target_name = "SWIPE"
            else:
                self.target_gesture_id = 3
                self.target_name = "MOVE"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_instruction_text(self) -> str:
        hand = self.required_hand
        if self.challenge_type == "STATIC":
            if self.target_name == "THUMBS UP":
                return f"Give a THUMBS UP to the camera with your {hand} hand"
            return f"Make the [{hand}] {self.target_name} gesture with your {hand} hand"
        if self.target_name == "SWIPE":
            return f"Point index finger — SWIPE fast horizontally with your {hand} hand"
        if self.target_name == "MOVE":
            return f"Move your entire {hand} hand across the screen"
        return f"Perform [{hand}] {self.target_name} motion"

    # ------------------------------------------------------------------
    # State-machine update
    # ------------------------------------------------------------------

    def update(
        self,
        static_gesture_id: Optional[int] = None,
        dynamic_gesture_id: Optional[int] = None,
        point_history: Optional[Sequence[Sequence[int]]] = None,
        palm_history: Optional[Sequence[Sequence[int]]] = None,
        landmark_list: Optional[list] = None,
        key_pressed: int = -1,
        detected_hand_label: Optional[str] = None,
    ) -> None:
        self.detected_hand = detected_hand_label
        now = time.time()
        elapsed = now - self.state_start_time

        # Feed trajectory histories every frame
        if point_history and len(point_history) > 0:
            self.swipe_history.append(point_history[-1])
        if palm_history and len(palm_history) > 0:
            self.palm_trail.append(palm_history[-1])

        # ---- IDLE -------------------------------------------------------
        if self.state == VerificationState.IDLE:
            if key_pressed in (ord("s"), ord("S"), 32):
                self.start_verification()
            return

        hand_matches = (
            detected_hand_label is not None
            and detected_hand_label.strip().upper() == self.required_hand
        )

        # ---- CHALLENGE (reading delay) ----------------------------------
        if self.state == VerificationState.CHALLENGE:
            if elapsed >= self.prep_time:
                self.state = VerificationState.VERIFYING
                self.state_start_time = now
                self.matched_gesture_start = None

        # ---- VERIFYING --------------------------------------------------
        elif self.state == VerificationState.VERIFYING:
            if elapsed >= self.time_limit:
                if self.current_attempt < self.max_attempts:
                    self.state = VerificationState.TRY_AGAIN
                    self.state_start_time = now
                    _beep(400, 300)
                else:
                    self.state = VerificationState.FAILED
                    self.state_start_time = now
                    _beep(300, 300)
                    threading.Timer(0.4, lambda: _beep(300, 300)).start()
                self.matched_gesture_start = None
                return

            if self.challenge_type == "STATIC":
                # Determine whether the gesture condition is met
                if self.target_name == "THUMBS UP":
                    gesture_ok = hand_matches and is_thumbs_up(landmark_list)
                else:
                    gesture_ok = (
                        hand_matches
                        and static_gesture_id is not None
                        and static_gesture_id == self.target_gesture_id
                    )

                if gesture_ok:
                    if self.matched_gesture_start is None:
                        self.matched_gesture_start = now
                    elif now - self.matched_gesture_start >= self.gesture_hold_time:
                        self.state = VerificationState.SUCCESS
                        self.state_start_time = now
                        _beep(1000, 200)
                else:
                    self.matched_gesture_start = None

            elif self.challenge_type == "DYNAMIC":
                self.matched_gesture_start = None
                if hand_matches:
                    pts = list(point_history) if point_history is not None else []
                    palms = list(palm_history) if palm_history is not None else []

                    if self.target_name == "SWIPE":
                        swipe_pts = list(self.swipe_history) or pts
                        if is_valid_swipe(swipe_pts):
                            self.state = VerificationState.SUCCESS
                            self.state_start_time = now
                            _beep(1000, 200)
                    elif self.target_name == "MOVE":
                        if is_valid_move(palms or list(self.palm_trail)):
                            self.state = VerificationState.SUCCESS
                            self.state_start_time = now
                            _beep(1000, 200)

        # ---- SUCCESS ----------------------------------------------------
        elif self.state == VerificationState.SUCCESS:
            if elapsed >= self.success_banner_time:
                self._return_to_idle()

        # ---- TRY_AGAIN --------------------------------------------------
        elif self.state == VerificationState.TRY_AGAIN:
            if elapsed >= self.try_again_banner_time:
                self.current_attempt += 1
                self._next_challenge()

        # ---- FAILED -----------------------------------------------------
        elif self.state == VerificationState.FAILED:
            if elapsed >= self.failed_banner_time:
                self._return_to_idle()

    # ------------------------------------------------------------------
    # Overlay drawing
    # ------------------------------------------------------------------

    def draw_overlay(self, image: np.ndarray) -> np.ndarray:
        annotated = image.copy()
        h, w = annotated.shape[:2]
        now = time.time()
        elapsed = now - self.state_start_time

        type_label = "STATIC POSE" if self.challenge_type == "STATIC" else "DYNAMIC MOTION"
        hand_tag = f"[{self.required_hand}]"
        instruction = self._get_instruction_text()
        formatted_target = f"{hand_tag} {self.target_name}"

        # ----------------------------------------------------------------
        # IDLE — welcome screen
        # ----------------------------------------------------------------
        if self.state == VerificationState.IDLE:
            overlay = annotated.copy()
            cv.rectangle(overlay, (0, 0), (w, 60), (18, 18, 28), -1)
            cv.rectangle(overlay, (0, h - 110), (w, h), (18, 18, 28), -1)
            cv.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)

            cv.putText(
                annotated, "HAND GESTURE VERIFICATION",
                (20, 38), cv.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv.LINE_AA,
            )
            cv.putText(
                annotated, "Press SPACE to begin",
                (w // 2 - 130, h - 78), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2, cv.LINE_AA,
            )
            cv.putText(
                annotated, "Static: OPEN | CLOSE | POINTER | OK | THUMBS UP",
                (20, h - 50), cv.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv.LINE_AA,
            )
            cv.putText(
                annotated, "Dynamic: SWIPE | MOVE",
                (20, h - 26), cv.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv.LINE_AA,
            )
            cv.putText(
                annotated, "Use the hand specified in [brackets]",
                (w - 360, h - 26), cv.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv.LINE_AA,
            )

        # ----------------------------------------------------------------
        # CHALLENGE — reading delay
        # ----------------------------------------------------------------
        elif self.state == VerificationState.CHALLENGE:
            overlay = annotated.copy()
            cv.rectangle(overlay, (0, 0), (w, 115), (30, 30, 30), -1)
            cv.rectangle(overlay, (0, h - 60), (w, h), (30, 30, 30), -1)
            cv.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

            prep_rem = max(0.0, self.prep_time - elapsed)
            cv.putText(
                annotated,
                f"CHALLENGE (Attempt {self.current_attempt}/{self.max_attempts}) — {type_label}",
                (20, 32), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA,
            )
            cv.putText(
                annotated, f"Target: {formatted_target}",
                (20, 68), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0, 215, 255), 2, cv.LINE_AA,
            )
            cv.putText(
                annotated, "Read the challenge above. Get ready!",
                (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv.LINE_AA,
            )
            cv.putText(
                annotated, f"Starting in {prep_rem:.1f}s...",
                (w // 2 - 130, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv.LINE_AA,
            )

        # ----------------------------------------------------------------
        # VERIFYING — active challenge
        # ----------------------------------------------------------------
        elif self.state == VerificationState.VERIFYING:
            overlay = annotated.copy()
            cv.rectangle(overlay, (0, 0), (w, 115), (20, 20, 20), -1)
            cv.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

            rem_time = max(0.0, self.time_limit - elapsed)

            cv.putText(
                annotated,
                f"VERIFYING — Attempt {self.current_attempt}/{self.max_attempts} [{type_label}]",
                (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA,
            )
            cv.putText(
                annotated, f"Target: {formatted_target}",
                (20, 65), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv.LINE_AA,
            )
            cv.putText(
                annotated, instruction,
                (20, 98), cv.FONT_HERSHEY_SIMPLEX, 0.52, (255, 215, 0), 1, cv.LINE_AA,
            )

            # Timer
            timer_color = (0, 255, 0) if rem_time > 2.0 else (0, 80, 255)
            cv.putText(
                annotated, f"Time: {rem_time:.1f}s",
                (w - 180, 45), cv.FONT_HERSHEY_SIMPLEX, 0.9, timer_color, 2, cv.LINE_AA,
            )

            # Hand-detection warnings
            if self.detected_hand is None:
                cv.putText(
                    annotated, "No hand detected! Show your hand to the camera.",
                    (20, h - 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv.LINE_AA,
                )
            elif self.detected_hand.upper() != self.required_hand:
                cv.putText(
                    annotated,
                    f"Wrong hand! Use your {self.required_hand} hand.",
                    (20, h - 60), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2, cv.LINE_AA,
                )

            # Static hold progress bar
            if self.challenge_type == "STATIC":
                bar_w, bar_h_px = 300, 20
                bar_x = w // 2 - bar_w // 2
                bar_y = h - 45

                if self.matched_gesture_start is not None:
                    hold_ratio = min(1.0, (now - self.matched_gesture_start) / self.gesture_hold_time)
                    cv.putText(
                        annotated, "Hold steady...",
                        (bar_x + 80, bar_y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA,
                    )
                    cv.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (50, 50, 50), -1)
                    cv.rectangle(
                        annotated,
                        (bar_x, bar_y),
                        (bar_x + int(bar_w * hold_ratio), bar_y + bar_h_px),
                        (0, 230, 0), -1,
                    )
                    cv.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (255, 255, 255), 2)
                else:
                    cv.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (50, 50, 50), -1)
                    cv.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (120, 120, 120), 1)

            # Dynamic trajectory trails
            if self.challenge_type == "DYNAMIC":
                if self.target_name == "SWIPE":
                    pts_list = [p for p in self.swipe_history if len(p) >= 2 and (p[0] != 0 or p[1] != 0)]
                    for i in range(1, len(pts_list)):
                        cv.line(annotated, tuple(map(int, pts_list[i - 1])), tuple(map(int, pts_list[i])), (255, 255, 0), 2, cv.LINE_AA)
                elif self.target_name == "MOVE":
                    palms_list = [p for p in self.palm_trail if len(p) >= 2 and (p[0] != 0 or p[1] != 0)]
                    for i in range(1, len(palms_list)):
                        cv.line(annotated, tuple(map(int, palms_list[i - 1])), tuple(map(int, palms_list[i])), (0, 165, 255), 2, cv.LINE_AA)

        # ----------------------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------------------
        elif self.state == VerificationState.SUCCESS:
            overlay = annotated.copy()
            cv.rectangle(overlay, (0, h // 2 - 70), (w, h // 2 + 70), (0, 140, 0), -1)
            cv.addWeighted(overlay, 0.72, annotated, 0.28, 0, annotated)
            cv.putText(
                annotated, "ACCESS GRANTED",
                (w // 2 - 210, h // 2 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv.LINE_AA,
            )
            cv.putText(
                annotated, "Verification Complete!",
                (w // 2 - 165, h // 2 + 38), cv.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2, cv.LINE_AA,
            )

        # ----------------------------------------------------------------
        # TRY_AGAIN
        # ----------------------------------------------------------------
        elif self.state == VerificationState.TRY_AGAIN:
            overlay = annotated.copy()
            cv.rectangle(overlay, (0, h // 2 - 70), (w, h // 2 + 70), (0, 110, 220), -1)
            cv.addWeighted(overlay, 0.72, annotated, 0.28, 0, annotated)
            cv.putText(
                annotated, "INCORRECT — TRY AGAIN",
                (w // 2 - 230, h // 2 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 3, cv.LINE_AA,
            )
            cv.putText(
                annotated,
                f"Attempt {self.current_attempt + 1} of {self.max_attempts} coming up...",
                (w // 2 - 200, h // 2 + 38), cv.FONT_HERSHEY_SIMPLEX, 0.72, (255, 230, 180), 2, cv.LINE_AA,
            )

        # ----------------------------------------------------------------
        # FAILED
        # ----------------------------------------------------------------
        elif self.state == VerificationState.FAILED:
            overlay = annotated.copy()
            cv.rectangle(overlay, (0, h // 2 - 70), (w, h // 2 + 70), (0, 0, 185), -1)
            cv.addWeighted(overlay, 0.72, annotated, 0.28, 0, annotated)
            cv.putText(
                annotated, "VERIFICATION FAILED",
                (w // 2 - 230, h // 2 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv.LINE_AA,
            )
            cv.putText(
                annotated, "All attempts exhausted. Press SPACE to retry.",
                (w // 2 - 290, h // 2 + 38), cv.FONT_HERSHEY_SIMPLEX, 0.62, (255, 180, 180), 2, cv.LINE_AA,
            )

        return annotated
