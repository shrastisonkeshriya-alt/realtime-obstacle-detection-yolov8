"""
Decision engine: converts left/right danger scores into a stable
movement command, using hysteresis so a decision only changes after
several consecutive frames agree.
"""

from config import CFG
from robot import RobotController


class DecisionEngine:
    """
    Wraps the raw left/right-danger decision in a small state machine so a
    decision only changes after CFG.HYSTERESIS_FRAMES consecutive frames
    agree. This fixes 'robot twitches left-right-left because one frame had
    a false detection'. The cost is a small reaction delay (a few frames,
    <200ms), which is a worthwhile trade for stability.
    """
    def __init__(self):
        self._candidate = None
        self._candidate_count = 0
        self._current = "PATH CLEAR → FORWARD"

    def _raw_decision(self, left_danger: float, right_danger: float) -> str:
        L = left_danger  > CFG.DANGER_THRESHOLD
        R = right_danger > CFG.DANGER_THRESHOLD

        if L and R:
            if left_danger > right_danger * 1.3:
                return "BOTH BLOCKED · BIAS LEFT → TURN RIGHT"
            elif right_danger > left_danger * 1.3:
                return "BOTH BLOCKED · BIAS RIGHT → TURN LEFT"
            else:
                return "BOTH BLOCKED → BACKWARD"
        elif L:
            return "LEFT BLOCKED → TURN RIGHT"
        elif R:
            return "RIGHT BLOCKED → TURN LEFT"
        else:
            return "PATH CLEAR → FORWARD"

    def decide(self, robot: RobotController, left_danger: float, right_danger: float) -> str:
        raw = self._raw_decision(left_danger, right_danger)

        if raw == self._current:
            self._candidate = None
            self._candidate_count = 0
        elif raw == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = raw
            self._candidate_count = 1

        if self._candidate is not None and self._candidate_count >= CFG.HYSTERESIS_FRAMES:
            self._current = self._candidate
            self._candidate = None
            self._candidate_count = 0

        # Dispatch robot command for whatever the CURRENT stable decision is
        if "TURN RIGHT" in self._current:
            robot.turn_right()
        elif "TURN LEFT" in self._current:
            robot.turn_left()
        elif "BACKWARD" in self._current:
            robot.backward()
        else:
            robot.forward()

        return self._current
