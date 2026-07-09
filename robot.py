"""
Robot controller: sends movement commands over serial, or simulates
them (prints only) when no port is given — so the pipeline can be tested
with just a webcam, no hardware required.
"""

import sys
import time
import collections
from typing import Optional

from config import CFG

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class RobotController:
    def __init__(self, port: Optional[str] = None):
        self.port        = port
        self.serial_conn = None
        self.simulated   = port is None
        self._last_cmd   = None
        self._last_time  = 0.0
        self.cmd_log: collections.deque = collections.deque(maxlen=8)

        if not self.simulated:
            if not SERIAL_AVAILABLE:
                print("[ERROR] Install pyserial:  pip install pyserial")
                sys.exit(1)
            try:
                self.serial_conn = serial.Serial(
                    port, CFG.BAUD_RATE, timeout=CFG.SERIAL_TIMEOUT
                )
                time.sleep(2)
                print(f"[ROBOT] Connected → {port} @ {CFG.BAUD_RATE} baud")
            except serial.SerialException as e:
                print(f"[ERROR] Cannot open {port}: {e}")
                sys.exit(1)
        else:
            print("[ROBOT] SIMULATION mode (no serial port)")

    def send(self, command: bytes, label: str):
        now = time.time()
        if command == self._last_cmd and (now - self._last_time) < CFG.COMMAND_COOLDOWN:
            return
        self._last_cmd  = command
        self._last_time = now
        self.cmd_log.append((time.strftime("%H:%M:%S"), label))

        if self.simulated:
            print(f"[CMD] {label}")
        else:
            self.serial_conn.write(command)
            print(f"[CMD → {self.port}] {label}")

    def forward(self):    self.send(CFG.CMD_FORWARD,   "▲  FORWARD")
    def backward(self):   self.send(CFG.CMD_BACKWARD,  "▼  BACKWARD")
    def turn_left(self):  self.send(CFG.CMD_LEFT,       "◀  TURN LEFT")
    def turn_right(self): self.send(CFG.CMD_RIGHT,      "▶  TURN RIGHT")
    def stop(self):       self.send(CFG.CMD_STOP,        "■  STOP")

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[ROBOT] Serial closed")
