"""Protocol-level six-motor simulator for regression tests (no physical bus)."""
import math
import struct
import unittest

from dm_debug import Joint, move


class SimBus:
    def __init__(self, fail_axis=None):
        self.frames = []
        self.enabled = set()
        self.pos = dict.fromkeys(range(1, 7), 32768)
        self.fail_axis = fail_axis
        self.disabled = []

    def send(self, ident, payload):
        if ident == 0x7ff:
            axis, op = struct.unpack_from("<HB", payload)
            if op == 0x33:
                rid = payload[3]
                value = {10: 1, 21: 12.5, 22: 30., 23: 10.}[rid]
                reply = payload[:4] + struct.pack("<I" if rid == 10 else "<f", value)
                self.frames.append((0x10 + axis, reply))
                return
            assert op == 0xcc
        else:
            axis = ident
            if payload == b'\xff' * 7 + b'\xfc':
                self.enabled.add(axis)
            elif payload == b'\xff' * 7 + b'\xfd':
                self.enabled.discard(axis)
                self.disabled.append(axis)
            elif axis in self.enabled:
                if axis == self.fail_axis:
                    raise OSError("simulated link loss")
                self.pos[axis] = int.from_bytes(payload[:2], "big")
        response = bytes([axis | (16 if axis in self.enabled else 0)])
        response += self.pos[axis].to_bytes(2, "big") + bytes.fromhex("80 08 00 19 1a")
        self.frames.append((0x10 + axis, response))

    def recv(self, timeout):
        return self.frames.pop(0) if self.frames else None


class SixAxisTest(unittest.TestCase):
    def setUp(self):
        self.joints = [Joint(i, 16 + i) for i in range(1, 7)]

    def test_six_motor_jog(self):
        bus = SimBus()
        offsets = [math.radians(i / 10) for i in range(1, 7)]
        move(bus, self.joints, offsets)
        self.assertFalse(bus.enabled)
        for motor, offset in zip(self.joints, offsets):
            self.assertLess(abs(motor.status(bus)[1] - offset), .001)

    def test_partial_failure_disables_all(self):
        bus = SimBus(fail_axis=4)
        with self.assertRaises(OSError):
            move(bus, self.joints, [0.] * 6)
        self.assertEqual(sorted(bus.disabled), list(range(1, 7)))
        self.assertFalse(bus.enabled)


if __name__ == "__main__":
    unittest.main()
