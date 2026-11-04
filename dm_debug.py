"""Six-axis Damiao commissioning CLI. Commands are relative to live motor angles."""
import argparse
from contextlib import closing
from dataclasses import dataclass
import math
import struct
import sys
import time


class U2Can:
    """Damiao U2CAN serial protocol (classic CAN, eight-byte payloads)."""
    def __init__(self, channel, baud):
        import serial
        self.port = serial.Serial(channel, baud, timeout=0.005, write_timeout=0.1)
        self.buf = bytearray()
        self.port.reset_input_buffer()

    def send(self, ident, data):
        frame = bytearray.fromhex("55 aa 1e 03 01 00 00 00 0a 00 00 00 00 00 00 00 00 00 08 00 00")
        frame.extend(data + b'\0')
        struct.pack_into("<I", frame, 13, ident)
        self.port.write(frame)

    def recv(self, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            while len(self.buf) >= 16:
                if self.buf[0] != 0xaa or self.buf[15] != 0x55:
                    del self.buf[0]
                    continue
                packet = self.buf[:16]
                del self.buf[:16]
                if packet[1] == 0x11:
                    return int.from_bytes(packet[3:7], "little"), bytes(packet[7:15])
            self.buf.extend(self.port.read(min(1024, self.port.in_waiting or 1)))
        return None

    def close(self):
        self.port.close()


class CanBus:
    def __init__(self, interface, channel, bitrate):
        import can
        self.can = can
        self.bus = can.Bus(interface=interface, channel=channel, bitrate=bitrate)

    def send(self, ident, data):
        self.bus.send(self.can.Message(arbitration_id=ident, data=data, is_extended_id=False), timeout=0.1)

    def recv(self, timeout):
        msg = self.bus.recv(timeout)
        if msg and not (msg.is_extended_id or msg.is_error_frame or msg.is_remote_frame) and msg.dlc == 8:
            return msg.arbitration_id, bytes(msg.data)
        return None

    def close(self):
        self.bus.shutdown()


@dataclass
class Joint:
    slave: int
    master: int
    limits: tuple = ()

    def exchange(self, bus, txid, data, match):
        bus.send(txid, data)
        deadline = time.monotonic() + 0.15
        while (remaining := deadline - time.monotonic()) > 0:
            msg = bus.recv(remaining)
            if msg and msg[0] == self.master and match(msg[1]):
                return msg[1]
        raise TimeoutError(f"CAN 0x{self.slave:03X}: feedback timeout")

    def parameter(self, bus, rid):
        prefix = struct.pack("<HBB", self.slave, 0x33, rid)
        reply = self.exchange(bus, 0x7ff, prefix + bytes(4), lambda d: d.startswith(prefix))
        return struct.unpack_from("<I" if rid == 10 else "<f", reply, 4)[0]

    def probe(self, bus):
        self.limits = tuple(self.parameter(bus, rid) for rid in (21, 22, 23))
        if not all(math.isfinite(x) and x > 0 for x in self.limits):
            raise ValueError(f"Invalid MIT ranges: {self.limits}")

    def status(self, bus, command=None):
        txid = self.slave
        if command is None:
            txid = 0x7ff
            command = struct.pack("<HB5x", self.slave, 0xcc)
        d = self.exchange(bus, txid, command, lambda d: d[0] & 15 == self.slave & 15)
        raw = (int.from_bytes(d[1:3], "big"), d[3] << 4 | d[4] >> 4, (d[4] & 15) << 8 | d[5])
        p, v, t = [(2 * n / top - 1) * limit for n, top, limit in zip(raw, (65535, 4095, 4095), self.limits)]
        return d[0] >> 4, p, v, t, d[6], d[7]

    def target(self, angle):
        pmax = self.limits[0]
        if not -pmax <= angle <= pmax:
            raise ValueError("Target exceeds motor PMAX")
        pos = round((angle / pmax + 1) * 65535 / 2)
        # MIT: velocity=0, Kp=3, Kd=0.15, feedforward torque=0.
        vel, kp, kd, tau = 2048, round(3 / 500 * 4095), round(.15 / 5 * 4095), 2048
        return ((pos << 48) | (vel << 36) | (kp << 24) | (kd << 12) | tau).to_bytes(8, "big")


def display(index, state):
    code, pos, vel, torque, mos, rotor = state
    print(f"J{index} status=0x{code:X}  {math.degrees(pos):8.2f} deg  {vel:7.2f} rad/s  "
          f"{torque:7.2f} Nm  MOS={mos} C rotor={rotor} C")


def move(bus, joints, offsets):
    initial = []
    owned = []
    try:
        for motor, delta in zip(joints, offsets):
            motor.probe(bus)
            if motor.parameter(bus, 10) != 1:
                raise ValueError(f"0x{motor.slave:X}: MIT mode required")
            state = motor.status(bus)
            if state[0] != 0:
                raise ValueError(f"0x{motor.slave:X}: start disabled and fault-free (status {state[0]:X})")
            initial.append(state[1])
            motor.target(state[1] + delta)
        for motor, pos in zip(joints, initial):
            owned.append(motor)
            motor.status(bus, motor.target(pos))
            bus.send(motor.slave, b'\xff' * 7 + b'\xfc')
            if not any(motor.status(bus)[0] == 1 for _ in range(3)):
                raise RuntimeError(f"0x{motor.slave:X}: enable not confirmed")
        ramp = max(.5, max(map(abs, offsets)) / math.radians(3))
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            fraction = min(1., elapsed / ramp)
            for motor, pos, delta in zip(joints, initial, offsets):
                target = pos + fraction * delta
                state = motor.status(bus, motor.target(target))
                if state[0] != 1 or abs(state[1] - target) > math.radians(8):
                    raise RuntimeError(f"0x{motor.slave:X}: fault or excessive tracking error")
                if elapsed >= ramp + .5:
                    print(f"0x{motor.slave:X}: target={math.degrees(target):.2f} measured={math.degrees(state[1]):.2f} deg")
            if elapsed >= ramp + .5:
                break
            time.sleep(.015)
    finally:
        failures = []
        for motor in reversed(owned):
            try:
                bus.send(motor.slave, b'\xff' * 7 + b'\xfd')
            except Exception as exc:
                failures.append(str(exc))
        for motor in owned:
            try:
                if not any(motor.status(bus)[0] == 0 for _ in range(3)):
                    failures.append(f"0x{motor.slave:X} still enabled or faulted")
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError("Disable unconfirmed: " + ", ".join(failures))


def ids(value):
    try:
        values = [int(v, 0) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected six comma-separated CAN IDs") from exc
    if len(values) != 6 or len(set(values)) != 6 or not all(0 < v < 0x7ff for v in values):
        raise argparse.ArgumentTypeError("expected six distinct nonzero standard IDs below 0x7ff")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", choices=["u2can", "pcan", "socketcan"], default="u2can")
    parser.add_argument("--channel", required=True)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--bitrate", type=int, default=1000000)
    parser.add_argument("--ids", type=ids, default=list(range(1, 7)))
    parser.add_argument("--masters", type=ids, default=list(range(0x11, 0x17)))
    cmds = parser.add_subparsers(dest="cmd", required=True)
    watch = cmds.add_parser("monitor")
    watch.add_argument("--once", action="store_true")
    jog = cmds.add_parser("jog")
    jog.add_argument("axis", type=int, choices=range(1, 7))
    jog.add_argument("degrees", type=float)
    group = cmds.add_parser("move6")
    group.add_argument("degrees", type=float, nargs=6)
    args = parser.parse_args()
    if set(args.ids) & set(args.masters):
        parser.error("command and feedback IDs must not overlap")
    joints = [Joint(s, m) for s, m in zip(args.ids, args.masters)]
    offsets = []
    if args.cmd != "monitor":
        offsets = args.degrees if args.cmd == "move6" else [args.degrees]
        if not all(math.isfinite(v) and abs(v) <= 5 for v in offsets):
            parser.error("relative motion limited to +/-5 degrees per command")
        if args.cmd == "jog":
            joints = [joints[args.axis - 1]]
    link = U2Can(args.channel, args.baud) if args.interface == "u2can" else CanBus(args.interface, args.channel, args.bitrate)
    with closing(link) as bus:
        if offsets:
            move(bus, joints, list(map(math.radians, offsets)))
            return
        while True:
            for i, motor in enumerate(joints, 1):
                try:
                    if not motor.limits:
                        motor.probe(bus)
                    display(i, motor.status(bus))
                except TimeoutError as exc:
                    motor.limits = ()
                    print(f"J{i} OFFLINE: {exc}", file=sys.stderr)
            if args.once:
                return
            time.sleep(.15)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        sys.exit(str(exc))
