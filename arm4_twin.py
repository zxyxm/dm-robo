"""Mirror six measured motor angles into the Assembly 4 MuJoCo model."""
import argparse
from contextlib import ExitStack
import math
import sys
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from arm4_fk import load_model
from dm_debug import CanBus, Joint, U2Can, ids, move


def read_feedback(bus, joints):
    samples = []
    for idx, joint in enumerate(joints, 1):
        if not joint.limits:
            joint.probe(bus)
        status, pos, vel, torque, *_ = joint.status(bus)
        if status not in (0, 1):
            raise RuntimeError(f"J{idx} fault 0x{status:X}")
        samples.append((pos,vel,torque))
    return np.array(samples).T


def read_positions(bus, joints):
    return read_feedback(bus,joints)[0]


def update_pose(model, data, measured, zero, signs):
    data.qpos[:] = signs * (measured - zero)
    mujoco.mj_forward(model, data)
    site = data.site("flange")
    return site.xpos.copy() * 1000, Rotation.from_matrix(site.xmat.reshape(3, 3)).as_euler("xyz", degrees=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--channel", help="U2CAN COM port, PCAN device, or SocketCAN channel")
    parser.add_argument("--interface", choices=("u2can", "pcan", "socketcan"), default="u2can")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--bitrate", type=int, default=1000000)
    parser.add_argument("--ids", type=ids, default=list(range(1, 7)))
    parser.add_argument("--masters", type=ids, default=list(range(0x11, 0x17)))
    parser.add_argument("--zero-deg", nargs=6, type=float, help="encoder readings at STEP home, J1..J6")
    parser.add_argument("--signs", nargs=6, type=int, choices=(-1, 1), default=[1]*6)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, help="stop after this many complete updates")
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--jog", nargs=2, metavar=("AXIS", "DEG"))
    control.add_argument("--move6", nargs=6, type=float, metavar="DEG")
    args = parser.parse_args()
    active, offsets = [], []
    if args.jog or args.move6:
        if not args.channel or args.frames:
            parser.error("motion requires --channel and cannot be combined with --frames")
        if args.jog:
            try:
                active, offsets = [int(args.jog[0])-1], [float(args.jog[1])]
            except ValueError:
                parser.error("expected --jog AXIS DEGREES")
        else:
            active, offsets = list(range(6)), args.move6
        if not all(0 <= i < 6 for i in active) or not all(math.isfinite(x) and abs(x) <= 5 for x in offsets):
            parser.error("expected axis 1..6 and motor increments within +/-5 degrees")
    if args.channel and args.zero_deg is None:
        parser.error("supply --zero-deg with all six encoder readings at the saved STEP pose")
    if set(args.ids) & set(args.masters):
        parser.error("command and feedback IDs must not overlap")
    zero = np.radians(args.zero_deg or [0]*6)
    if not np.isfinite(zero).all() or (args.frames is not None and args.frames < 1):
        parser.error("invalid zero or frame count")

    model = load_model()
    data = mujoco.MjData(model)
    joints = [Joint(s, m) for s, m in zip(args.ids, args.masters)]
    signs = np.array(args.signs)
    with ExitStack() as stack:
        bus = viewer = None
        if args.channel:
            bus = (U2Can(args.channel, args.baud) if args.interface == "u2can"
                   else CanBus(args.interface, args.channel, args.bitrate))
            stack.callback(bus.close)
        # Require a complete first hardware sample before opening a window.
        first = read_positions(bus, joints) if bus else np.zeros(6)
        update_pose(model, data, first, zero, signs)
        if not args.headless:
            from mujoco import viewer as mjviewer
            viewer = stack.enter_context(mjviewer.launch_passive(model, data))
            viewer.cam.lookat[:] = [0, 0, .18]
            viewer.cam.distance = .78
            viewer.cam.elevation = -18
            viewer.cam.azimuth = 140
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        if active:
            def mirror(feedback):
                if viewer and not viewer.is_running():
                    raise RuntimeError("viewer closed")
                # Reuse motion feedback; poll uncommanded joints on the same connection.
                measured = [feedback[j.slave] if j.slave in feedback else read_positions(bus, [j])[0]
                            for j in joints]
                xyz, rpy = update_pose(model, data, np.array(measured), zero, signs)
                if viewer:
                    viewer.sync()
                print(f"XYZ(mm)={xyz.round(2)} RPY(deg)={rpy.round(2)}", flush=True)

            move(bus, [joints[i] for i in active], np.radians(offsets), on_sample=mirror)
            return
        start = time.monotonic()
        last_print = -math.inf
        count = 0
        print("DEMO | synthetic angles" if args.demo else "LIVE | read-only CAN feedback", flush=True)
        while viewer is None or viewer.is_running():
            tick = time.monotonic()
            elapsed = tick - start
            if count == 0:
                measured = first
            elif bus:
                measured = read_positions(bus, joints)
            else:
                measured = np.radians([28, 18, 25, 20, 18, 40]) * np.sin(elapsed * np.array([.6,.42,.55,.73,.8,.95]))
            xyz, rpy = update_pose(model, data, measured, zero, signs)
            if viewer:
                viewer.sync()
            count += 1
            if elapsed - last_print >= .25 or count == args.frames:
                print(f"t={elapsed:.2f} J(deg)={np.round(np.degrees(data.qpos), 1)} "
                      f"XYZ(mm)={xyz.round(2)} RPY(deg)={rpy.round(2)}", flush=True)
                last_print = elapsed
            if count == args.frames:
                break
            time.sleep(max(0, .04 - (time.monotonic() - tick)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        sys.exit(f"Twin stopped: {exc}")
