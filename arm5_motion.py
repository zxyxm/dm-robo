"""Time-parameterized joint playback for the ultrasound simulator (rad, seconds)."""
import numpy as np
import mujoco
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

# Comfortable simulation limits for the loaded shoulder and faster wrist axes.
SPEED = np.radians([24, 18, 22, 32, 38, 42])
ACCEL = np.radians([60, 45, 55, 80, 95, 105])


def motion_frames(poses, dt=.025):
    points = np.unwrap(np.asarray(poses), axis=0)
    points = points[np.r_[True, np.any(abs(np.diff(points, axis=0)) > 1e-9, axis=1)]]
    if len(points) == 1:
        yield points[0]
        return
    delta = np.abs(np.diff(points, axis=0))
    duration = np.maximum(.065, np.max(delta / SPEED, axis=1))
    knots = np.r_[0., np.cumsum(duration)]
    slopes = PchipInterpolator(knots, points).derivative()(knots)
    slopes[[0, -1]] = 0
    path = CubicHermiteSpline(knots, points, slopes)
    # Per-interval quadratic velocity has extrema at ends or at zero acceleration.
    a, b, c = path.c[:3]
    interior = np.clip(np.divide(-b, 3*a, out=np.zeros_like(b), where=a!=0), 0, duration[:,None])
    peak_v = np.maximum(abs(slopes).max(axis=0), abs((3*a*interior+2*b)*interior+c).max(axis=0))
    peak_a = np.maximum(abs(2*b),abs(6*a*duration[:,None]+2*b)).max(axis=0)
    stretch = max(1., np.max(peak_v / SPEED), np.sqrt(np.max(peak_a / ACCEL)))
    frames = int(np.ceil(knots[-1]*stretch/dt))
    for i in range(1, frames+1):
        yield path(knots[-1]*i/frames)


def ceiling_park_frames(work,initial,stop):
    """Lift clear of the surface, then fold along the +Y ceiling edge."""
    def clear(q):
        work.d.qpos[:] = q
        mujoco.mj_kinematics(work.m,work.d)
        return work.clearance_ok()

    def nearby(point,frame,reference):
        poses = [reference+(q-reference+np.pi)%(2*np.pi)-np.pi
                 for q in work.solutions(point,frame)]
        return sorted(poses,key=lambda q:np.linalg.norm(q-reference))

    clear(initial)
    site = work.d.site("acoustic")
    point = site.xpos.copy()
    frame = site.xmat.reshape(3,3).copy()
    lifts = []
    for q in nearby(point+[0,0,.035],frame,initial):
        if np.max(abs(q-initial))<1. and clear(q):
            lifts.append(q)
            break
    lifts.append(initial)
    # Explore folded wrist orientations as well as elbow and shoulder branches.
    rng = np.random.default_rng(517)
    candidates = []
    for _ in range(35000):
        if stop.is_set():
            return []
        q = initial + rng.uniform(-np.pi,np.pi,6)
        work.d.qpos[:] = q
        mujoco.mj_kinematics(work.m,work.d)
        tip = site.xpos
        if tip[1]<.35 or not .18<tip[2]<.29:
            continue
        if clear(q):
            # Maximize clearance below the entire arm, not just the probe.
            bottom = min(np.min(vertices @ work.d.geom_xmat[i].reshape(3,3)[2]
                                + work.d.geom_xpos[i,2]) for i,vertices in work.hulls.items())
            candidates.append((bottom,q))
    candidates.sort(key=lambda item:item[0],reverse=True)
    for lift in lifts:
        for _,q in candidates:
            if stop.is_set():
                return []
            goal = lift+(q-lift+np.pi)%(2*np.pi)-np.pi
            if not all(clear(v) for v in np.linspace(lift,goal,24)):
                continue
            frames = list(motion_frames([initial,lift,goal]))
            if all(not stop.is_set() and clear(v) for v in frames):
                return frames
    return []
