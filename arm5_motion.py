"""Time-parameterized joint playback for the ultrasound simulator (rad, seconds)."""
import numpy as np
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
