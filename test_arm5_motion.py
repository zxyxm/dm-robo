import unittest

import numpy as np

from arm5_motion import ACCEL, SPEED, motion_frames


class ScanMotionTest(unittest.TestCase):
    def test_wrap_and_axis_limits(self):
        q = np.radians([[178, 15, 0, 0, 0, 0], [-178, 12, 3, 0, 0, 0],
                        [-162, -21, 8, 10, 23, 32], [-166, -19, 3, 12, 24, 22]])
        frames = np.vstack([q[0], list(motion_frames(q))])
        v = np.diff(frames, axis=0) / .025
        a = np.diff(v, axis=0) / .025
        self.assertTrue(np.all(abs(v) <= SPEED + 1e-9))
        self.assertTrue(np.all(abs(a) <= ACCEL + 1e-9))
        self.assertTrue(np.all(frames[:,0] > np.pi/2))
        np.testing.assert_allclose(frames[-1], np.unwrap(q,axis=0)[-1])
