"""Regression check of every shipped Assembly 5 playback pose."""
import unittest

import numpy as np

from arm5_scan import ROOT, Workspace, load_model, target_pose


class ScanTest(unittest.TestCase):
    def test_playback_points_contact_and_clearance(self):
        work = Workspace(load_model(ROOT / "arm5.xml"))
        rows = np.genfromtxt(ROOT / "arm5_passes.csv", delimiter=",", skip_header=1)
        previous = None
        for row in rows:
            y, lo, hi, angle = row[:4]
            q = np.radians(row[4:])
            self.assertTrue(any(work.feasible(q, *target_pose(y-250, angle, sign)) for sign in (1,-1)))
            if previous is not None and np.array_equal(row[:3], previous[:3]):
                self.assertLess(np.max(np.abs(row[4:]-previous[4:])),20)
            previous = row


if __name__ == "__main__":
    unittest.main()
