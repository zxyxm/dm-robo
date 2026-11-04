import unittest
from unittest.mock import patch

import numpy as np

import arm4_twin
from arm4_fk import reference_fk
from test_dm_debug import SimBus


class TwinTest(unittest.TestCase):
    def test_move6_with_feedback_calibration(self):
        bus = SimBus()
        bus.close = lambda: None
        samples = []
        original = arm4_twin.update_pose

        def record(model, data, measured, zero, signs):
            result = original(model, data, measured, zero, signs)
            expected = reference_fk(signs * (measured - zero))
            np.testing.assert_allclose(result[0], 1000 * expected[:3, 3], atol=1e-7)
            samples.append(data.qpos.copy())
            return result

        args = ["twin", "--channel", "sim", "--headless", "--zero-deg", *["3"]*6,
                "--signs", "-1", "1", "1", "1", "1", "1", "--move6", *["1"]*6]
        with patch("sys.argv", args), patch.object(arm4_twin, "U2Can", return_value=bus),              patch.object(arm4_twin, "update_pose", side_effect=record), patch("builtins.print"):
            arm4_twin.main()
        self.assertFalse(bus.enabled)
        self.assertGreater(len(samples), 2)
        np.testing.assert_allclose(np.degrees(samples[-1]), [2,-2,-2,-2,-2,-2], atol=.04)


if __name__ == "__main__":
    unittest.main()
