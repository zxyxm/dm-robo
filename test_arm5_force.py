import unittest

import mujoco
import numpy as np

from arm5_force import ProbeControl, estimate_force, jacobians
from arm5_scan import ROOT, load_model, load_passes


class ForceTest(unittest.TestCase):
    def test_contact_loop_and_motor_force_sign(self):
        model = load_model(ROOT/"arm5.xml")
        data = mujoco.MjData(model)
        path = next(v for k,v in load_passes().items() if k.startswith("Y=250"))
        q = path[len(path)//2]
        control = ProbeControl()
        for _ in range(150):
            control.correct(model,data,q,2.,.025)
        self.assertAlmostEqual(control.previous_force,2.,places=2)
        j,_ = jacobians(model,data)
        f = np.array([1.,-2.,3.])
        bias = np.arange(6)*.02
        tau = data.qfrc_bias-j.T@f-bias
        np.testing.assert_allclose(estimate_force(j,data.qfrc_bias,tau,bias),f)
        self.assertIsNone(estimate_force(j*0,data.qfrc_bias,tau,bias))
