"""Quasi-static probe force in box coordinates (environment acting on the probe)."""
import mujoco
import numpy as np

from arm5_scan import AXES, CENTER


def jacobians(model,data):
    linear = np.empty((3,model.nv))
    angular = np.empty_like(linear)
    mujoco.mj_jacSite(model,data,linear,angular,data.site("acoustic").id)
    return linear,angular


def estimate_force(linear, gravity, torque, bias):
    # Equilibrium: motor + J.T @ environment_force = gravity.
    force,_,rank,s = np.linalg.lstsq(linear.T,gravity-torque-bias,rcond=1e-3)
    if rank < 3 or s[-1] < .003 or s[0]/s[-1] > 80:
        return None
    return force


def contact(point,radii=AXES,center=CENTER):
    unit = (point-center)/radii
    gradient = unit/radii
    length = np.linalg.norm(gradient)
    if point[2] < center[2] or length < 1e-8:
        return np.zeros(3),0.
    # Local signed distance from the ellipsoid implicit surface, linearized at TCP.
    depth = max(0., (1-unit@unit)/(2*length))
    return gradient/length, 1500.*depth


class ProbeControl:
    def __init__(self):
        self.offset = 0.
        self.previous_force = 0.

    def correct(self,model,data,nominal,target,dt):
        data.qpos[:] = nominal
        mujoco.mj_forward(model,data)
        radii = AXES * model.mesh_scale[model.mesh("phantom").id]
        normal,_ = contact(data.site("acoustic").xpos,radii,model.site("phantom_center").pos)
        speed = np.clip(.0015*(target-self.previous_force),-.003,.003)
        self.offset = np.clip(self.offset+speed*dt,0.,.004)
        goal = data.site("acoustic").xpos.copy()-normal*self.offset
        for _ in range(3):
            linear,angular = jacobians(model,data)
            jac = np.vstack((linear,.1*angular))
            error = np.r_[goal-data.site("acoustic").xpos,np.zeros(3)]
            delta = jac.T @ np.linalg.solve(jac@jac.T + 1e-6*np.eye(6),error)
            data.qpos[:] += np.clip(delta,-.005,.005)
            mujoco.mj_forward(model,data)
        _,self.previous_force = contact(data.site("acoustic").xpos,radii,model.site("phantom_center").pos)
        return data.qpos.copy()
