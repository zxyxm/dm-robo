"""Cartesian figure-eight writing path in the board's X/Z plane."""
import mujoco
import numpy as np


def pen_trajectory(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tip = data.site("pen_tip")
    home = tip.xpos.copy()
    board = model.geom("writing_board")
    center = board.pos.copy()
    center[1] += board.size[1]
    lifted = center + [0, .012, 0]
    theta = np.linspace(0, 2*np.pi, 161)
    curve = center + np.column_stack((.021*np.sin(2*theta), np.zeros_like(theta), .036*np.sin(theta)))
    phases = [(np.linspace(home, lifted, 16), False), (np.linspace(lifted, center, 14), False),
              (curve, True), (np.linspace(center, lifted, 14), False)]
    jac = np.zeros((3, model.nv))
    for targets, ink in phases:
        for target in targets:
            for _ in range(20):
                error = target - tip.xpos
                if np.linalg.norm(error) < .00002:
                    break
                mujoco.mj_jacSite(model, data, jac, None, tip.id)
                delta = jac.T @ np.linalg.solve(jac @ jac.T + .00001*np.eye(3), error)
                data.qpos[:] += np.clip(delta, -.06, .06)
                mujoco.mj_forward(model, data)
            if np.linalg.norm(target - tip.xpos) > .00005:
                raise RuntimeError("笔尖未到达目标点")
            yield data.qpos.copy(), ink


def draw_ink(scene, points):
    scene.ngeom = 0
    for a, b in zip(points, points[1:]):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                           np.zeros(3), np.eye(3).ravel(), np.array([.05,.15,.40,1], np.float32))
        # Offset the ink slightly toward the writing side to avoid z-fighting.
        off = np.array([0, .0004, 0])
        mujoco.mjv_makeConnector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .001, *(a+off), *(b+off))
        scene.ngeom += 1
