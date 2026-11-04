"""World axes and preview curves for the two selectable vertical slice families."""
from functools import lru_cache

import mujoco
import numpy as np

from arm5_scan import AXES, CENTER


@lru_cache(maxsize=64)
def guides(axis, position):
    offset = position/1000 - CENTER[axis]
    if abs(offset) >= AXES[axis]:
        return []
    scale = np.sqrt(1-(offset/AXES[axis])**2)
    angle = np.linspace(0, np.pi, 65)
    points = np.tile(CENTER,(len(angle),1))
    points[:,axis] += offset
    points[:,1-axis] += AXES[1-axis]*scale*np.cos(angle)
    points[:,2] += AXES[2]*scale*np.sin(angle) + .0003
    return [(a,b,.00062,[.95,.32,.04,1]) for a,b in zip(points,points[1:])]


def draw_coordinates(scene, lines, base=None, start=None):
    def geom(kind, pos, color):
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,kind,np.zeros(3),pos,np.eye(3).ravel(),np.asarray(color,np.float32))
        scene.ngeom += 1
        return g

    frames = [(np.zeros(3), np.eye(3), "+", .085)]
    if base is not None:
        frames.append((base.xpos, base.xmat.reshape(3,3), "B", .065))
    for origin, rotation, prefix, length in frames:
        for direction, color, label in zip(rotation.T, ([.88,.08,.06,1],[.05,.65,.16,1],[.12,.34,.95,1]), "XYZ"):
            if scene.ngeom+2 > scene.maxgeom:
                return
            end = origin + direction*length
            arrow = geom(mujoco.mjtGeom.mjGEOM_ARROW,origin,color)
            mujoco.mjv_makeConnector(arrow,mujoco.mjtGeom.mjGEOM_ARROW,.003,*origin,*end)
            geom(mujoco.mjtGeom.mjGEOM_LABEL,end+direction*.013,color).label = prefix+label
    if start is not None and scene.ngeom < scene.maxgeom:
        marker = geom(mujoco.mjtGeom.mjGEOM_SPHERE,start/1000,[.1,.9,.22,1])
        marker.size[0] = .0025
    for a,b,radius,color in lines:
        if scene.ngeom >= scene.maxgeom:
            break
        g = geom(mujoco.mjtGeom.mjGEOM_CAPSULE,a,color)
        mujoco.mjv_makeConnector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,radius,*a,*b)
