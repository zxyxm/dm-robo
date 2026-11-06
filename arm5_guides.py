"""World axes and preview curves for the two selectable vertical slice families."""
from functools import lru_cache

import mujoco
import numpy as np

from arm5_scan import AXES, CENTER


# Fixed top corner in MuJoCo world coordinates (metres).
BOX_ORIGIN = np.array([0., .5, .3])


def box_mm(world):
    return (np.asarray(world)-BOX_ORIGIN)*1000


@lru_cache(maxsize=64)
def guides(axis, position, radii=tuple(AXES),center=tuple(CENTER)):
    offset = position/1000 - CENTER[axis]
    if abs(offset) >= radii[axis]:
        return []
    scale = np.sqrt(1-(offset/radii[axis])**2)
    angle = np.linspace(0, np.pi, 65)
    points = np.tile(center,(len(angle),1))
    points[:,axis] += offset
    points[:,1-axis] += radii[1-axis]*scale*np.cos(angle)
    points[:,2] += radii[2]*scale*np.sin(angle) + .0003
    return [(a,b,.00062,[.95,.32,.04,1]) for a,b in zip(points,points[1:])]


def draw_coordinates(scene, lines, base=None, start=None, flange=None):
    def geom(kind, pos, color):
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,kind,np.zeros(3),pos,np.eye(3).ravel(),np.asarray(color,np.float32))
        scene.ngeom += 1
        return g

    frames = [(BOX_ORIGIN, np.eye(3), "+", .085)]
    if base is not None:
        frames.append((base.xpos, base.xmat.reshape(3,3), "B", .065))
    if flange is not None:
        frames.append((flange.xpos, flange.xmat.reshape(3,3), "T", .080))
        # Parallel reference beside the tool, visible even in the close-up view.
        origin = flange.xpos + np.array([0.,0.,.095])
        end = origin + np.array([.065,0.,0.])
        if scene.ngeom+2 <= scene.maxgeom:
            ref = geom(mujoco.mjtGeom.mjGEOM_ARROW,origin,[1.,.53,.05,1])
            mujoco.mjv_makeConnector(ref,mujoco.mjtGeom.mjGEOM_ARROW,.0025,*origin,*end)
            geom(mujoco.mjtGeom.mjGEOM_LABEL,end+[.013,0.,0.],[1.,.53,.05,1]).label = "World +X"
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
