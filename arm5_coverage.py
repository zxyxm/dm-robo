"""Sample pose reachability independently of the scan start or force setpoint."""
import numpy as np
import mujoco
from copy import copy
from itertools import product

from arm5_scan import CENTER, Workspace, section_pose

BLUE = np.array([.04,.40,.95,1],dtype=np.float32)


def sample_section(work, axis, position, step, stop):
    fixed = (position/1000-CENTER[axis])/work.radii[axis]
    samples = []
    for angle in range(0,181,step):
        if stop.is_set():
            return []
        point,frame = section_pose(axis,fixed,angle,work.radii,work.center)
        reachable = axis == 0 and any(work.feasible(q,point,frame)
                                      for q in work.solutions(point,frame))
        normal = (point-work.center)/work.radii**2
        samples.append((point+.00045*normal/np.linalg.norm(normal),reachable))
    return samples


def triangle(a,b,c,center=CENTER):
    u,v = b-a,c-a
    n = np.cross(u,v)
    if n@(a-center)<0:
        u,v,n = v,u,-n
    lengths = np.array([np.linalg.norm(u),np.linalg.norm(v),np.linalg.norm(n)])
    matrix = np.column_stack((u,v,n))/lengths
    return lengths,a,matrix.ravel()


def solve_area(model,axis,stop,progress=None):
    work = Workspace(model)
    middle = CENTER[axis]*1000
    radius = work.radii[axis]*1000
    previous = []
    patches = []
    step = min(4.,radius/4)
    offsets = np.arange(-radius+step,radius-step/2,step)
    for done,offset in enumerate(offsets,1):
        row = sample_section(work,axis,middle+offset,6,stop)
        if stop.is_set():
            return []
        for a,b,c,d in zip(previous,previous[1:],row,row[1:]):
            if all(sample[1] for sample in (a,b,c,d)):
                patches.extend((triangle(a[0],b[0],c[0],work.center),triangle(b[0],d[0],c[0],work.center)))
        previous = row
        if progress is not None:
            progress(done,len(offsets))
    return patches


def draw_area(scene,patches):
    for size,pos,mat in patches:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_TRIANGLE,
                            size,pos,mat,BLUE)
        scene.ngeom += 1


def optimize_mount(model,stop,progress):
    """Compare equal-resolution YZ surface areas; prefer the current mount on ties."""
    trial = copy(model)
    mount = trial.body("mount").pos
    current = tuple(mount[:2])
    candidates = list(dict.fromkeys([current,*product((.075,.15,.25,.35,.425),repeat=2)]))
    best, best_area = current,-1.
    for i,xy in enumerate(candidates,1):
        if stop.is_set():
            return None
        mount[:2] = xy
        patches = solve_area(trial,0,stop)
        if stop.is_set():
            return None
        area = sum(size[2] for size,_,_ in patches)*500000.  # triangle area in mm²
        if area > best_area+1e-6:
            best,best_area = xy,area
        progress(i,len(candidates),best,best_area)
    return best,best_area
