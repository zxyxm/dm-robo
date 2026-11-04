"""Sample pose reachability independently of the scan start or force setpoint."""
import numpy as np
import mujoco

from arm5_scan import AXES, CENTER, Workspace, section_pose

BLUE = np.array([.04,.40,.95,1],dtype=np.float32)


def sample_section(work, axis, position, step, stop):
    fixed = (position/1000-CENTER[axis])/AXES[axis]
    samples = []
    for angle in range(0,181,step):
        if stop.is_set():
            return []
        point,frame = section_pose(axis,fixed,angle)
        reachable = any(work.feasible(q,point,rotation)
                        for rotation in (frame,frame@np.diag([-1,-1,1]))
                        for q in work.solutions(point,rotation))
        normal = (point-CENTER)/AXES**2
        samples.append((point+.00045*normal/np.linalg.norm(normal),reachable))
    return samples


def triangle(a,b,c):
    u,v = b-a,c-a
    n = np.cross(u,v)
    if n@(a-CENTER)<0:
        u,v,n = v,u,-n
    lengths = np.array([np.linalg.norm(u),np.linalg.norm(v),np.linalg.norm(n)])
    matrix = np.column_stack((u,v,n))/lengths
    return lengths,a,matrix.ravel()


def solve_area(model,axis,stop):
    work = Workspace(model)
    middle = CENTER[axis]*1000
    radius = round(AXES[axis]*1000)
    previous = []
    patches = []
    for offset in range(-radius+4,radius,4):
        row = sample_section(work,axis,middle+offset,6,stop)
        if stop.is_set():
            return []
        for a,b,c,d in zip(previous,previous[1:],row,row[1:]):
            if all(sample[1] for sample in (a,b,c,d)):
                patches.extend((triangle(a[0],b[0],c[0]),triangle(b[0],d[0],c[0])))
        previous = row
    return patches


def draw_area(scene,patches):
    for size,pos,mat in patches:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_TRIANGLE,
                            size,pos,mat,BLUE)
        scene.ngeom += 1
