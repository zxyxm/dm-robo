"""Assembly 5 fixed-section scan. SI internally; exported CSV uses mm and degrees."""
from pathlib import Path
import csv
import itertools
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from arm4_fk import load_model

ROOT = Path(__file__).resolve().parent
AXES = np.array([.08, .06, .05])
CENTER = np.array([.25, .25, 0.])
SLICES = np.arange(-50, 51, 10)
ANGLES = np.arange(0, 181, 2)


def turn(axis, angle):
    return Rotation.from_rotvec(axis * angle).as_matrix()


def signed(a, b, axis):
    return np.arctan2(axis @ np.cross(a, b), a @ b)


def hits_unit_sphere(tris):
    a,b,c = tris.transpose(1,0,2)
    dot = lambda x,y: np.einsum("ij,ij->i",x,y)
    n = np.cross(b-a,c-a)
    nn = np.maximum(dot(n,n),1e-24)
    h = dot(a,n)
    proj = n * (h/nn)[:,None]
    inside = np.ones(len(a),dtype=bool)
    for start,end in ((a,b),(b,c),(c,a)):
        edge = end-start
        t = np.clip(-dot(start,edge)/np.maximum(dot(edge,edge),1e-24),0,1)
        closest = start+t[:,None]*edge
        if np.any(dot(closest,closest)<1):
            return True
        inside &= dot(np.cross(edge,proj-start),n)>=0
    return np.any(inside & (h*h<nn))


class Workspace:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        mujoco.mj_forward(model, self.d)
        d = self.d
        self.axes = d.xaxis.copy()
        p = d.xanchor
        # Wrist center at intersection of axes 4, 5 and 6.
        proj = np.eye(3) - self.axes[3:, :, None] * self.axes[3:, None, :]
        self.w = np.linalg.solve(proj.sum(0), np.einsum("ijk,ik->j", proj, p[3:]))
        self.s = p[1].copy()
        self.elbow = p[2].copy()
        self.u, self.v = self.elbow - self.s, self.w - self.elbow
        self.u -= self.axes[1] * (self.u @ self.axes[1])
        self.v -= self.axes[1] * (self.v @ self.axes[1])
        site = d.site("acoustic")
        self.home = site.xmat.reshape(3, 3).copy()
        self.offset = self.home.T @ (site.xpos - self.w)
        self.basis = np.column_stack((self.axes[4], np.cross(self.axes[3], self.axes[4]), self.axes[3]))
        self.beta = signed(self.axes[3], self.axes[5], self.axes[4])
        self.hulls = {}
        self.facets = {}
        for i in range(model.ngeom):
            if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH or model.geom(i).name == "phantom":
                continue
            mesh = model.geom_dataid[i]
            start = model.mesh_vertadr[mesh]
            verts = model.mesh_vert[start:start+model.mesh_vertnum[mesh]]
            hull = ConvexHull(verts)
            self.hulls[i] = verts[hull.vertices]
            self.facets[i] = verts[hull.simplices]
        self.probe_cache = {}
        self.phantom = model.geom("phantom").id

    def solutions(self, target, orientation):
        a = self.axes
        wrist = target - orientation @ self.offset
        v = wrist - self.s
        radial = np.cross(a[1], a[0])
        heading = np.arctan2(v @ np.cross(a[0], radial), v @ radial)
        for j1 in (heading, heading + np.pi):
            r1 = turn(a[0], j1)
            goal = r1.T @ v
            l1, l2 = np.linalg.norm(self.u), np.linalg.norm(self.v)
            cosine = (goal @ goal - l1*l1-l2*l2) / (2*l1*l2)
            if abs(cosine) > 1+1e-10:
                continue
            for bend in (np.arccos(np.clip(cosine,-1,1)), -np.arccos(np.clip(cosine,-1,1))):
                j3 = (bend - signed(self.u, self.v, a[1])) * (a[1] @ a[2])
                j2 = signed(self.u + turn(a[2], j3) @ self.v, goal, a[1])
                upstream = r1 @ turn(a[1], j2) @ turn(a[2], j3)
                residual = upstream.T @ orientation @ self.home.T @ turn(a[4], self.beta)
                j4, mid, j6 = Rotation.from_matrix(self.basis.T @ residual @ self.basis).as_euler("ZXZ")
                for x,b,z in ((j4,mid,j6),(j4+np.pi,-mid,j6+np.pi)):
                    yield (np.array([j1,j2,j3,x,b-self.beta,z])+np.pi)%(2*np.pi)-np.pi

    def feasible(self, q, target, orientation):
        d = self.d
        d.qpos[:] = q
        mujoco.mj_kinematics(self.m, d)
        site = d.site("acoustic")
        if np.linalg.norm(site.xpos-target) > .0001 or np.linalg.norm(site.xmat.reshape(3,3)-orientation) > .001:
            return False
        for i, vertices in self.hulls.items():
            points = vertices @ d.geom_xmat[i].reshape(3,3).T + d.geom_xpos[i]
            if np.any(points.min(0)<-.00001) or np.any(points.max(0)>[.50001,.50001,.30001]):
                return False
            tolerance = .0005 if i == self.m.geom("probe_shell").id else .00001
            axes = AXES - tolerance
            lo,hi = points.min(0),points.max(0)
            if np.any(lo > CENTER + axes) or np.any(hi < CENTER - axes):
                continue
            # Scale the ellipsoid to a unit sphere and test each convex-hull facet.
            key = tuple(np.round(np.r_[target,orientation.flatten()],8)) if i == self.m.geom("probe_shell").id else None
            if key in self.probe_cache:
                if self.probe_cache[key]:
                    return False
                continue
            tris = ((self.facets[i] @ d.geom_xmat[i].reshape(3,3).T + d.geom_xpos[i]) - CENTER) / axes
            collision = hits_unit_sphere(tris)
            if key is not None:
                self.probe_cache[key] = collision
            if collision:
                return False
        for i,j in itertools.combinations(self.hulls,2):
            if abs(int(self.m.geom_bodyid[i])-int(self.m.geom_bodyid[j])) <= 1:
                continue
            if mujoco.mj_geomDistance(self.m,d,i,j,.001,None) < -.0001:
                return False
        return True


def target_pose(y_mm, angle, flip):
    a,b,c = AXES
    f = np.sqrt(1-(y_mm/1000/b)**2)
    t = np.deg2rad(angle)
    p = CENTER + [a*f*np.cos(t), y_mm/1000, c*f*np.sin(t)]
    inward = -np.array([np.cos(t)/a, 0., np.sin(t)/c])
    inward /= np.linalg.norm(inward)
    lateral = np.array([0.,float(flip),0.])
    return p, np.column_stack((np.cross(lateral,inward),lateral,inward))


def solve():
    solver = Workspace(load_model(ROOT / "arm5.xml"))
    print("lengths mm", [np.linalg.norm(v)*1000 for v in (solver.u,solver.v,solver.offset)])
    rows = []
    for y in SLICES:
        paths = []
        costs = []
        def finish():
            if not paths:
                return
            index = int(np.argmin(costs))
            for row, states, parents in reversed(paths):
                row[7:] = np.degrees(states[index]).tolist()
                index = parents[index]
            paths.clear()

        for angle in ANGLES:
            choices = []
            reachable = False
            for flip in (1,-1):
                p,r = target_pose(y,angle,flip)
                for q in solver.solutions(p,r):
                    reachable = True
                    if solver.feasible(q,p,r):
                        choices.append(q)
            row = [y+250,angle,*(p*1000),int(reachable),int(bool(choices)),*(['']*6)]
            rows.append(row)
            if not choices:
                finish()
                continue
            states = np.array(choices)
            if not paths:
                costs = .001*np.linalg.norm(states,axis=1)
                parents = np.zeros(len(states),dtype=int)
            else:
                delta = (states[:,None,:]-paths[-1][1][None,:,:]+np.pi)%(2*np.pi)-np.pi
                penalty = np.linalg.norm(delta,axis=2) + 1000*(np.max(np.abs(delta),axis=2)>.35)
                totals = costs[None,:]+penalty
                parents = np.argmin(totals,axis=1)
                costs = totals[np.arange(len(states)),parents]
            paths.append((row,states,parents))
        finish()
        print(y+250, sum(row[6] for row in rows[-len(ANGLES):]),flush=True)
    with (ROOT/"arm5_scan.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.writer(file)
        writer.writerow(["slice_y_mm","angle_deg","x_mm","y_mm","z_mm","kinematic","screened",*[f"J{i}_deg" for i in range(1,7)]])
        writer.writerows(rows)


def build_passes():
    """Audit the intermediate half-degree poses, splitting on IK branch changes."""
    rows = np.genfromtxt(ROOT/"arm5_scan.csv",delimiter=",",skip_header=1)
    solver = Workspace(load_model(ROOT/"arm5.xml"))
    output = []
    for y in np.unique(rows[:,0]):
        candidates = rows[(rows[:,0]==y)&(rows[:,6]==1)]
        if not len(candidates):
            continue
        q = np.radians(candidates[:,7:])
        change = (np.diff(q,axis=0)+np.pi)%(2*np.pi)-np.pi
        breaks = (np.diff(candidates[:,1])>2) | (np.max(abs(change),axis=1)>.35)
        for group in np.split(candidates,np.flatnonzero(breaks)+1):
            prev = np.radians(group[0,7:])
            first,last = group[[0,-1],1]
            for angle in np.arange(first,last+.1,.5):
                choices = []
                for flip in (1,-1):
                    target,r = target_pose(y-250,angle,flip)
                    for candidate in solver.solutions(target,r):
                        delta = (candidate-prev+np.pi)%(2*np.pi)-np.pi
                        if max(abs(delta)) < .35 and solver.feasible(candidate,target,r):
                            choices.append((np.linalg.norm(delta),prev+delta))
                if not choices:
                    raise RuntimeError(f"Discontinuous pass at Y={y}, angle={angle}")
                prev = min(choices,key=lambda t:t[0])[1]
                output.append([y,first,last,angle,*np.degrees(prev)])
    np.savetxt(ROOT/"arm5_passes.csv",output,delimiter=",",fmt="%.8f",comments="",
               header="y_mm,start_deg,end_deg,angle_deg,J1,J2,J3,J4,J5,J6")


def load_passes():
    rows = np.genfromtxt(ROOT/"arm5_passes.csv",delimiter=",",skip_header=1)
    return {f"Y={y:.0f} mm   {lo:.0f}–{hi:.0f}°":np.radians(np.array(list(group))[:,4:])
            for (y,lo,hi),group in itertools.groupby(rows,key=lambda r:tuple(r[:3]))}


if __name__ == "__main__":
    solve()
    build_passes()

def parallel_sections(model, initial_q, *, plane=None, offsets=(0, 2, 4, 6, 8, -2, -4, -6, -8)):
    """Yield (offset_mm, poses) for nine planes parallel to the captured acoustic plane."""
    work = Workspace(model)
    data = work.d
    data.qpos[:] = initial_q
    mujoco.mj_kinematics(model, data)
    site = data.site("acoustic")
    normal = site.xmat.reshape(3, 3)[:, 1].copy()
    distance = (site.xpos - CENTER) @ normal
    if plane is not None:
        axis, position_mm = plane
        normal = np.eye(3)[axis]
        distance = position_mm / 1000 - CENTER[axis]
    # Transform to the unit sphere, where every plane cuts a circle.
    scaled = AXES * normal
    length = np.linalg.norm(scaled)
    pole = scaled / length
    up = np.array([0., 0., 1.])
    if abs(pole @ up) > .99:
        up = np.array([1., 0., 0.])
    vertical = up - pole * (up @ pole)
    vertical /= np.linalg.norm(vertical)
    across = np.cross(pole, vertical)
    previous = np.array(initial_q)
    for offset in offsets:
        level = (distance + offset / 1000) / length
        runs = []
        run = []
        if abs(level) < 1:
            for angle in np.deg2rad(np.arange(-90, 270, 2)):
                circle = np.cos(angle)*across + np.sin(angle)*vertical
                unit = level*pole + np.sqrt(1-level**2)*circle
                target = CENTER + AXES*unit
                inward = -unit / AXES
                inward -= normal * (normal @ inward)
                inward /= np.linalg.norm(inward)
                frame = np.column_stack((np.cross(normal,inward), normal, inward))
                found = None
                if target[2] >= 0:
                    ref = run[-1] if run else previous
                    choices = []
                    for q in work.solutions(target,frame):
                        delta = (q-ref+np.pi)%(2*np.pi)-np.pi
                        choices.append((np.linalg.norm(delta),ref+delta))
                    for _,q in sorted(choices,key=lambda item:item[0]):
                        if work.feasible(q,target,frame):
                            found = q
                            break
                if run and (found is None or np.max(np.abs(found-run[-1])) > .35):
                    runs.append(run)
                    run = []
                if found is not None:
                    run.append(found)
        if run:
            runs.append(run)
        best = max(runs,key=len,default=[])
        if len(best) < 2:
            best = []
        elif np.linalg.norm(previous-best[-1]) < np.linalg.norm(previous-best[0]):
            best.reverse()
        if best:
            previous = best[-1]
        yield offset,best


def surface_start(axis, xyz_mm):
    """Radially project in normalized slice coordinates, preserving the slice offset."""
    xyz = np.asarray(xyz_mm, dtype=float) / 1000
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)) or xyz[2] < 0:
        raise ValueError("起点需要三个有限坐标，Z 必须大于或等于零")
    unit = (xyz - CENTER) / AXES
    if abs(unit[axis]) >= 1:
        raise ValueError("所选断面超出半椭球，或已退化为一点")
    horizontal = 1-axis
    angle = np.arctan2(unit[2], unit[horizontal]) if np.hypot(unit[2],unit[horizontal]) > 1e-12 else np.pi/2
    scale = np.sqrt(1-unit[axis]**2)
    unit[[horizontal,2]] = scale*np.array([np.cos(angle),np.sin(angle)])
    return (CENTER + AXES*unit)*1000, np.degrees(angle)


def surface_runs(work, axis, position, start_angle, left, right, initial, *, connected=False):
    """Two sweeps from the requested start, splitting at gaps and branch changes."""
    fixed = (position/1000-CENTER[axis])/AXES[axis]
    if abs(fixed) >= 1:
        return
    lateral = np.eye(3)[axis]
    ends = dict.fromkeys((max(0.,start_angle-right), min(180.,start_angle+left)))
    for end in ends:
        count = int(np.ceil(abs(end-start_angle)/2))
        if not count and len(ends) > 1:
            continue
        run = []
        for angle in np.linspace(start_angle,end,count+1):
            t = np.radians(angle)
            unit = lateral*fixed
            unit[[1-axis,2]] = np.sqrt(1-fixed**2)*np.array([np.cos(t),np.sin(t)])
            target = CENTER + AXES*unit
            beam = -unit/AXES
            beam[axis] = 0
            beam /= np.linalg.norm(beam)
            frame = np.column_stack((np.cross(lateral,beam), lateral, beam))
            ref = run[-1] if run else initial
            choices = []
            for rotation in (frame, frame @ np.diag([-1,-1,1])):
                for q in work.solutions(target,rotation):
                    delta = (q-ref+np.pi)%(2*np.pi)-np.pi
                    choices.append((np.linalg.norm(delta),ref+delta,rotation))
            found = next((q for _,q,r in sorted(choices,key=lambda item:item[0])
                          if work.feasible(q,target,r)),None)
            if connected and (found is None or (run and np.max(np.abs(found-ref)) > .35)):
                break
            if run and (found is None or np.max(np.abs(found-ref)) > .35):
                yield run
                run = []
            if found is not None:
                run.append(found)
        if run:
            yield run
        if not (left or right):
            break
