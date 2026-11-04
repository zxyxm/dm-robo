"""Assembly 4 flange FK. Angles are degrees relative to the saved CAD pose."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

MODEL = Path(__file__).with_name("arm4.xml")


def reference_fk(angles):
    """Product of rotations around the home-pose axes, in the base frame."""
    tree = ET.parse(MODEL)
    pose = np.eye(4)
    anchor = np.zeros(3)
    for body, angle in zip(tree.iter("body"), angles):
        anchor += np.fromstring(body.get("pos"), sep=" ")
        axis = np.fromstring(body.find("joint").get("axis"), sep=" ")
        rotation = Rotation.from_rotvec(angle * axis).as_matrix()
        step = np.eye(4)
        step[:3, :3] = rotation
        step[:3, 3] = anchor - rotation @ anchor
        pose = pose @ step
    home = np.eye(4)
    quat = np.fromstring(tree.find(".//site").get("quat"), sep=" ")
    home[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    home[:3, 3] = anchor
    return pose @ home


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("angles", type=float, nargs="*", help="J1 to J6 in degrees")
    args = parser.parse_args()
    if len(args.angles) not in (0, 6):
        parser.error("supply six joint angles, or omit all six for CAD home")
    angles = np.deg2rad(args.angles or [0] * 6)
    model = mujoco.MjModel.from_xml_string(MODEL.read_text(encoding="utf-8"))
    data = mujoco.MjData(model)
    data.qpos[:] = angles
    mujoco.mj_forward(model, data)
    flange = data.site("flange")
    pose = np.eye(4)
    pose[:3, :3] = flange.xmat.reshape(3, 3)
    pose[:3, 3] = flange.xpos
    np.testing.assert_allclose(pose, reference_fk(angles), atol=1e-10, rtol=0)
    orientation = Rotation.from_matrix(pose[:3, :3])
    np.set_printoptions(precision=8, suppress=True)
    print("MuJoCo", mujoco.__version__, "| XYZ (mm)", flange.xpos * 1000)
    print("RPY (deg), Rz(yaw) Ry(pitch) Rx(roll):", orientation.as_euler("xyz", degrees=True))
    print("Quaternion (w,x,y,z):", orientation.as_quat()[[3, 0, 1, 2]])
    print("Base_T_flange (translation in m):\n", pose)


if __name__ == "__main__":
    main()
