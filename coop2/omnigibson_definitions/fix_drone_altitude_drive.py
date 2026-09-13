"""Give the Crazyflie a driven z joint, so it can hold an altitude.

The importer's holonomic option builds six virtual base joints but puts a
`PhysicsDriveAPI` on exactly three -- x, y, rz (`import_custom_robot.py`) --
because that is what `HolonomicBaseJointController` drives; it asserts 3 DOFs.
For a wheeled base the free z joint is harmless: the floor holds it up. For a
drone whose body has weight (it does, once `base_footprint_link_name` names the
body rather than a virtual link above the z joint) a free z joint simply falls.
(An earlier `z=+0.0500` blamed on this was the pose getter reading the wrong
link -- the body was at 1.2 m all along -- but the drive is needed regardless.)

This applies `PhysicsDriveAPI:linear` to `base_footprint_z_joint` with authored
gains, which makes `JointPrim.driven` true (HasAPI and the robot's is_driven)
and lets `set_pos(z, drive=True)` set a PhysX position target. Upstream refuses a
driven joint that no controller owns (`update_controller_mode` asserts), so the
model YAML hands z to the drone's otherwise joint-less `arm_0` JointController;
its idle command is the current joint position, so the altitude holds.
`place_robots` pins the target to the spawn altitude before the first action.
Gains: gravity is disabled on every body link not fixed to the base footprint,
so the drive only has to hold the virtual links; kp 100 / kd 10 is ample and
matches m.BASE_JOINT_CONTROLLER_POSITION_KP. Idempotent.

    python coop2/omnigibson_definitions/fix_drone_altitude_drive.py
"""
from __future__ import annotations

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
USD = os.path.join(ROOT, "datasets", "omnigibson-robot-assets", "objects", "robot",
                   "v4_crazyflie_cf2x", "usd", "v4_crazyflie_cf2x.usda")
JOINT = "base_footprint_z_joint"
KP, KD, MAX_FORCE = 100.0, 10.0, 1.0e4


def main():
    import omnigibson as og  # noqa: PLC0415
    from omnigibson import lazy  # noqa: PLC0415
    og.launch()
    Usd, UsdPhysics, Sdf = lazy.pxr.Usd, lazy.pxr.UsdPhysics, lazy.pxr.Sdf
    stage = Usd.Stage.Open(USD)
    hits = [p for p in stage.Traverse() if p.GetName() == JOINT and p.IsA(UsdPhysics.PrismaticJoint)]
    assert len(hits) == 1, f"expected one {JOINT}, found {[p.GetPath() for p in hits]}"
    prim = hits[0]
    if prim.HasAPI(UsdPhysics.DriveAPI, "linear"):
        drive = UsdPhysics.DriveAPI(prim, "linear")
        print(f"[crazyflie] {JOINT} already driven: kp={drive.GetStiffnessAttr().Get()} "
              f"kd={drive.GetDampingAttr().Get()}; nothing to do")
    else:
        backup = USD + ".before_z_drive"
        if not os.path.exists(backup):
            shutil.copy2(USD, backup)
        drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(KP)
        drive.CreateDampingAttr(KD)
        drive.CreateMaxForceAttr(MAX_FORCE)
        drive.CreateTargetPositionAttr(0.0)
        stage.GetRootLayer().Save()
        print(f"[crazyflie] applied PhysicsDriveAPI:linear to {prim.GetPath()} (kp={KP}, kd={KD}, "
              f"maxForce={MAX_FORCE}); original kept at {os.path.basename(backup)}")
    og.shutdown()


if __name__ == "__main__":
    sys.exit(main())
