#!/usr/bin/env python3
"""Prepare resolved URDF inputs and YAMLs for BEHAVIOR's official importer.

This does not run Isaac Sim or overwrite an OmniGibson dataset.  Run this
first, inspect the generated inputs, then use ``import_official_v4_robots.sh``
to invoke the official ``import_custom_robot`` example.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SRC = HERE / "robot_sources"
OUT = HERE / "robot_import"
PREFIX = OUT / "ament_prefix"
#: How to run xacro. ROS 2's own binary when a ROS install is present;
#: otherwise the `xacro` PyPI package in the active env, whose only missing
#: dependency (`ament_index_python`, not on PyPI) is supplied by _ament_shim.
#: Override either with the XACRO_BIN / XACRO_PYTHON environment variables.
XACRO = Path(os.environ.get("XACRO_BIN", "/opt/ros/humble/bin/xacro"))
XACRO_PYTHON = os.environ.get("XACRO_PYTHON", "/usr/bin/python3")
AMENT_SHIM = HERE / "_ament_shim"


def register_package(name: str, source: Path) -> None:
    """Expose an unbuilt source package to ROS 2 xacro via a tiny ament index."""
    (PREFIX / "share" / "ament_index" / "resource_index" / "packages").mkdir(parents=True, exist_ok=True)
    marker = PREFIX / "share" / "ament_index" / "resource_index" / "packages" / name
    marker.touch()
    share = PREFIX / "share" / name
    share.parent.mkdir(parents=True, exist_ok=True)
    if share.exists() or share.is_symlink():
        share.unlink()
    share.symlink_to(source.resolve(), target_is_directory=True)


def xacro(source: Path, target: Path, extra_env: dict[str, str] | None = None) -> None:
    if not XACRO.is_file():
        raise RuntimeError(
            f"xacro is missing: {XACRO}. Either install ROS 2 Humble, or "
            f"`pip install xacro` in this env and set XACRO_BIN/XACRO_PYTHON "
            f"to it (the shim in {AMENT_SHIM.name} supplies ament_index_python)."
        )
    env = os.environ.copy()
    # ROS Humble's xacro is packaged for Ubuntu's Python 3.10.  A caller's
    # Conda Python (currently 3.14 in this workspace) otherwise intercepts
    # the console entry point and cannot find ROS package metadata.
    for key in ("PYTHONHOME", "PYTHONPATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
        env.pop(key, None)
    if XACRO_PYTHON == "/usr/bin/python3":
        # A ROS 2 install: use its own interpreter's package tree.
        env["PYTHONPATH"] = ":".join(
            [
                "/opt/ros/humble/local/lib/python3.10/dist-packages",
                "/opt/ros/humble/lib/python3.10/site-packages",
            ]
        )
    else:
        # The env's own xacro. Keep its site-packages (the caller's env is the
        # one that has it) and prepend the shim so `$(find pkg)` resolves.
        env.pop("PYTHONHOME", None)
        env["PYTHONPATH"] = str(AMENT_SHIM)
    env["AMENT_PREFIX_PATH"] = f"{PREFIX}:/opt/ros/humble:" + env.get("AMENT_PREFIX_PATH", "")
    env.update(extra_env or {})
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        subprocess.run([XACRO_PYTHON, str(XACRO), str(source)], check=True, env=env, stdout=stream)


def replace_package_uri(urdf: Path, package: str, package_root: Path) -> None:
    """Make ROS package mesh URIs directly readable by urdfpy / Isaac.

    The official OmniGibson importer consumes a resolved URDF, but its URDF
    reader does not consult the ROS ament index for ``package://`` mesh URIs.
    Keep the upstream xacro unchanged and rewrite only our generated copy.
    """
    uri = f"package://{package}/"
    replacement = str(package_root.resolve()) + "/"
    text = urdf.read_text(encoding="utf-8")
    if uri in text:
        urdf.write_text(text.replace(uri, replacement), encoding="utf-8")


def sanitize_mesh_filenames(urdf: Path) -> None:
    """Provide importer-safe aliases for mesh basenames containing hyphens.

    Isaac's URDF importer derives USD prim names from a mesh basename.  A
    source mesh such as ``jackal-base.stl`` consequently fails USD's identifier
    validation. We create symlink aliases beside generated inputs and replace
    only the generated URDF references, never modifying the source checkout.
    """
    aliases = OUT / "import_meshes" / urdf.stem
    aliases.mkdir(parents=True, exist_ok=True)
    text = urdf.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        source = Path(match.group(1))
        if "-" not in source.name:
            return match.group(0)
        safe_name = source.name.replace("-", "_")
        alias = aliases / safe_name
        if not alias.exists() and not alias.is_symlink():
            alias.symlink_to(source.resolve())
        return f'filename="{alias}"'

    urdf.write_text(re.sub(r'filename="([^"]+)"', replace, text), encoding="utf-8")


def strip_ros_control_transmissions(urdf: Path) -> None:
    """Drop legacy ROS-control transmission metadata from a generated URDF.

    OmniGibson imports the joints directly. The Ridgeback UR5 description
    includes duplicate transmission names from its base and arm overlays, and
    urdfpy rejects those before the actual joint graph is read.
    """
    text = urdf.read_text(encoding="utf-8")
    urdf.write_text(re.sub(r"\s*<transmission\b.*?</transmission>\s*", "\n", text, flags=re.DOTALL), encoding="utf-8")


def importer_config(name: str, urdf: Path, wheel_links=None, wheel_joints=None) -> dict:
    return {
        "urdf_path": str(urdf.resolve()),
        "name": name,
        "dataset_name": "omnigibson-robot-assets",
        "headless": True,
        "overwrite": False,
        "merge_fixed_joints": False,
        "base_motion": {
            "wheel_links": wheel_links or [],
            "wheel_joints": wheel_joints or [],
            "use_sphere_wheels": False,
            "use_holonomic_joints": False,
        },
        "collision": {
            "decompose_method": "convex",
            "hull_count": 8,
            "coacd_links": [],
            "convex_links": [],
            "no_decompose_links": [],
            "no_collision_links": [],
        },
        "eef_vis_links": [],
        "camera_links": [],
        "lidar_links": [],
        "curobo": {},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    resolved = OUT / "resolved_urdf"
    # Created here rather than relied on as a side effect of the first xacro()
    # call: with the FANUC source absent that call is skipped, and the next
    # writer found no directory.
    resolved.mkdir(parents=True, exist_ok=True)
    configs = OUT / "import_configs"
    configs.mkdir(parents=True, exist_ok=True)

    # Which upstream checkouts are actually present. The sources are large and
    # gitignored, and a run only needs the robots its task uses -- the FANUC
    # mesh source alone is ~614 MB and is only used by the checkpoint arms of
    # the H* tasks -- so each robot is prepared only if its source is there.
    have = {
        "v4_fanuc_crx10ial": (SRC / "fanuc_description").is_dir(),
        "v4_ridgeback_ur5": (SRC / "ridgeback_base").is_dir()
        and (SRC / "ridgeback_manipulation").is_dir()
        and (SRC / "ur_description_ros1").is_dir(),
        "v4_crazyflie_cf2x": (SRC / "crazyswarm2").is_dir(),
        "v4_jackal": (SRC / "jackal").is_dir(),
    }
    for name, present in have.items():
        print(f"V4_OG_IMPORT_SOURCE={name}:{'present' if present else 'ABSENT (skipped)'}")

    # ROS package overlays required by the official xacro files.
    if have["v4_fanuc_crx10ial"]:
        register_package("fanuc_crx_description", SRC / "fanuc_description" / "fanuc_crx_description")
    register_package("ridgeback_description", SRC / "ridgeback_base" / "ridgeback_description")
    register_package("ridgeback_ur_description", SRC / "ridgeback_manipulation" / "ridgeback_ur_description")
    # The Ridgeback integration is ROS1-era and requires this matching ROS1 UR5 xacro.
    register_package("ur_description", SRC / "ur_description_ros1" / "ur_description")

    if have["v4_fanuc_crx10ial"]:
        xacro(
            SRC / "fanuc_description" / "fanuc_crx_description" / "robot" / "crx10ia_l.urdf.xacro",
            resolved / "fanuc_crx10ial.urdf",
        )
        replace_package_uri(
            resolved / "fanuc_crx10ial.urdf",
            "fanuc_crx_description",
            SRC / "fanuc_description" / "fanuc_crx_description",
        )
        sanitize_mesh_filenames(resolved / "fanuc_crx10ial.urdf")
    # Same principle for Ridgeback: omit optional Gazebo/accessory xacros that
    # depend on legacy ROS1 lidar packages. The physical base and UR5 remain.
    ridgeback_src = SRC / "ridgeback_base" / "ridgeback_description" / "urdf" / "ridgeback.urdf.xacro"
    ridgeback_text = ridgeback_src.read_text(encoding="utf-8")
    ridgeback_text = ridgeback_text.replace(
        '<xacro:include filename="$(find ridgeback_description)/urdf/ridgeback.gazebo" />', ""
    ).replace('<xacro:include filename="$(find ridgeback_description)/urdf/accessories.urdf.xacro" />', "")
    ridgeback_tmp = resolved / "ridgeback_ur5_base.urdf.xacro"
    ridgeback_tmp.write_text(ridgeback_text, encoding="utf-8")
    xacro(
        ridgeback_tmp,
        resolved / "ridgeback_ur5.urdf",
        {
            "RIDGEBACK_URDF_EXTRAS": str(
                SRC / "ridgeback_manipulation" / "ridgeback_ur_description" / "urdf" / "ridgeback_ur5_description.urdf.xacro"
            ),
            "RIDGEBACK_UR_XYZ": "0 0 0",
            "RIDGEBACK_UR_RPY": "0 0 0",
        },
    )
    replace_package_uri(
        resolved / "ridgeback_ur5.urdf",
        "ridgeback_description",
        SRC / "ridgeback_base" / "ridgeback_description",
    )
    replace_package_uri(
        resolved / "ridgeback_ur5.urdf",
        "ur_description",
        SRC / "ur_description_ros1" / "ur_description",
    )
    strip_ros_control_transmissions(resolved / "ridgeback_ur5.urdf")
    sanitize_mesh_filenames(resolved / "ridgeback_ur5.urdf")

    # Crazyswarm's file is already a URDF. Replace its ROS package URI with an
    # absolute local mesh path so the OmniGibson importer can resolve it.
    crazy_src = SRC / "crazyswarm2" / "crazyflie_description" / "urdf" / "crazyflie_description.urdf"
    crazy_text = crazy_src.read_text(encoding="utf-8").replace(
        "package://crazyflie_description/urdf/",
        str((SRC / "crazyswarm2" / "crazyflie_description" / "urdf").resolve()) + "/",
    ).replace('name="$NAME"', 'name="crazyflie_cf2x"')
    # The upstream visualization-only description has one link and no joint,
    # whereas the official OG importer requires an articulation root. Add the
    # smallest physical wrapper in our generated copy only.
    crazy_text = crazy_text.replace(
        '<link name="crazyflie_cf2x">',
        '''<link name="world"/>
  <joint name="world_to_crazyflie" type="fixed">
    <parent link="world"/><child link="crazyflie_cf2x"/>
  </joint>
  <link name="crazyflie_cf2x">''',
    ).replace(
        '<origin rpy="0 0 0" xyz="0.0 0 0" />',
        '''<origin rpy="0 0 0" xyz="0.0 0 0" />
    <inertial><mass value="0.027"/><inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00002"/></inertial>
    <collision><geometry><sphere radius="0.06"/></geometry></collision>''',
    )
    (resolved / "crazyflie_cf2x.urdf").write_text(crazy_text, encoding="utf-8")

    # Jackal's stock xacro unconditionally imports optional ROS1 sensor
    # packages. For the V4 base robot we deliberately omit Gazebo/accessory
    # includes; the V4 cargo plate and suction are added after import.
    jackal_src = SRC / "jackal" / "jackal_description" / "urdf" / "jackal.urdf.xacro"
    jackal_text = jackal_src.read_text(encoding="utf-8")
    jackal_text = jackal_text.replace(
        '<xacro:include filename="$(find jackal_description)/urdf/jackal.gazebo" />', ""
    ).replace('<xacro:include filename="$(find jackal_description)/urdf/accessories.urdf.xacro" />', "")
    jackal_tmp = resolved / "jackal_base.urdf.xacro"
    jackal_tmp.write_text(jackal_text, encoding="utf-8")
    register_package("jackal_description", SRC / "jackal" / "jackal_description")
    xacro(
        jackal_tmp,
        resolved / "jackal.urdf",
        {
            "JACKAL_URDF_EXTRAS": str(SRC / "jackal" / "jackal_description" / "urdf" / "empty.urdf"),
            "CPR_URDF_EXTRAS": str(SRC / "jackal" / "jackal_description" / "urdf" / "empty.urdf"),
        },
    )
    replace_package_uri(
        resolved / "jackal.urdf",
        "jackal_description",
        SRC / "jackal" / "jackal_description",
    )
    sanitize_mesh_filenames(resolved / "jackal.urdf")

    all_specs = {
        "v4_jackal": importer_config(
            "v4_jackal", resolved / "jackal.urdf",
            ["front_left_wheel_link", "front_right_wheel_link", "rear_left_wheel_link", "rear_right_wheel_link"],
            ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"],
        ),
        "v4_ridgeback_ur5": importer_config(
            "v4_ridgeback_ur5", resolved / "ridgeback_ur5.urdf",
            ["front_left_wheel_link", "front_right_wheel_link", "rear_left_wheel_link", "rear_right_wheel_link"],
            ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"],
        ),
        "v4_crazyflie_cf2x": importer_config("v4_crazyflie_cf2x", resolved / "crazyflie_cf2x.urdf"),
        "v4_fanuc_crx10ial": importer_config("v4_fanuc_crx10ial", resolved / "fanuc_crx10ial.urdf"),
    }
    specs = {name: cfg for name, cfg in all_specs.items() if have.get(name)}
    for name, config in specs.items():
        with (configs / f"{name}.yaml").open("w", encoding="utf-8") as stream:
            # JSON is valid YAML and avoids requiring a host-Python PyYAML
            # installation merely to prepare official importer inputs.
            json.dump(config, stream, indent=2)
            stream.write("\n")
    print(f"V4_OG_IMPORT_INPUTS_READY={OUT}")
    for name in specs:
        print(f"V4_OG_IMPORT_CONFIG={configs / (name + '.yaml')}")


if __name__ == "__main__":
    main()
