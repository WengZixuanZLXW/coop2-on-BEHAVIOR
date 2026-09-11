"""Minimal stand-in for ROS 2's ament_index_python, for offline xacro runs.

`xacro` resolves `$(find pkg)` through
`ament_index_python.packages.get_package_share_directory`, and that package is
not on PyPI -- it ships only with a ROS 2 installation. The robot import needs
nothing else from ROS, so rather than requiring a full ROS Humble install for
one lookup, this resolves the same layout ROS does:

    <prefix>/share/ament_index/resource_index/packages/<name>   (marker)
    <prefix>/share/<name>                                       (the answer)

`prepare_official_robot_imports.register_package` already builds exactly that
layout for the unbuilt upstream checkouts, so the two fit together unchanged.
"""

import os


class PackageNotFoundError(KeyError):
    pass


def get_package_share_directory(package_name: str) -> str:
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        marker = os.path.join(prefix, "share", "ament_index", "resource_index", "packages", package_name)
        share = os.path.join(prefix, "share", package_name)
        if os.path.exists(marker) and os.path.isdir(share):
            return share
    raise PackageNotFoundError(f"package '{package_name}' not found on AMENT_PREFIX_PATH")


def get_package_prefix(package_name: str) -> str:
    return os.path.dirname(os.path.dirname(get_package_share_directory(package_name)))
