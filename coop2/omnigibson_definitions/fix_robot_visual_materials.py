"""Give the imported V4 robots the colours their URDFs specify.

Isaac 5.1's URDF importer writes a robot's `<material>` colours only onto
visuals that are geometric primitives (a `<cylinder>`, a `<box>`). For every
`<mesh>` visual it instead creates one white `DefaultMaterial*` per mesh
*file* and binds the Mesh prim to that, dropping the URDF material entirely --
the coloured `material_<name>` Looks it does create are bound to nothing the
mesh inherits from. Measured on the imports this project made: Jackal 7 of 7
mesh prims white, Ridgeback+UR5 23 of 23 (the UR5's DAE parts carry their own
colours and are fine), Crazyflie 0 (a DAE).

This walks the resolved URDF for link -> visual mesh -> colour and rebinds each
white Mesh prim under `/visuals/<link>/` to `/<robot>/Looks/material_<name>`,
creating that Material in the importer's own OmniPBR shape when the URDF colour
has none yet. Rebinding per mesh, not recolouring the shared DefaultMaterial:
Ridgeback's `lights.stl` is white on the front link and black on the rear, and
both share one DefaultMaterial.

    python coop2/omnigibson_definitions/fix_robot_visual_materials.py [names...]

Edits the .usda in place after copying the original next to it as `<name>.usda.orig`
(once; never overwritten). Idempotent: a second run finds nothing to do.
"""
from __future__ import annotations

import os
import shutil
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ASSETS = os.path.join(ROOT, "datasets", "omnigibson-robot-assets", "objects", "robot")
URDFS = os.path.join(HERE, "robot_import", "resolved_urdf")
ROBOTS = ("v4_jackal", "v4_ridgeback_ur5", "v4_crazyflie_cf2x")


def urdf_colours(urdf_path):
    """{link_name: {mesh_stem: (material_name, (r, g, b))}} from the URDF."""
    root = ET.parse(urdf_path).getroot()
    global_mats = {}
    for m in root.findall("material"):
        c = m.find("color")
        if c is not None:
            global_mats[m.get("name")] = tuple(float(x) for x in c.get("rgba").split()[:3])
    out = {}
    for link in root.findall("link"):
        for v in link.findall("visual"):
            mesh = v.find("geometry/mesh")
            m = v.find("material")
            if mesh is None or m is None:
                continue
            stem = os.path.splitext(os.path.basename(mesh.get("filename")))[0]
            c = m.find("color")
            rgb = (tuple(float(x) for x in c.get("rgba").split()[:3]) if c is not None
                   else global_mats.get(m.get("name")))
            if rgb is None:
                continue
            out.setdefault(link.get("name"), {})[stem] = (m.get("name"), rgb)
    return out


def ensure_material(stage, looks_path, name, rgb, Sdf, UsdShade, Gf):
    """`/…/Looks/material_<name>`, in exactly the shape the importer writes."""
    path = looks_path.AppendChild(f"material_{name}")
    mat = UsdShade.Material.Get(stage, path)
    if mat and mat.GetPrim().IsValid():
        return mat, False
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for term in ("surface", "displacement", "volume"):
        mat.CreateOutput(f"mdl:{term}", Sdf.ValueTypeNames.Token).ConnectToSource(out)
    return mat, True


def fix(robot, lazy):
    Usd, UsdShade, UsdGeom, Sdf, Gf = lazy.pxr.Usd, lazy.pxr.UsdShade, lazy.pxr.UsdGeom, lazy.pxr.Sdf, lazy.pxr.Gf
    usd = os.path.join(ASSETS, robot, "usd", f"{robot}.usda")
    urdf = os.path.join(URDFS, f"{robot}_with_meta_links.urdf")
    if not (os.path.exists(usd) and os.path.exists(urdf)):
        print(f"[{robot}] missing {usd if not os.path.exists(usd) else urdf}; skipped")
        return
    colours = urdf_colours(urdf)
    stage = Usd.Stage.Open(usd)
    default_prim = stage.GetDefaultPrim()
    looks = default_prim.GetPath().AppendChild("Looks")
    visuals = stage.GetPrimAtPath("/visuals")
    if not visuals.IsValid():
        print(f"[{robot}] no /visuals scope; nothing to do")
        return

    rebound, created, unmatched, already = [], [], [], 0
    for link_prim in visuals.GetChildren():
        link = link_prim.GetName()
        for mesh_prim in Usd.PrimRange(link_prim):
            if not mesh_prim.IsA(UsdGeom.Mesh):
                continue
            binding = UsdShade.MaterialBindingAPI(mesh_prim).GetDirectBinding().GetMaterial()
            bound_name = binding.GetPrim().GetName() if binding and binding.GetPrim().IsValid() else ""
            if not bound_name.startswith("DefaultMaterial"):
                already += 1
                continue
            hit = colours.get(link, {}).get(mesh_prim.GetName())
            if hit is None:
                unmatched.append(f"{link}/{mesh_prim.GetName()}")
                continue
            name, rgb = hit
            mat, new = ensure_material(stage, looks, name, rgb, Sdf, UsdShade, Gf)
            if new:
                created.append(name)
            UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(mat)
            rebound.append((f"{link}/{mesh_prim.GetName()}", name, rgb))

    if rebound:
        backup = usd + ".orig"
        if not os.path.exists(backup):
            shutil.copy2(usd, backup)
        stage.GetRootLayer().Save()
    print(f"[{robot}] rebound {len(rebound)} white mesh(es); created {len(created)} material(s) "
          f"{created}; left alone {already} already-coloured; unmatched {len(unmatched)} {unmatched[:4]}")
    for path, name, rgb in rebound:
        print(f"    {path:44} -> material_{name:12} {tuple(round(c, 3) for c in rgb)}")


def main(argv):
    import omnigibson as og  # noqa: PLC0415
    from omnigibson import lazy  # noqa: PLC0415
    og.launch()
    for robot in (argv or ROBOTS):
        fix(robot, lazy)
    og.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
