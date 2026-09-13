"""Shared-spawn team layouts for a COOHAVIOR scene: everyone starts in the most
open room, packed around the centroid of its largest free region.

The S2/S3 generalisation of make_s1_shared_spawn_layouts.py (2026-09-13). The
same measurement -- traversable floor from the no-object map minus the oriented
footprint of every object the scene edits keep, eroded by the largest robot
(a 0.7 m Jackal, square kernel) -- and the same packing: ground vehicles
nearest-to-centroid first in station order, pairwise clearance = sum of
half-footprints + 0.15 m, each Crazyflie hovering at 1.2 m over its own
Jackal, so sets_<k> is the first 3k robots of sets_5. Teams are team_1..team_k
in order of appearance and robots <type>_<ordinal>. V4's own staged poses are
not used (its Beechwood frame is not ours); the station id is kept as _v4_station.

    python feasibility_verify/make_shared_spawn_layouts.py --scene Beechwood_0_int --tag s2 \
        --edits coop2/team_layouts/v4_s2_v4_ll_scene.json --stations G1 G2 G3 G4 G5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from feasibility_verify.segroom import build  # noqa: E402

HALF = {"ridgeback": 0.25, "jackal": 0.35, "drone": 0.12}
MODEL = {"ridgeback": ("v4_ridgeback_ur5", 0.5, True), "jackal": ("v4_jackal", 0.7, False), "drone": ("v4_crazyflie_cf2x", 2.0, False)}
KINDS = ("ridgeback", "jackal", "drone")
RES = 0.01
MARGIN = 0.05
GAP = 0.15
HOVER_Z = 1.2
SKIP_CATS = ("floors", "walls", "ceilings", "window", "door")


def yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def kept_footprints(scene, edits):
    scn = json.load(open(f"{ROOT}/datasets/behavior-1k-assets/scenes/{scene}/json/{scene}_best.json"))
    info = scn["objects_info"]["init_info"]; reg = scn["state"]["registry"]["object_registry"]
    gone = set(edits.get("deactivate", [])); moves = edits.get("move", {}) or {}
    boxes = []
    for n, i in info.items():
        a = i["args"]; cat = a.get("category")
        if n in gone or cat in SKIP_CATS or n not in reg:
            continue
        meta = f"{ROOT}/datasets/behavior-1k-assets/objects/{cat}/{a['model']}/misc/metadata.json"
        if not os.path.exists(meta):
            continue
        m = json.load(open(meta)); scale = np.array(a.get("scale", [1, 1, 1]))
        bb = np.array(m["bbox_size"]) * scale; off = np.array(m.get("base_link_offset", [0, 0, 0])) * scale
        pos = np.array(moves.get(n, reg[n]["root_link"]["pos"])); th = yaw(reg[n]["root_link"]["ori"])
        c, s = math.cos(th), math.sin(th); R = np.array([[c, -s], [s, c]])
        ctr = pos[:2] + R @ off[:2]; hx, hy = bb[0] / 2, bb[1] / 2
        corners = np.array([R @ np.array([sx * hx, sy * hy]) for sx in (-1, 1) for sy in (-1, 1)]) + ctr
        zmin = pos[2] + off[2] - bb[2] / 2; zmax = pos[2] + off[2] + bb[2] / 2
        boxes.append((n, cat, corners[:, 0].min(), corners[:, 0].max(), corners[:, 1].min(), corners[:, 1].max(), zmin, zmax))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tag", required=True, help="output folder under coop2/team_layouts, e.g. s2")
    ap.add_argument("--edits", required=True, help="scene-edit json (deactivate list)")
    ap.add_argument("--stations", nargs="+", required=True, help="station ids in route order, e.g. G1 G2 G3 G4 G5")
    ap.add_argument("--coohavior", default="", help="task family label for the comment, e.g. S2")
    args = ap.parse_args()
    scene = args.scene
    edits = json.load(open(args.edits))
    boxes = kept_footprints(scene, edits)
    print(f"kept objects with footprints: {len(boxes)}")

    base = f"{ROOT}/datasets/behavior-1k-assets/scenes/{scene}/layout"
    trav = cv2.imread(f"{base}/floor_trav_no_obj_0.png", cv2.IMREAD_GRAYSCALE); H = trav.shape[0]

    def to_px(xy): return int(xy[1] / RES + H / 2), int(xy[0] / RES + H / 2)
    def to_xy(rc): return ((rc[1] - H / 2) * RES, (rc[0] - H / 2) * RES)

    occ = np.zeros_like(trav, dtype=np.uint8)
    for n, cat, x0, x1, y0, y1, z0, z1 in boxes:
        if z1 < 0.02:
            continue
        r0, c0 = to_px((x0, y0)); r1, c1 = to_px((x1, y1))
        occ[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1] = 1

    def free_mask(kind):
        h = HALF[kind] + MARGIN
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * int(h / RES) + 1,) * 2)
        floor = cv2.erode((trav > 0).astype(np.uint8), k); blocked = cv2.dilate(occ, k)
        return (floor == 1) & (blocked == 0)

    seg1 = build(scene, 0.01); inv1 = {v: k for k, v in seg1["names"].items()}
    big = free_mask("jackal")
    areas = {}
    for rname, rid in inv1.items():
        m = big & (seg1["ins"] == rid); areas[rname] = m.sum() * RES * RES
    print("admissible floor per room for a 0.7 m robot (m2):")
    for r, a in sorted(areas.items(), key=lambda kv: -kv[1]):
        print(f"  {r:20} {a:6.2f}")
    spawn_room = max(areas, key=areas.get); print("most open:", spawn_room)
    mask = big & (seg1["ins"] == inv1[spawn_room])
    num, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    comp = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])); mask = (lab == comp)
    cx, cy = cents[comp]; centre = to_xy((cy, cx))
    print(f"largest free component {stats[comp, cv2.CC_STAT_AREA] * RES * RES:.2f} m2, centroid {tuple(round(c, 2) for c in centre)}")
    rr, cc = np.where(mask); cells = np.stack([rr, cc], 1)
    dist_c = np.hypot(*(cells - np.array([cy, cx])).T) * RES; idx = np.argsort(dist_c)

    order = [(kind, st) for st in args.stations for kind in KINDS]
    placed = {}
    for kind, st in order:
        if kind == "drone":
            continue
        h = HALF[kind]
        for i in idx:
            cxy = to_xy(cells[i])
            if all(max(abs(cxy[0] - q[0]), abs(cxy[1] - q[1])) >= h + HALF[k2] + GAP for (k2, _), q in placed.items()):
                placed[(kind, st)] = (round(cxy[0], 3), round(cxy[1], 3)); break
        else:
            raise SystemExit(f"could not place {kind} {st}")
    for kind, st in order:
        if kind == "drone":
            placed[(kind, st)] = placed[("jackal", st)]
    print("packed spawn:")
    for key in order:
        print(f"  {key[0]:10} {key[1]:4} {placed[key]}  {math.hypot(placed[key][0] - centre[0], placed[key][1] - centre[1]):.2f} m from centroid")

    outdir = f"{ROOT}/coop2/team_layouts/{args.tag}"; os.makedirs(outdir, exist_ok=True)
    nsets = len(args.stations)
    for k in range(1, nsets + 1):
        robots = []
        for i, (kind, st) in enumerate(order[:3 * k]):
            model, scale, lock = MODEL[kind]
            pos = list(placed[(kind, st)]) + ([HOVER_Z] if kind == "drone" else [])
            ordinal = (i // 3) + 1
            e = {"name": f"{kind}_{ordinal}", "model": model, "position": pos, "team": f"team_{ordinal}",
                 "_v4_station": st, "scale": scale}
            if lock:
                e["base_locked_while_holding"] = True
            if kind == "drone":
                e["drone"] = True
            robots.append(e)
        doc = {"_comment": [
            f"COOHAVIOR {args.coohavior or args.tag.upper()} robots, {k} set{'s' if k > 1 else ''} ({3 * k} robots), for {scene} -- all spawned together.",
            "Teams are team_1.. in order of appearance and robots are <type>_<team ordinal>",
            "(user, 2026-09-13); the COOHAVIOR station a team stands for is _v4_station.",
            "",
            f"Everyone starts in {spawn_room}, packed around the centroid of its largest free",
            "region (decision: user, 2026-09-12). The room is the most open one measured as",
            "traversable floor (no-object map) minus the footprints of every object the scene",
            "edits keep, eroded by the largest robot footprint; admissible area per room, m2: "
            + ", ".join(f"{r} {a:.1f}" for r, a in sorted(areas.items(), key=lambda kv: -kv[1]) if a > 0) + ".",
            "Ground vehicles are placed nearest-to-centroid first in station order with pairwise",
            "clearance = sum of half-footprints + 0.15 m, so sets_<k> is the first 3k robots of",
            f"sets_{nsets}. Each Crazyflie spawns at z = 1.2 m directly above its own team's Jackal.",
            "",
            "V4's staged poses are not used: COOHAVIOR's coordinates for this scene are in a",
            "different frame from our dataset's. Scales as the S1 layouts (ridgeback 0.5,",
            "jackal 0.7, crazyflie 2.0). Written by feasibility_verify/make_shared_spawn_layouts.py.",
        ], "robots": robots}
        path = f"{outdir}/sets_{k}.json"
        json.dump(doc, open(path, "w"), indent=2); open(path, "a").write("\n")
        print(f"wrote {args.tag}/sets_{k}.json  ({len(robots)} robots)")


if __name__ == "__main__":
    main()
