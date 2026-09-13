"""Where does cargo actually go when a drone loads it onto a Jackal? (GPU)

In the first routed LL episode (run ..._011125_636624, 2026-09-13) the video
showed the die lying beside jackal_2 on the floor from the moment drone_2's
load_onto began (tick ~840) until drone_3 unloaded it (~940), while the world
model said the die was riding on jackal_2 the whole time. This script does that
handoff with no LLM and prints the die's pose against the Jackal's body every
few ticks, so the question "on its back, in the air beside it, or on the
floor" is answered by numbers rather than by a top-down frame.

    OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/verify_carrier_cargo_gpu.py
"""

from __future__ import annotations

import math
import os
import sys
import traceback

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ACTIVITY = "v4_s1_v4_ll"
SCENE = "Merom_1_int"
# sets_1: the drone spawns at 1.2 m, as in the routed episodes. The LL layouts
# leave the drone's z out and it starts on the floor, which hides the question.
LAYOUT = "coop2/team_layouts/s1/sets_1.json"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    # The symbolic set, as behavior_action dispatches: the Starter enum's members
    # carry different values, and apply_ref goes by value -- the first version of
    # this script asked for NAVIGATE_TO and the controller ran TOGGLE_ON.
    from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
        SymbolicSemanticActionPrimitiveSet as P,
    )

    from coop2.behavior_env import primitive_engine as pe
    from coop2.behavior_env.coop_env import CooperativeBehaviorEnv
    from coop2.behavior_env.team_config import load_team_layout

    layout = load_team_layout(LAYOUT)
    env = CooperativeBehaviorEnv(
        n_agents=len(layout.robots), seed=0, scene_model=SCENE, bddl_activity=ACTIVITY,
        bddl_instance_id=0, room="living_room_0", length=5000, team_layout=layout,
    )
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(f"  {'PASS' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    try:
        env.reset()
        scene = env.env.scene
        robots = {r.name: r for r in env.env.robots}
        drone, jackal, arm = robots["drone_1"], robots["jackal_1"], robots["ridgeback_1"]
        die = env.env.task.object_scope["die.n.01_1"]
        engine = env.engine

        def pose_line(tag: str) -> None:
            d = die.get_position_orientation()[0]
            j = jackal.get_position_orientation()[0]
            lo, hi = jackal.aabb
            c = jackal.aabb_center
            inside_xy = bool(lo[0] <= d[0] <= hi[0] and lo[1] <= d[1] <= hi[1])
            print(f"  [{tag:>22}] tick {engine.env_step:5d}  die z={float(d[2]):.3f} "
                  f"dxy-from-jackal-origin={math.hypot(float(d[0]-j[0]), float(d[1]-j[1])):.3f} "
                  f"dxy-from-jackal-aabb-centre={math.hypot(float(d[0]-c[0]), float(d[1]-c[1])):.3f} "
                  f"jackal top={float(hi[2]):.3f} origin z={float(j[2]):.3f} inside-footprint={inside_xy}")

        def run(agent: str, primitive, target, label: str, watch_every: int = 25) -> None:
            outcome = engine.assign(agent, primitive, target)
            if outcome is not None:
                print(f"  [{label}] refused: {outcome.reason_code} {outcome.failure_reason}")
                return
            ticks = 0
            while engine.has_active(agent):
                outcomes = engine.tick()
                ticks += 1
                if ticks % watch_every == 0:
                    pose_line(f"{label} +{ticks}")
                if agent in outcomes:
                    o = outcomes[agent]
                    print(f"  [{label}] {o.status} after {o.ticks} ticks"
                          + (f": {o.reason_code} {o.failure_reason}" if o.status != "success" else ""))

        print("\n0. geometry: Jackal origin vs body")
        j = jackal.get_position_orientation()[0]; lo, hi = jackal.aabb; c = jackal.aabb_center
        print(f"  origin=({float(j[0]):.3f}, {float(j[1]):.3f}, {float(j[2]):.3f})  "
              f"aabb centre=({float(c[0]):.3f}, {float(c[1]):.3f}, {float(c[2]):.3f})  "
              f"extent=({float(hi[0]-lo[0]):.3f}, {float(hi[1]-lo[1]):.3f}, {float(hi[2]-lo[2]):.3f})")
        check(math.hypot(float(j[0]-c[0]), float(j[1]-c[1])) < 0.10,
              "the Jackal's origin is within 10 cm (xy) of its body centre")
        pose_line("start")

        print("\n1. drone: navigate_to(die), grasp(die)")
        run("drone_1", P.NAVIGATE_TO, die.name, "drone nav")
        body = drone.get_position_orientation()[0]
        print(f"  drone body z={float(body[2]):.3f}  eef('0') pos={[round(float(v),3) for v in drone.get_eef_position('0')]}  "
              f"grasping_mode={getattr(drone, 'grasping_mode', None)}  arm_names={list(drone.arm_names)}")
        outcome = engine.assign("drone_1", P.GRASP, die.name)
        assert outcome is None, outcome
        print(f"  before grasp: drone ang vel={float(drone.get_angular_velocity().norm()):.3f} rad/s, lin vel={float(drone.get_linear_velocity().norm()):.3f}")
        for t in range(1, 12):
            engine.tick()
            d = die.get_position_orientation()[0]
            ag = getattr(drone, "_ag_obj_in_hand", None)
            print(f"  [grasp tick +{t:2d}] die z={float(d[2]):.3f} xy=({float(d[0]):.3f}, {float(d[1]):.3f})  "
                  f"_ag_obj_in_hand={ {k: getattr(v, 'name', v) for k, v in (ag or {}).items()} }  "
                  f"die vel={float(die.get_linear_velocity().norm()):.3f}  drone ang vel={float(drone.get_angular_velocity().norm()):.3f}  drone lin vel={float(drone.get_linear_velocity().norm()):.3f}")
        while engine.has_active("drone_1"):
            engine.tick()
        pose_line("after grasp")
        # Does the spin persist once the primitive is over? 100 idle ticks,
        # watching the six virtual base joints (x y z rx ry rz).
        idx = [int(i) for i in drone.base_idx]
        def base_q():
            q = drone.get_joint_positions()
            return [round(float(q[i]), 3) for i in idx]
        print(f"  base joints (x y z rx ry rz) before: {base_q()}; driven flags: "
              f"{[bool(getattr(list(drone.joints.values())[i], 'driven', None)) for i in idx] if hasattr(drone, 'joints') else '?'}")
        import omnigibson.utils.transform_utils as T
        body_link = drone.links.get("crazyflie_cf2x") if hasattr(drone, "links") else None
        print(f"  root link: {drone.root_link.name}; body link found: {body_link is not None}")
        def yaw_deg():
            return float(T.quat2euler(drone.get_position_orientation()[1])[2]) * 180.0 / math.pi
        yaw0 = yaw_deg(); pos0 = drone.get_position_orientation()[0].clone()
        for t in range(100):
            engine.tick()
            if t % 25 == 24:
                p_now = drone.get_position_orientation()[0]
                print(f"  [idle +{t+1:3d}] root ang vel={float(drone.get_angular_velocity().norm()):.3f}  "
                      f"body ang vel={float(body_link.get_angular_velocity().norm()) if body_link is not None else float('nan'):.3f}  "
                      f"body yaw={yaw_deg():.1f} deg (start {yaw0:.1f})  body moved={float((p_now - pos0).norm()):.4f} m  "
                      f"die vel={float(die.get_linear_velocity().norm()):.3f}  die z={float(die.get_position_orientation()[0][2]):.3f}  base q={base_q()}")
        # Hypothesis: the hanging weight swings the free rx/ry joints for ever.
        # Take gravity off the die and see whether the spin dies out.
        try:
            die.disable_gravity()
            for t in range(50):
                engine.tick()
                if t % 25 == 24:
                    print(f"  [no-gravity +{t+1:3d}] root ang vel={float(drone.get_angular_velocity().norm()):.3f}  body yaw={yaw_deg():.1f} deg")
            die.enable_gravity()
        except Exception as error:  # noqa: BLE001
            print(f"  gravity toggle unavailable: {error}")
        print(f"  eef('0') pos after grasp={[round(float(v),3) for v in drone.get_eef_position('0')]}  body z={float(drone.get_position_orientation()[0][2]):.3f}")
        held = engine.controllers["drone_1"]._get_obj_in_hand()
        check(held is die, "the world says the drone holds the die")
        check(float(die.get_position_orientation()[0][2]) > 0.5, "the die is up in the air with the drone, not on the floor")

        print("\n1b. drone flies to the jackal holding the die: does the die come along, and at what height?")
        die_before = die.get_position_orientation()[0].clone()
        run("drone_1", P.NAVIGATE_TO, jackal.name, "drone->jackal", watch_every=50)
        d = die.get_position_orientation()[0]; eef = drone.get_eef_position("0"); body = drone.get_position_orientation()[0]
        print(f"  die z={float(d[2]):.3f} moved {math.hypot(float(d[0]-die_before[0]), float(d[1]-die_before[1])):.2f} m; "
              f"eef z={float(eef[2]):.3f} body z={float(body[2]):.3f}; die-to-eef {float((d-eef).norm()):.3f} m")
        # The user saw a drone start turning after a hop and never stop. Watch
        # the body's yaw for 150 idle ticks right after the navigate, holding.
        yaw_a = yaw_deg(); rz_i = idx[5]
        targets = getattr(drone, "get_joint_position_targets", None)
        for t in range(150):
            engine.tick()
            if t % 30 == 29:
                q = drone.get_joint_positions()
                tgt = f" rz target={float(targets()[rz_i]):.3f}" if callable(targets) else ""
                print(f"  [post-nav idle +{t+1:3d}] body yaw={yaw_deg():.1f} deg (was {yaw_a:.1f})  rz joint={float(q[rz_i]):.3f}{tgt}  "
                      f"body ang vel={float(body_link.get_angular_velocity().norm()) if body_link is not None else float('nan'):.3f}  die z={float(die.get_position_orientation()[0][2]):.3f}")
        check(abs(yaw_deg() - yaw_a) < 2.0, f"no drift in yaw while hovering with the die ({yaw_a:.1f} -> {yaw_deg():.1f} deg)")
        check(math.hypot(float(d[0]-die_before[0]), float(d[1]-die_before[1])) > 0.5, "the die travelled with the drone")
        check(float((d - eef).norm()) < 0.15, "and is still at the drone's suction mount")
        check(float(d[2]) > 0.5, "up in the air, not on the floor")

        print("\n2. jackal: navigate_to(drone) so the two are within reach")
        run("jackal_1", P.NAVIGATE_TO, drone.name, "jackal nav")

        print("\n3. drone: load_onto(jackal) -- watch the die")
        pose_line("before load")
        run("drone_1", pe.LOAD_ONTO, jackal.name, "load_onto", watch_every=10)
        pose_line("after load")
        d = die.get_position_orientation()[0]; lo, hi = jackal.aabb
        check(bool(lo[0] <= d[0] <= hi[0] and lo[1] <= d[1] <= hi[1]), "die xy is inside the Jackal's footprint")
        check(abs(float(d[2]) - float(hi[2])) < 0.08, f"die sits at the Jackal's top ({float(d[2]):.3f} vs top {float(hi[2]):.3f})")
        check(float(d[2]) > 0.15, "die is not on the floor")

        print("\n4. jackal drives away; the die must come along")
        before = die.get_position_orientation()[0].clone()
        run("jackal_1", P.NAVIGATE_TO, "bed_zrumze_0" if scene.object_registry("name", "bed_zrumze_0") else env.env.task.object_scope["bed.n.01_1"].name, "jackal drive", watch_every=50)
        after = die.get_position_orientation()[0]; jpos = jackal.get_position_orientation()[0]
        moved = math.hypot(float(after[0]-before[0]), float(after[1]-before[1]))
        print(f"  die moved {moved:.2f} m; die-to-jackal xy {math.hypot(float(after[0]-jpos[0]), float(after[1]-jpos[1])):.3f} m")
        check(moved > 0.5, "the die travelled with the Jackal")
        check(math.hypot(float(after[0]-jpos[0]), float(after[1]-jpos[1])) < 0.3, "and is still over its body")

        print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S):"))
        for f in failures:
            print(f"  - {f}")
        return 1 if failures else 0
    except BaseException:
        traceback.print_exc(); sys.stderr.flush(); raise
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
