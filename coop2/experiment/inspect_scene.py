"""Load a task's scene and stop, so the layout can be checked by eye.

Takes the same three files a run does -- the BDDL activity, its cached
instance, and the team layout -- builds the environment, and then does
*nothing*. No agents, no LLM, no plan loop. What it prints is the one question
worth asking before spending an episode on a task: is everything where it was
meant to be?

For each robot it reports where the layout asked for it and where it actually
ended up, because those differ for real reasons -- a pinned pose that lands
inside furniture is moved by ``place_robots``, and a robot whose base sits
above its wheels settles to a different z than the one it was spawned at.

Examples:
    # Print the layout and save a picture of it.
    python -m coop2.experiment.inspect_scene \\
        --scene Merom_1_int --room childs_room_0 \\
        --bddl-activity v4_s1_v4_ll \\
        --team-config coop2/team_layouts/v4_s1_v4_ll.json \\
        --shot /tmp/scene.png

    # Open a window and leave it open to orbit around (needs a DISPLAY).
    python -m coop2.experiment.inspect_scene --gui --hold ... 
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coop2.behavior_env.coop_env import CooperativeEnv
from coop2.behavior_env.team_config import homogeneous_layout, load_team_layout


def _xyz(entity):
    position = entity.get_position_orientation()[0]
    return tuple(float(v) for v in position[:3])


def _room_of(env, xy):
    try:
        import torch as th  # noqa: PLC0415

        return env.env.scene.seg_map.get_room_instance_by_point(
            th.tensor(list(xy[:2]), dtype=th.float32)
        )
    except Exception as error:  # noqa: BLE001 - a scene may have no seg map
        return f"(unknown: {type(error).__name__})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", type=str, default=None, help="Scene model")
    parser.add_argument("--room", type=str, default=None, help="Room to place unpinned robots in")
    parser.add_argument("--bddl-activity", type=str, default=None, metavar="NAME")
    parser.add_argument("--bddl-instance-id", type=int, default=0)
    parser.add_argument("--team-config", type=str, default=None, metavar="PATH",
                        help="Team layout JSON: robot models, teams and start poses")
    parser.add_argument("--agents", type=int, default=2,
                        help="Robot count, when there is no --team-config")
    parser.add_argument("--team-size", type=int, default=None, metavar="K")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle", type=int, default=60,
                        help="Physics steps to run before reporting, so anything "
                             "spawned in the air has landed")
    parser.add_argument("--gui", action="store_true",
                        help="Open a window. The only way to get one -- "
                             "OMNIGIBSON_HEADLESS alone cannot, because the env "
                             "sets gm.HEADLESS itself. Needs a DISPLAY.")
    parser.add_argument("--hold", action="store_true",
                        help="With --gui, keep stepping so the window stays "
                             "responsive until Ctrl-C")
    parser.add_argument("--view", type=str, default=None, metavar="AGENT",
                        help="Print the symbolic view this robot is given -- the "
                             "actual prompt text. The cheapest way to see whether "
                             "the task is even stateable: a destination the agent "
                             "cannot see is one it cannot name.")
    parser.add_argument("--focus", type=str, default=None, metavar="AGENT",
                        help="Frame --shot tightly on one robot instead of the "
                             "whole group. A 7 cm drone is invisible in a shot "
                             "wide enough to hold a Ridgeback.")
    parser.add_argument("--shot", type=str, default=None, metavar="PATH",
                        help="Save a PNG of the scene, framed on the robots and "
                             "the task's objects")
    args = parser.parse_args()

    if args.team_config:
        layout = load_team_layout(args.team_config)
    else:
        layout = homogeneous_layout(args.agents, room=args.room, team_size=args.team_size or 1)
    print(f"\n[layout] {args.team_config or f'--agents {args.agents}'}\n{layout.describe()}")

    import omnigibson as og  # noqa: PLC0415
    from omnigibson.macros import gm  # noqa: PLC0415

    if args.gui or args.shot:
        gm.RENDER_VIEWER_CAMERA = True

    env_kwargs = dict(
        area=(64, 64), view=(9, 9), size=(84, 84), reward=True,
        length=1000, n_players=layout.n_agents, seed=args.seed,
        coop_config_path="paper",
        headless=not args.gui,
        team_layout=layout,
        video_path=None,
        # Headless still needs a camera to take a picture with. Recording used
        # to be the only thing that turned one on.
        want_viewer_camera=bool(args.shot),
    )
    if args.scene:
        env_kwargs["scene_model"] = args.scene
    if args.room:
        env_kwargs["room"] = args.room
    if args.bddl_activity:
        env_kwargs["bddl_activity"] = args.bddl_activity
        env_kwargs["bddl_instance_id"] = args.bddl_instance_id

    env = CooperativeEnv(**env_kwargs)
    env.reset()
    for _ in range(args.settle):
        og.sim.step()

    wanted = {robot.name: robot for robot in layout.robots}
    print(f"\n=== ROBOTS after a {args.settle}-step settle ===")
    print(f"{'agent':<10} {'model':<20} {'asked for':<26} {'actual':<26} {'moved':>7}  "
          f"{'size WxDxH (m)':<18} room")
    for name, robot in zip(env.agent_names, env.env.robots):
        spec = wanted.get(name)
        actual = _xyz(robot)
        if spec is not None and spec.position is not None:
            asked = f"({spec.position[0]:+.3f}, {spec.position[1]:+.3f})"
            drift = ((actual[0] - spec.position[0]) ** 2 + (actual[1] - spec.position[1]) ** 2) ** 0.5
            drift_text = f"{drift:6.3f}"
        else:
            asked = f"room {spec.room}" if spec is not None else "-"
            drift_text = "     -"
        try:
            extent = robot.aabb_extent
            size = f"{float(extent[0]):.2f}x{float(extent[1]):.2f}x{float(extent[2]):.2f}"
        except Exception:  # noqa: BLE001
            size = "?"
        print(f"{name:<10} {getattr(robot, 'model', '?'):<20} {asked:<26} "
              f"({actual[0]:+.3f}, {actual[1]:+.3f}, {actual[2]:+.3f})  {drift_text}  "
              f"{size:<18} {_room_of(env, actual)}")

    scope = getattr(getattr(env.env, "task", None), "object_scope", None) or {}
    movable = {
        name: entity for name, entity in scope.items()
        if entity is not None and not name.startswith(("agent.n.01", "floor.n.01"))
    }
    if movable:
        print(f"\n=== TASK OBJECTS ===")
        print(f"{'bddl instance':<28} {'scene name':<22} {'position':<28} room")
        for name, entity in sorted(movable.items()):
            obj = getattr(entity, "wrapped_obj", entity)
            position = _xyz(obj)
            print(f"{name:<28} {getattr(obj, 'name', '?'):<22} "
                  f"({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f})    "
                  f"{_room_of(env, position)}")

    if args.view:
        info = getattr(env, "_last_info", None) or {}
        entry = info.get(args.view) or {}
        view = entry.get("symbolic_view")
        if view:
            print(f"\n=== WHAT {args.view} IS TOLD ===\n{view}")
        else:
            print(f"\nno symbolic view for {args.view!r}; have {sorted(info)}")

    task = getattr(env.env, "task", None)
    if task is not None and hasattr(task, "compiled_task"):
        try:
            met, breakdown = task.compiled_task.check_goal(task._evaluate_predicate)
            print(f"\n=== GOAL ===\n  satisfied at load: {met}   {breakdown}"
                  f"\n  (False is what you want: a task already met is not a task)")
        except Exception as error:  # noqa: BLE001
            print(f"\n=== GOAL ===\n  could not evaluate: {type(error).__name__}: {error}")

    if args.shot:
        _save_shot(env, args.shot, args.focus)

    if args.hold:
        print("\nholding the viewer open; Ctrl-C to quit")
        try:
            while True:
                og.sim.step()
        except KeyboardInterrupt:
            print("\nclosing")

    og.shutdown()
    return 0


def _save_shot(env, path, focus=None):
    """Photograph the robots and task objects from directly overhead.

    Straight down, deliberately. A three-quarter view is the natural choice and
    it does not work in these rooms: the eye lands inside a wall whenever the
    subject stands near one, and the frame comes back flat grey. The recorder
    learned the same thing the hard way -- see `chase_pose` in recording.py,
    where every oblique geometry was tried and only overhead ever produced a
    usable frame. A 2.4 m ceiling is what makes it so unforgiving.
    """
    import numpy as np  # noqa: PLC0415
    import torch as th  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    import omnigibson as og  # noqa: PLC0415
    import omnigibson.utils.transform_utils as T  # noqa: PLC0415

    if focus:
        chosen = [
            robot for name, robot in zip(env.agent_names, env.env.robots) if name == focus
        ]
        if not chosen:
            print(f"no such robot to focus on: {focus!r}; have {list(env.agent_names)}")
            return
        points = [chosen[0].get_position_orientation()[0]]
    else:
        points = [robot.get_position_orientation()[0] for robot in env.env.robots]
        scope = getattr(getattr(env.env, "task", None), "object_scope", None) or {}
        for name, entity in scope.items():
            if entity is None or name.startswith(("agent.n.01", "floor.n.01")):
                continue
            obj = getattr(entity, "wrapped_obj", entity)
            points.append(obj.get_position_orientation()[0])
    if not points:
        print("nothing to frame")
        return

    camera = og.sim.viewer_camera
    if camera is None:
        print("no viewer camera: pass --shot (which asks for one) or --gui")
        return

    stacked = th.stack(points)
    centre = stacked.mean(dim=0)
    # Just below a 2.4 m ceiling. Higher would frame more and see nothing.
    # Focused on one robot, come right down on it instead: a 7 cm drone is a
    # couple of pixels from ceiling height.
    height = float(centre[2]) + 0.45 if focus else 2.25
    eye = th.tensor([float(centre[0]), float(centre[1]), height])
    # Looking straight down: -Z forward, +Y of the image pointing along world +Y.
    forward = th.tensor([0.0, 0.0, -1.0])
    up = th.tensor([0.0, 1.0, 0.0])
    right = th.linalg.cross(forward, up)
    camera.set_position_orientation(
        position=eye, orientation=T.mat2quat(th.stack([right, up, -forward], dim=1))
    )
    for _ in range(5):
        og.sim.render()
    frame = camera.get_obs()[0]["rgb"][:, :, :3].cpu().numpy().astype(np.uint8)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    Image.fromarray(frame).save(path)
    spread = float(th.norm(stacked[:, :2] - centre[:2], dim=1).max())
    print(f"\nwrote {path}  (overhead at {height:.2f} m, {len(points)} points "
          f"spanning {2 * spread:.2f} m)")


if __name__ == "__main__":
    sys.exit(main())
