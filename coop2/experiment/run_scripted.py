"""
Run a written-down plan instead of asking a model (`scripted`).

Same environment, teams, barrier and outputs as `individual` -- video included
-- with the planning call replaced by a lookup in a JSON script: the n-th entry
of a robot is handed to it on its team's n-th round, and a robot the script
does not name holds position. No credentials are needed; no model is called.

    # a skeleton for this layout, without launching the simulator
    python -m coop2.experiment.run_scripted \\
        --team-config coop2/team_layouts/s1/sets_3.json --print-template > script.json

    # run it
    python -m coop2.experiment.run_scripted --script script.json \\
        --team-config coop2/team_layouts/s1/sets_3.json \\
        --scene Merom_1_int --room living_room_0 \\
        --bddl-activity v4_s1_v4_lh --steps 2000

The script's format, and the closed set of actions it may use, are documented
in ``coop2/comm_topology/scripted_team.py``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coop2.experiment.run_individual import main  # noqa: E402

if __name__ == "__main__":
    main(topology="scripted")
