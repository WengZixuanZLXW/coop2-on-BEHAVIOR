"""
Run LLM agents with the shared task graph topology (`tag`, from DIG-TAG).

Every team plans for itself around one shared task graph (comm_topology.
llm_tag.TagTeamBrain over the vendored coop2.dig_tag): each planning or
interrupt call is shown the graph, answers with task-graph actions that land
first, then notifies the teams it names -- an interrupt, within a per-step
budget -- then its plans. Same flags as run_individual plus --notify-budget;
the run directory gains tag.json (the graph), tag_rounds.json (every team
output: what it saw, did, sent, dropped) and tag.pdf / tag.png (the graph
drawn as the bipartite graph it is).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coop2.experiment.run_individual import main  # noqa: E402

if __name__ == "__main__":
    main(topology="tag")
