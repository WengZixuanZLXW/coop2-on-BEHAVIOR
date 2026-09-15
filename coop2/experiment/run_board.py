"""
Run LLM agents with the shared-message-board ablation (`board`, from DIG-TAG).

The `tag` team with the task graph replaced by an unstructured shared board
(comm_topology.llm_board.BoardTeamBrain): each planning or interrupt call is
shown the board's complete history, answers with the message to post, then the
teams to notify -- an interrupt, within a per-step budget -- then its plans.
Same round, same budget, same prompt sections as `tag`; the shared space is
the only difference, which is what makes it the ablation.

Same flags as run_individual plus --notify-budget; the run directory gains
board.json (every post) and board_rounds.json (every team output: what it saw,
wrote, sent, dropped).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coop2.experiment.run_individual import main  # noqa: E402

if __name__ == "__main__":
    main(topology="board")
