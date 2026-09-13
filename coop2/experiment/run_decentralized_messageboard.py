"""
Run LLM agents with the Decentralized Message Board topology.

Individual planning around one shared board (comm_topology.llm_team.
MessageboardTeamBrain, comm_topology.message_board): no team waits for or
messages another, but every team is shown the board when it plans and posts
one entry with its plans. Same flags as run_individual; the board is saved
as message_board.json in the run directory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coop2.experiment.run_individual import main  # noqa: E402

if __name__ == "__main__":
    main(topology="decentralized_messageboard")
