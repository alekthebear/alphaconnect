from itertools import chain
from unittest import TestCase

import numpy as np
from alphaconnect.agents.minimax import MAX_SCORE, MIN_SCORE, MinimaxAgent
from alphaconnect.game import State


def _get_almost_win_state():
    board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 2, 2, 2, 0, 0, 0],
    ]
    return State(board=list(chain.from_iterable(board)), mark=1, step=6)


class TestMinimaxAgent(TestCase):
    def test_almost_win(self):
        agent = MinimaxAgent()
        state = _get_almost_win_state()
        action = agent.act(state)

        self.assertEqual(action, 0)

        _, scores = agent.negamax(state, 0, MIN_SCORE, MAX_SCORE)
        self.assertEqual(scores[0], MAX_SCORE)
        self.assertTrue(all(s == MIN_SCORE for s in scores[1:]))
