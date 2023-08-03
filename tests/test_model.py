from itertools import product
from unittest import TestCase

import torch as th
import numpy as np

from alphaconnect.game import State, play_move
from alphaconnect.model import ConnNet, states_to_tensor


def _get_test_states(batch_size=4):
    states = []
    state = State(board=[0] * 7 * 6, mark=1, step=0, winner=None)
    states.append(state)
    for _ in range(batch_size - 1):
        state = play_move(state, 1)
        states.append(state)
    return states


class TestModel(TestCase):
    def test_model_forward(self):
        model = ConnNet(3, 1)
        states = _get_test_states()
        x = states_to_tensor(states)
        p, v = model(x)
        self.assertEqual(p.shape, (len(states), 7))
        self.assertTrue(th.all(th.isclose(th.sum(p, dim=-1), th.tensor(1.0))))
        self.assertEqual(v.shape, (len(states),))

    def test_model_predict(self):
        model = ConnNet(3, 1)
        state = _get_test_states(batch_size=1)[0]
        p, v = model.predict(state)
        self.assertEqual(p.shape, (7,))
        self.assertTrue(np.all(np.isclose(np.sum(p, axis=-1), 1.0)))
        self.assertTrue(-1 <= v <= 1)

    def test_state_to_tensor(self):
        states = _get_test_states()
        x = states_to_tensor(states)
        rows, cols = x.shape[-2:]

        for i in range(len(states)):
            for r, c in product(range(rows), range(cols)):
                if states[i].board_2d[r][c] == 0:
                    self.assertEqual(x[i, 0, r, c].item(), 0)
                elif states[i].board_2d[r][c] == states[i].mark:
                    self.assertEqual(x[i, 0, r, c].item(), 1.0)
                else:
                    self.assertEqual(x[i, 0, r, c].item(), -1.0)
