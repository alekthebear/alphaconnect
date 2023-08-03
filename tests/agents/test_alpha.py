from itertools import chain
from queue import Queue
from unittest import TestCase
from unittest.mock import MagicMock

import torch as th
import numpy as np

from alphaconnect.agents.mcts import Node
from alphaconnect.agents.alpha import AlphaAgent
from alphaconnect.game import State, init_state
from alphaconnect.model import ConnNet


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


class TestAlphaAgent(TestCase):
    def test_tree_policy_expand(self):
        net = MagicMock()
        net.predict.return_value = np.array([1 / 7] * 7), 0
        agent = AlphaAgent(net)
        root = Node(init_state())
        agent._tree_policy(root)

        self.assertEqual(len(root.children), 1)
        (child,) = list(root.children.values())
        self.assertEqual(child.state.step, 1)
        self.assertEqual(child.state.mark, 2)
        self.assertEqual(child.parent, root)

    def test_build_tree(self):
        net = MagicMock()
        net.predict = lambda x: (th.tensor([1 / 7] * 7), th.tensor(0.0).uniform_(-1, 1))
        itrs = 20
        agent = AlphaAgent(net, iterations=itrs)
        root = Node(init_state())
        agent.build_tree(root)

        self.assertEqual(root.visits, itrs)
        self.assertEqual(sum(c.visits for c in root.children.values()), itrs)
        self.assertTrue(
            th.isclose(sum(c.value for c in root.children.values()), root.value * -1)
        )
        self.assertNotEqual(sum(len(c.children) for c in root.children.values()), 0)

    def test_biased_prior(self):
        """
        Test that a very biased prior (where only 1 move is recommended) dictates the
        child selection.
        """
        net = MagicMock()
        net.predict = lambda x: (th.tensor([1] + [0] * 6), th.tensor(1.0))
        itrs = 35
        agent = AlphaAgent(net, iterations=itrs)
        root = Node(init_state())
        agent.build_tree(root)

        self.assertEqual(root.visits, itrs)
        self.assertEqual(sum(c.visits for c in root.children.values()), itrs)
        # traversal through tree and ensure only move 0 nodes are visited more than once
        q = Queue()
        q.put(root)
        while not q.empty():
            node = q.get()
            for move, child in node.children.items():
                if move != 0:
                    self.assertEqual(child.visits, 1)
                q.put(child)

    def test_act(self):
        net = ConnNet(4)
        agent = AlphaAgent(net)
        state = _get_almost_win_state()
        action = agent.act(state)

        self.assertEqual(action, 0)

        move_prob = agent.get_search_policy(state)
        self.assertEqual(np.argmax(move_prob), 0)
