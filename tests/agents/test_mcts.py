from itertools import chain
from unittest import TestCase

from alphaconnect.agents.mcts import MCTSAgent, Node
from alphaconnect.game import State, init_state


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


class TestMCTS(TestCase):
    def test_tree_policy_expand(self):
        agent = MCTSAgent()
        state = init_state()
        root = Node(state)
        agent._tree_policy(root)

        self.assertEqual(len(root.children), 1)
        (child,) = list(root.children.values())
        self.assertEqual(child.state.step, 1)
        self.assertEqual(child.state.mark, 2)
        self.assertEqual(child.parent, root)

    def test_build_tree(self):
        itrs = 20
        agent = MCTSAgent(iterations=itrs)
        state = init_state()
        root = Node(state)
        agent.build_tree(root)

        self.assertEqual(root.visits, itrs)
        self.assertEqual(sum(c.visits for c in root.children.values()), itrs)
        self.assertEqual(sum(c.value for c in root.children.values()), root.value * -1)
        self.assertNotEqual(sum(len(c.children) for c in root.children.values()), 0)

    def test_build_tree_exploration_factor(self):
        state = _get_almost_win_state()
        small_c_root = Node(state)
        MCTSAgent(c=0.01).build_tree(small_c_root)
        big_c_root = Node(state)
        MCTSAgent(c=1).build_tree(big_c_root)

        self.assertEqual(small_c_root.visits, big_c_root.visits)
        self.assertGreater(
            small_c_root.children[0].visits, big_c_root.children[0].visits
        )

    def test_act(self):
        agent = MCTSAgent()
        state = _get_almost_win_state()
        action = agent.act(state)

        self.assertEqual(action, 0)
