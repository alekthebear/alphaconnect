from unittest import TestCase

from alphaconnect.agents.mcts import MCTSAgent
from alphaconnect.train import Params, self_play, self_play_training


class TestTrain(TestCase):
    def test_self_play(self):
        agent = MCTSAgent(iterations=10)
        trajectories = self_play(agent, use_symmetry=False)

        # check that player 1 always goes first
        self.assertEqual(trajectories[0][0].mark, 1)

        # check that # of moves played is the length of trajectory
        state, search_policy, reward = trajectories[-1]
        moves = sum([1 for i in state.board if i != 0])
        self.assertEqual(moves + 1, len(trajectories))
        self.assertAlmostEqual(sum(search_policy), 1)

        win, loss = 0, 0
        player_marks = [0, 0]
        for state, _, reward in trajectories:
            player_marks[state.mark - 1] += 1
            if reward == 1:
                win += 1
            elif reward == -1:
                loss += 1
        self.assertTrue(abs(player_marks[1] - player_marks[0]) <= 1)
        self.assertTrue(abs(win - loss) <= 1)

    def test_self_play_training(self):
        """Basic test to ensure that everything at least runs"""
        params = Params(
            self_play_iterations=2,
            self_play_games=10,
            sample_queue_size=100,
            arena_games=5,
            dedupe_samples=True,
            tree_iterations=10,
            layers=2,
            filters=32,
            batch_size=32,
            processes=1,
            device="cpu",
            run_dir=None,
        )
        self_play_training(params)
