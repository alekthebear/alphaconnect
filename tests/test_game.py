from itertools import chain
from unittest import TestCase
from unittest.mock import patch

from alphaconnect.agents.mcts import MCTSAgent
from alphaconnect.game import State, get_row, play_move, count
from alphaconnect.play import evaluate


class TestGame(TestCase):
    def test_play_vertical_win(self):
        board = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 2, 2, 2, 0, 0, 0],
        ]
        state = State(board=list(chain.from_iterable(board)), mark=1, step=6)
        next_state = play_move(state, 0)
        self.assertEqual(next_state.winner, 1)

    def test_play_horizontal_win(self):
        board = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 2, 2, 2, 0, 0, 0],
        ]
        state = State(board=list(chain.from_iterable(board)), mark=1, step=6)
        state = play_move(state, 6)
        self.assertEqual(state.winner, None)
        state = play_move(state, 4)
        self.assertEqual(state.winner, 2)

    def test_play_diag_win(self):
        board = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 0, 0, 0],
            [0, 1, 2, 1, 0, 0, 0],
            [1, 2, 2, 1, 2, 0, 0],
        ]
        state = State(board=list(chain.from_iterable(board)), mark=1, step=11)
        state = play_move(state, 3)
        self.assertEqual(state.winner, 1)

    def test_count(self):
        board = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 0, 0, 0],
            [0, 1, 2, 1, 0, 0, 0],
            [1, 2, 2, 1, 2, 0, 0],
        ]
        # test placing token at column 1
        state = State(board=list(chain.from_iterable(board)), mark=1, step=11)
        col = 1
        row = get_row(state.board, col)
        self.assertEqual(row, 3)
        self.assertEqual(count(state.board, state.mark, row, col, 1, 0), 1)
        self.assertEqual(count(state.board, state.mark, row, col, 0, 1), 1)
        self.assertEqual(count(state.board, state.mark, row, col, 0, -1), 0)
        self.assertEqual(count(state.board, state.mark, row, col, -1, -1), 0)
        self.assertEqual(count(state.board, state.mark, row, col, 1, 1), 0)
        self.assertEqual(count(state.board, state.mark, row, col, -1, 1), 0)
        self.assertEqual(count(state.board, state.mark, row, col, 1, -1), 0)

        # test placing token at column 3
        col = 3
        row = get_row(state.board, col)
        self.assertEqual(row, 2)
        self.assertEqual(count(state.board, state.mark, row, col, 1, 0), 0)
        self.assertEqual(count(state.board, state.mark, row, col, 0, 1), 0)
        self.assertEqual(count(state.board, state.mark, row, col, 0, -1), 0)
        self.assertEqual(count(state.board, state.mark, row, col, -1, -1), 0)
        self.assertEqual(count(state.board, state.mark, row, col, 1, 1), 0)
        self.assertEqual(count(state.board, state.mark, row, col, -1, 1), 0)
        self.assertEqual(count(state.board, state.mark, row, col, 1, -1), 3)

    @patch("alphaconnect.play.play_game")
    def test_evaluate_rotate_first_move(self, mock_play_game):
        agent1 = MCTSAgent(iterations=10)
        agent2 = MCTSAgent(iterations=10)
        mock_play_game.return_value = 1  # first player always wins
        a1_wins, a2_wins = evaluate(agent1, agent2, 10)

        self.assertEqual(mock_play_game.call_count, 10)
        for i, call in enumerate(mock_play_game.call_args_list):
            if i % 2 == 0:
                self.assertEqual(call.args, (agent1, agent2))
            else:
                self.assertEqual(call.args, (agent2, agent1))
        self.assertEqual(a1_wins, a2_wins)
