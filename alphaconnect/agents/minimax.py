import math
from random import choice

import numpy as np

from alphaconnect.agents.base_agent import Agent
from alphaconnect.game import Env, State, count, get_row, play_move


MAX_SCORE = math.inf
MIN_SCORE = -math.inf


class MinimaxAgent(Agent):
    def __init__(
        self,
        max_depth=4,
        alpha_beta=True,
        deterministic=False,
        debug=False,
    ):
        """
        Minimax agent for Connect 4.

        Args:
            max_depth (int): The maximum depth to search in the game tree. Defaults to 4.
            alpha_beta (bool): Use alpha-beta pruning. Defaults to True.
            deterministic (bool): If True, the agent will make make deterministic moves,
                always choosing the first option when there are moves with equivalent
                scores. Otherwise, the agent will randomly choose from equivalent moves.
                Defaults to False.
            debug (bool): Outputs search tree for debugging purposes. Defaults to False.
        """
        self.max_depth = max_depth
        self.alpha_beta = alpha_beta
        self.deterministic = deterministic
        # configuration
        self.debug = debug

    def _score_move(self, board, mark, col):
        """
        Trivial heuristic:
        1. Counts the number of connecting pieces if the move is made
        2. Encourage center positions
        """
        score = 0

        row = get_row(board, col)
        # vertical connections
        score += count(board, mark, row, col, 1, 0)
        # horizontal connections
        score += count(board, mark, row, col, 0, 1)
        score += count(board, mark, row, col, 0, -1)
        # top left diagonal connections
        score += count(board, mark, row, col, -1, -1)
        score += count(board, mark, row, col, 1, 1)
        # top left diagonal connections
        score += count(board, mark, row, col, -1, 1)
        score += count(board, mark, row, col, 1, -1)

        # encourge center positions
        score += col - Env.columns // 2
        return score

    def negamax(self, state, depth, alpha, beta):
        # Recursively check all columns.
        col_scores = [MIN_SCORE] * Env.columns
        best_score = MIN_SCORE
        for col in state.valid_moves:
            next_state = play_move(state, col)

            # check win condition
            if next_state.winner == state.mark:
                self.print_debug(
                    f"[Player {state.mark}] {depth * ' '}Depth: {depth} | "
                    f"Col: {col} | "
                    f"Score: {MAX_SCORE} (win) | "
                    f"alpha/beta: {alpha}/{beta}"
                )
                col_scores[col] = MAX_SCORE
                return MAX_SCORE, col_scores

            # if max_depth reached, use heuristic
            if depth == self.max_depth:
                score = self._score_move(state.board, state.mark, col)
            # otherwise, create and evaluate child nodes
            else:
                (score, _) = self.negamax(
                    state=next_state,
                    depth=depth + 1,
                    alpha=beta * -1,
                    beta=alpha * -1,
                )
                score = score * -1

            # update score
            if score > best_score or (
                not self.deterministic and score == best_score and choice([True, False])
            ):
                best_score = score
                col_scores[col] = score

            # check alpha beta
            alpha = max(alpha, best_score)
            self.print_debug(
                f"[Player {state.mark}] {depth * ' '}Depth: {depth} | "
                f"Col: {col} (Best: {np.argmax(col_scores)}) | "
                f"Score: {score} (Best: {best_score}) | "
                f"alpha/beta: {alpha}/{beta}"
            )
            if self.alpha_beta:
                if alpha > beta or (self.deterministic and alpha == beta):
                    self.print_debug(f"alpha > beta: {alpha}/{beta}")
                    break
        return best_score, col_scores

    def act(self, state: State, print_scores: bool = False) -> int:
        _, scores = self.negamax(state, 0, MIN_SCORE, MAX_SCORE)
        scores = np.array(scores)
        valid_mask = np.full(Env.columns, False)
        valid_mask[state.valid_moves] = True
        best_cols = np.where((scores == scores.max()) & valid_mask)[0]
        column = best_cols[0] if self.deterministic else np.random.choice(best_cols)
        if print_scores:
            print(f"Scores: {scores}")
            print(f"Move: {column}")
        return column

    def print_debug(self, msg):
        if self.debug:
            print(msg)
