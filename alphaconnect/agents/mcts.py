from __future__ import annotations

from math import sqrt
from random import choice

import numpy as np

from alphaconnect.agents.base_agent import Agent
from alphaconnect.game import Env, State, play_move


class Node:
    state: State
    children: dict[int, Node]
    parent: Node | None
    visits: int
    value: float

    def __init__(self, state: State, parent: Node | None = None):
        self.state = state
        self.children = {}
        self.parent = parent
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.state.valid_moves)

    def is_terminal(self) -> bool:
        return self.state.winner is not None

    def expand(self) -> Node:
        col = choice([m for m in self.state.valid_moves if m not in self.children])
        next_state = play_move(self.state, col)
        child_node = Node(next_state, self)
        self.children[col] = child_node
        return child_node

    @property
    def q_value(self) -> float:
        return self.value / self.visits


class MCTSAgent(Agent):
    def __init__(self, iterations: int = 1000, c: float = 0.5):
        """
        Monte Carlo Tree Search agent.

        Args:
            iterations (int): The number of iterations for the MCTS algorithm.
               Defaults to 1000.
            c (float): The exploration constant for UCT algorithm. Defaults to 0.5.
        """
        # MCTS configurations
        self.iterations = iterations
        self.c = c

    def _tree_policy(self, node: Node) -> Node:
        # use tree policy to select and expand node
        while True:
            if node.is_terminal():
                return node
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = self._select_child(node)

    def _select_child(self, node: Node) -> Node:
        """UCT algorithm is used for node selection"""
        child_scores = []
        for move, child in node.children.items():
            u_value = self.c * sqrt(node.visits / child.visits)
            value = -child.q_value + u_value
            child_scores.append((value, child))
        node = max(child_scores, key=lambda c: c[0])[1]
        return node

    def _default_policy(self, node: Node) -> float:
        state = node.state
        player = state.mark
        while state.winner is None:
            col = choice(state.valid_moves)
            state = play_move(state, col)
        if state.winner == 0:
            return 0
        else:
            return 1 if state.winner == player else -1

    @staticmethod
    def _backpropagate(node: Node | None, score: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += score
            node = node.parent
            score *= -1  # switch score for opponent

    def build_tree(self, root: Node) -> None:
        for _ in range(self.iterations):
            leaf = self._tree_policy(root)
            reward = self._default_policy(leaf)
            self._backpropagate(leaf, reward)

    def act(self, state: State, print_scores: bool = False) -> int:
        search_policy = self.get_search_policy(state)
        column = np.argmax(search_policy).item()
        if print_scores:
            print(f"Policy: {search_policy}")
            print(f"Move: {column}")
        return column

    def get_search_policy(self, state: State, t: float = 1) -> np.ndarray:
        root = Node(state)
        self.build_tree(root)

        # calculate search policy
        weights = []
        for col in range(Env.columns):
            if col not in root.children:
                weights.append(0)
            else:
                weights.append(root.children[col].visits ** (1 / t))
        policy = np.array(weights)
        return policy / policy.sum()


def print_tree(
    node: Node,
    depth: int = 0,
    move: int | None = None,
    max_depth: int | None = None,
) -> None:
    """Prints a tree in a readable format, used for debugging."""
    if (max_depth is not None and depth > max_depth) or node.visits == 0:
        return
    print(f"{'  ' * depth}-> Move {move} [Q: {node.q_value} | " f"N: {node.visits}]")
    for move, child in sorted(node.children.items()):
        print_tree(child, depth + 1, move, max_depth)
