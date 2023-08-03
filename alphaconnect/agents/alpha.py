from math import sqrt
from multiprocessing.managers import DictProxy, SyncManager

import numpy as np

from alphaconnect.agents.mcts import MCTSAgent, Node
from alphaconnect.model import ConnNet
from alphaconnect.game import State


class AlphaAgent(MCTSAgent):
    def __init__(
        self,
        net: ConnNet,
        iterations: int = 500,
        c: float = 3,
    ):
        """
        AlphaZero agent for Connect 4. Based on MCTS, the differences being:
          1. For the selection method, we follow the Alpha Go Zero paper and use a
             modified PUCT algorithm, which combines a "prior" distribution from the model
             with the UCT exploration value.
          2. For the default policy, instead of simulating games to the end, we use a
             trained network to predict the value of each state.

        Args:
            net (ConnNet): The network used for training.
            iterations (int, optional): The number of iterations. Defaults to 500.
            c (float, optional): The value of c. Defaults to 3.
        """
        super().__init__(iterations=iterations, c=c)
        self.net = net
        self.Ps = {}  # type: DictProxy[State, np.ndarray] | dict[State, np.ndarray]
        self.Vs = {}  # type: DictProxy[State, float] | dict[State, float]

    def _select_child(self, node: Node) -> Node:
        if node.state not in self.Vs:
            priors, value = self.net.predict(node.state)
            self.Ps[node.state] = priors
            self.Vs[node.state] = value

        # normalize priors and mask out invalid moves
        priors = self.Ps[node.state]
        valid_mask = np.zeros(len(priors))
        valid_mask[node.state.valid_moves] = 1.0
        priors = priors * valid_mask
        priors = priors / priors.sum()

        # select best child
        child_scores = []
        for move, child in node.children.items():
            u_value = self.c * priors[move] * sqrt(node.visits) / (1 + child.visits)
            value = -child.q_value + u_value
            child_scores.append((value, child))
        node = max(child_scores, key=lambda c: c[0])[1]
        return node

    def _default_policy(self, node: Node) -> float:
        if node.is_terminal():
            if node.state.winner == 0:
                return 0
            elif node.state.winner == node.state.mark:
                return 1
            else:
                return -1

        if node.state not in self.Vs:
            priors, value = self.net.predict(node.state)
            self.Ps[node.state] = priors
            self.Vs[node.state] = value
        return self.Vs[node.state]

    @property
    def device(self):
        return self.net.device

    def prep_multiprocess(self, manager: SyncManager):
        """
        AlphaAgent does the following to prepare the agent for multiprocessing:
        1. Share the model so that all agents of the process use the same model, this is
           especially important if using the GPU so that processes don't load its own
           model into memory.
        2. Share the priors and values of the states. This reduces the number of queries
           to the model. Especially for earlier states where its likely for each process
           to repeatedly query the model for the same states.
        """
        self.net.share_memory()
        self.Ps = manager.dict()
        self.Vs = manager.dict()
