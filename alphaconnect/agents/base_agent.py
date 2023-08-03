from abc import ABC, abstractmethod
from multiprocessing.managers import SyncManager

import numpy as np

from alphaconnect.game import State


class Agent(ABC):
    @abstractmethod
    def act(self, state: State, print_scores: bool = False) -> int:
        """
        Return the next move to play by the agent.
        Args:
            state (State): Current state.
            print_scores (bool, optional): Whether to print scores representing value of
                each action. Defaults to False.
        Returns:
            int: Move to make by agent
        """
        pass

    def prep_multiprocess(self, manager: SyncManager) -> None:
        """
        Prepares the agent for multiprocessing when playing multiple games in parallel.
        Main use case is for agent classes to take advantage of shared resources (e.g.
        sharing the same model on the GPU, sharing calculated state values)

        Args:
            manager (SyncManager): The sync manager to be used for multiprocessing.
        Returns:
            None
        """
        pass
