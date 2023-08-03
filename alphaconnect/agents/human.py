from alphaconnect.agents.base_agent import Agent
from alphaconnect.game import State


class HumanAgent(Agent):
    def act(self, state: State, print_scores: bool = False) -> int:
        print("---------------------")
        print("(0, 1, 2, 3, 4, 5, 6)")
        move = int(input("Enter your move: "))
        if move not in state.valid_moves:
            print(f"Move {move} is invalid")
            return self.act(state, print_scores)
        return move
