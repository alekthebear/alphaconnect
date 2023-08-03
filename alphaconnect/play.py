import argparse
import os
import sys

from torch.multiprocessing import Manager, Pool
from tqdm.autonotebook import tqdm
import torch as th

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from alphaconnect.agents.alpha import AlphaAgent
from alphaconnect.agents.base_agent import Agent
from alphaconnect.agents.human import HumanAgent
from alphaconnect.agents.mcts import MCTSAgent
from alphaconnect.agents.minimax import MinimaxAgent
from alphaconnect.game import init_state, play_move
from alphaconnect.model import ConnNet
from alphaconnect.utils import get_device


def play_game(a1: Agent, a2: Agent, print_steps: bool = False) -> int:
    """
    Plays a game between two agents and returns the winner.

    Args:
        a1 (Agent): The first agent.
        a2 (Agent): The second agent.
        print_steps (bool, optional): Whether to print the game steps. Defaults to False.

    Returns:
        int: The winner of the game.
    """

    state = init_state()
    agent_order = [a1, a2]
    while state.winner is None:
        if print_steps:
            mark_agent = agent_order[state.mark - 1]
            agent_name = mark_agent.__class__.__name__
            print(f"Step: {state.step} | Player {state.mark} ({agent_name})")
            state.print_board()

        if state.mark == 1:
            move = a1.act(state, print_scores=print_steps)
        else:
            move = a2.act(state, print_scores=print_steps)
        state = play_move(state, move)
    if print_steps:
        print("--------------------")
        state.print_board()
        if state.winner == 0:
            print("Draw!")
        else:
            print(f"Player {state.winner} ({agent_name}) wins!")
    return state.winner


def evaluate(
    agent1: Agent, agent2: Agent, num_games: int, processes: int = 1
) -> list[int]:
    """
    Evaluate the performance of two agents by simulating a specified number of games.

    Args:
        agent1 (Agent): The first agent.
        agent2 (Agent): The second agent.
        num_games (int): The number of games to simulate.
        processes (int, optional): The number of concurrent processes to use.
          Defaults to 1.
    Returns:
        list[int]: number of wins for agent1 and agent2 respectively.
    """
    agents = [agent1, agent2]
    wins = [0, 0]
    play_orders = [[0, 1] if i % 2 == 0 else [1, 0] for i in range(num_games)]

    if processes == 1:
        for play_order in tqdm(play_orders, desc="Games"):
            winner = play_game(agents[play_order[0]], agents[play_order[1]])
            wins[play_order[winner - 1]] += 1
    else:
        pbar = tqdm(total=num_games, desc="Games")
        results = []
        with Manager() as manager:
            agent1.prep_multiprocess(manager)
            agent2.prep_multiprocess(manager)
            with Pool(processes) as pool:
                for play_order in play_orders:
                    results.append(
                        pool.apply_async(
                            play_game,
                            args=(agents[play_order[0]], agents[play_order[1]]),
                            callback=lambda _: pbar.update(),
                        )
                    )
                pool.close()
                pool.join()
        for play_order, result in zip(play_orders, results):
            winner = result.get()
            wins[play_order[winner - 1]] += 1
    return wins


def get_agent(
    agent_str: str,
    device: str,
    model_path: str | None = None,
) -> Agent:
    if agent_str == "mcts":
        return MCTSAgent()
    if agent_str == "alphazero":
        if device == th.device("cuda"):
            th.multiprocessing.set_sharing_strategy("file_system")
            th.multiprocessing.set_start_method("spawn", force=True)
        if not model_path:
            raise ValueError("Model path is required for alphazero agent")
        net = ConnNet.load(model_path).to(device)
        return AlphaAgent(net)
    if agent_str == "alphabeta":
        return MinimaxAgent(alpha_beta=True, max_depth=4)
    if agent_str == "minimax":
        return MinimaxAgent(alpha_beta=False, max_depth=4)
    if agent_str == "human":
        return HumanAgent()
    raise ValueError(f"Unknown agent: {agent_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a game of connect four")
    parser.add_argument(
        "agent1",
        help="Type of agent for player 1, choices are: mcts, alphazero, alphabeta",
    )
    parser.add_argument(
        "-m1",
        "--model1",
        help="Path to model for player 1, required if agent1 is alphazero",
    )
    parser.add_argument(
        "agent2",
        help="Type of agent for player 2, choices are: mcts, alphazero, alphabeta",
    )
    parser.add_argument(
        "-m2",
        "--model2",
        help="Path to model for player 2, required if agent2 is alphazero",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=1,
        help=(
            "Number of games to play. If set to 1, the full game trajectory is shown. "
            "Otherwise, only the final win tally is shown"
        ),
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of concurrent processes to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_device(),
        help="Device to use: cuda, cpu. Defaults to cuda if available, otherwise cpu.",
    )
    args = parser.parse_args()
    agent1 = get_agent(args.agent1, args.device, args.model1)
    agent2 = get_agent(args.agent2, args.device, args.model2)
    if args.num_games > 1:
        wins = evaluate(agent1, agent2, args.num_games, processes=args.processes)
        draws = args.num_games - wins[0] - wins[1]
        print(
            f"Player 1 ({agent1.__class__.__name__}) wins: {wins[0]} | "
            f"Player 2 ({agent2.__class__.__name__}) wins: {wins[1]} | "
            f"Draws: {draws}"
        )
    else:
        winner = play_game(agent1, agent2, print_steps=True)
