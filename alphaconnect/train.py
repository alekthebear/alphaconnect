from collections import deque
from copy import deepcopy
from dataclasses import dataclass, asdict
from datargs import parse
from glob import glob
from itertools import chain
from typing import Any, Iterable
import datetime
import json
import logging
import os
import pickle
import random
import sys

from torch.multiprocessing import Pool, Manager
from tqdm.autonotebook import tqdm
import numpy as np
import torch as th

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from alphaconnect import utils
from alphaconnect.agents.alpha import AlphaAgent
from alphaconnect.agents.mcts import MCTSAgent
from alphaconnect.game import State, play_move, init_state
from alphaconnect.model import ConnNet, states_to_tensor
from alphaconnect.play import evaluate


###################
# Hyperparameters #
###################
@dataclass(frozen=True, eq=True)
class Params:
    # Self-Play Parameters
    self_play_iterations: int = 10  # number of self-play training iterations
    self_play_games: int = 1000  # number of games to play in each iteration
    sample_queue_size: int = 500_000  # number of training samples to keep in FIFO queue
    arena_games: int = 100  # number of games to play when evaluating performance
    dedupe_samples: bool = True  # dedupe samples by average the value/policy scores
    update_agent_threshold: float = 0.5  # fraction of wins agent needs to be kept
    # RL Parameters
    tree_iterations: int = 500  # MCTS iterations per move
    cpuct: float = 3  # c_puct exploration constant
    temp_change_step: int = 15  # temperature is 1.0 until move n, 0.3 afterwards
    # Neural Network Parameters
    layers: int = 8
    filters: int = 128
    batch_size: int = 1024
    lr: float = 0.001
    weight_decay: float = 0.0001
    # Other parameters
    processes: int = 8
    device: str = str(utils.get_device())
    seed: int = 42
    # run_dir: if None, no logs/samples/models/metrics are saved
    run_dir: str = f"runs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def dump(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str):
        with open(path, "r") as f:
            return Params(**json.load(f))


#############
# Self-Play #
#############
def self_play(
    agent: AlphaAgent,
    temp_change_step: int = 15,
    use_symmetry: bool = True,
) -> list[tuple[State, np.ndarray, int]]:
    """
    Agent plays a single game against itself.

    Args:
        agent (AlphaAgent): Agent
        temp_change_step (int): step where temperature switches from 1 -> 0.3
        use_symmetry (bool): whether to use symmetry when collecting samples
    Returns:
        list[tuple[State, np.ndarray, int]]: trajectory samples of single game, list of
           (state, search_policy, reward)
    """
    state = init_state()
    records = []
    step = 0
    while state.winner is None:
        temp = 1 if step < temp_change_step else 0.3
        search_policy = agent.get_search_policy(state, temp)
        records.append([state, search_policy])
        move = np.random.choice(len(search_policy), p=search_policy)
        state = play_move(state, move)
        step += 1
    winner = state.winner
    trajectory = []
    for state, policy in records:
        rew = 0 if winner == 0 else 1 if winner == state.mark else -1
        trajectory.append((state, policy, rew))
        if use_symmetry:
            state, policy = _get_symmetric(state, policy)
            trajectory.append((state, policy, rew))
    return trajectory


def _get_symmetric(state: State, policy: np.ndarray) -> tuple[State, np.ndarray]:
    """
    Taking advantage of symmetry in the game, flips the board and search policy to get
    additional training samples for free.
    """
    new_board = list(chain(*[row[::-1] for row in state.board_2d]))
    new_pi = policy[::-1]
    return State(new_board, state.mark, state.step, state.winner), new_pi


def get_self_play_samples(
    agent: AlphaAgent,
    num_games: int,
    processes: int = 1,
    temp_change_step: int = 15,
    checkpoint_size: int = 1000,
    checkpoint_dir: str | None = None,
) -> list[tuple[State, np.ndarray, int]]:
    """
    Gather trajectory samples from self-play
    Args:
        agent (AlphaAgent): Agent
        num_games (int): number of games to play
        processes (int): number of processes for multiprocessing
        temp_change_step (int): step where temperature switches from 1->0.3
        checkpoint_size (int): save samples every checkpoint_size games
        checkpoint_dir (str): directory to save samples. If None, no samples are saved
    Returns:
        list[tuple[State, np.ndarray, int]]: samples from a number of self-play games,
            list of (state, search_policy, reward)
    """
    all_samples = []
    played = 0
    while played < num_games:
        samples = []
        checkpoint_size = min(checkpoint_size, num_games - played)
        logging.info(f"Playing {checkpoint_size} games, {played}/{num_games} played.")

        # self play game
        if processes == 1:
            for _ in tqdm(range(checkpoint_size), desc="Self-play games"):
                samples.extend(self_play(agent, temp_change_step))
        else:
            pbar = tqdm(total=checkpoint_size, desc="Self-play games")
            results = []
            with Manager() as manager:
                agent.prep_multiprocess(manager)
                with Pool(processes) as pool:
                    for _ in range(checkpoint_size):
                        results.append(
                            pool.apply_async(
                                self_play,
                                args=(agent, temp_change_step),
                                callback=lambda _: pbar.update(),
                            )
                        )
                    pool.close()
                    pool.join()
            for r in results:
                samples.extend(r.get())

        # save samples
        if checkpoint_dir:
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(
                checkpoint_dir, f"{time_str}_trajectories_{checkpoint_size}.pkl"
            )
            save_trajectories(samples, save_path)
            logging.info(f"Saved checkpoint {save_path}")

        # update progress
        all_samples.extend(samples)
        played += checkpoint_size
    return all_samples


#####################
# Sample Management #
#####################
def save_trajectories(
    trajectories: list[tuple[State, np.ndarray, int]], output_path: str
) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)


def load_trajectories(input_path: str) -> list[tuple[State, np.ndarray, int]]:
    with open(input_path, "rb") as f:
        trajectories = pickle.load(f)
    return trajectories


def dedupe_trajectories(
    trajectories: Iterable[tuple[State, np.ndarray, int]]
) -> list[tuple[State, np.ndarray, int]]:
    states = {}  # type: dict[State, dict[str, list[Any]]]

    # aggregate
    for s, p, v in trajectories:
        if s not in states:
            states[s] = {"p": [], "v": []}
        states[s]["p"].append(p)
        states[s]["v"].append(v)

    # average
    return [
        (s, np.mean(values["p"], axis=0).tolist(), np.mean(values["v"]))
        for s, values in states.items()
    ]


############
# Optimize #
############
def optimize(
    model: ConnNet,
    trajectories: list[tuple[State, np.ndarray, int]],
    batch_size: int = 32,
    lr: float = 0.01,
    weight_decay: float = 0.0001,
    epochs: int = 1,
    logger: utils.MetricLogger = utils.MetricLogger(),
):
    # setup optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = model.device
    training_data = trajectories.copy()

    for epoch in tqdm(range(epochs), desc="Epoch"):
        random.shuffle(training_data)
        for i in tqdm(range(0, len(training_data), batch_size), "Batch", leave=False):
            optimizer.zero_grad()

            # get batch
            batch = training_data[i : i + batch_size]
            x = states_to_tensor([t[0] for t in batch]).to(device)
            pi = th.tensor([t[1] for t in batch]).to(device)
            reward = th.tensor([t[2] for t in batch]).to(device)

            # forward
            model.train()
            policy, value = model.forward(x)

            # calculate loss and backprop
            value_loss = ((reward - value) ** 2).mean()
            policy_loss = (-th.linalg.vecdot(pi, th.log(policy))).mean()
            loss = value_loss + policy_loss
            loss.backward()

            # update
            optimizer.step()

            # log
            logger.add_metric("epoch", epoch + 1)
            logger.add_metric("value_loss", value_loss.item())
            logger.add_metric("policy_loss", policy_loss.item())
            logger.add_metric("grad_norm", utils.get_grad_norm(model))
            logger.add_metric("loss", loss.item())
            logger.step()


######################
# Main Training Loop #
######################
def _setup_run(
    params: Params,
) -> tuple[Params, utils.MetricLogger, utils.MetricLogger, deque, ConnNet]:
    """
    Sets up the self-play training run. Continues the previous run if it exists.
    Args:
        params (Params): An instance of the Params class containing the parameters.

    Returns:
        Tuple[Params, MetricLogger, MetricLogger, deque, ConnNet]: A tuple containing:
          - params,
          - MetricLogger for loss
          - MetricLogger for self-play metrics
          - sample queue
          - model
    """
    utils.seed_all(params.seed)
    # setup multiprocessing when GPU is used
    if params.device == "cuda":
        th.multiprocessing.set_sharing_strategy("file_system")
        th.multiprocessing.set_start_method("spawn", force=True)

    # empty run_dir
    if not params.run_dir:
        utils.setup_logging(stderr=True)
        return (
            params,
            utils.MetricLogger(),
            utils.MetricLogger(),
            deque(maxlen=params.sample_queue_size),
            ConnNet(params.layers, filters=params.filters).to(params.device),
        )

    # new run
    if not os.path.exists(params.run_dir):
        os.makedirs(params.run_dir)
        params.dump(f"{params.run_dir}/params.json")
        utils.setup_logging(f"{params.run_dir}/out.log", stderr=True)
        return (
            params,
            utils.MetricLogger(),
            utils.MetricLogger(),
            deque(maxlen=params.sample_queue_size),
            ConnNet(params.layers, filters=params.filters).to(params.device),
        )

    # continue old run
    utils.setup_logging(f"{params.run_dir}/out.log", stderr=True)
    params = params.load(f"{params.run_dir}/params.json")
    logging.info(f"Resuming from {params.run_dir}")
    queue = deque(maxlen=params.sample_queue_size)
    for f in sorted(glob(params.run_dir + "/*trajectories*.pkl")):
        queue.extend(load_trajectories(f))
        logging.info(f"Loaded trajectory: {f}")
    logging.info(f"Total sample queue size: {len(queue)}")
    model = ConnNet.load(params.run_dir + "/best.model").to(params.device)
    logging.info(f"Loaded model: {params.run_dir}/best.model")
    return (
        params,
        utils.MetricLogger.load(f"{params.run_dir}/loss.csv"),
        utils.MetricLogger.load(f"{params.run_dir}/itr_metrics.csv"),
        queue,
        model,
    )


def self_play_training(params: Params):
    # Setup
    params, loss_logger, itr_logger, sample_queue, curr_model = _setup_run(params)
    logging.info(f"Training with: {params}")
    start_itr = len(itr_logger.all_kv) + 1

    # Main loop
    for i in range(start_itr, params.self_play_iterations + 1):
        loss_logger.add_metric("iteration", i)
        itr_logger.add_metric("iteration", i)
        logging.info(f"=====Iteration {i}=====")

        # gather samples
        curr_agent = AlphaAgent(
            curr_model, iterations=params.tree_iterations, c=params.cpuct
        )
        samples = get_self_play_samples(
            curr_agent,
            params.self_play_games,
            params.processes,
            params.temp_change_step,
            checkpoint_size=params.self_play_games,
            checkpoint_dir=params.run_dir,
        )
        logging.info(f"{params.self_play_games} played. Got {len(samples)} samples")
        itr_logger.add_metric("samples", len(samples))
        sample_queue.extend(samples)
        if params.dedupe_samples:
            samples = dedupe_trajectories(sample_queue)
            logging.info(
                f"Sample Queue {len(sample_queue)} deduped to {len(samples)} training samples"
            )
        itr_logger.add_metric("sample_queue", len(sample_queue))
        itr_logger.add_metric("deduped_samples", len(samples))

        # train model
        new_model = deepcopy(curr_model)
        optimize(
            new_model,
            samples,
            params.batch_size,
            params.lr,
            params.weight_decay,
            logger=loss_logger,
        )
        logging.info("Train complete, evaluating model")

        # evaluate new model vs previous model
        new_agent = AlphaAgent(
            new_model, iterations=params.tree_iterations, c=params.cpuct
        )
        prev_wins, new_wins = evaluate(
            curr_agent, new_agent, params.arena_games, params.processes
        )
        itr_logger.add_metric("wins_vs_old", new_wins)
        itr_logger.add_metric("loss_vs_old", prev_wins)
        logging.info(f"Evaluation - new/old: {new_wins}/{prev_wins}")

        # evaluate new model vs basic MCTS
        mcts_agent = MCTSAgent(c=params.cpuct)
        mcts_wins, alpha_wins = evaluate(
            mcts_agent, new_agent, params.arena_games, params.processes
        )
        itr_logger.add_metric("wins_vs_mcts", alpha_wins)
        itr_logger.add_metric("loss_vs_mcts", mcts_wins)
        logging.info(f"Evaluation - new/MCTS: {alpha_wins}/{mcts_wins}")
        itr_logger.step()

        # save latest model and update metrics csv
        if params.run_dir:
            new_model.save(f"{params.run_dir}/iteration_{i}.model")
            loss_logger.dump(f"{params.run_dir}/loss.csv")
            itr_logger.dump(f"{params.run_dir}/itr_metrics.csv")

        # update best model if better than previous model
        if new_wins / params.arena_games >= params.update_agent_threshold:
            logging.info(
                f"Model won {new_wins / params.arena_games * 100}%, updating model"
            )
            curr_model = new_model
            if params.run_dir:
                utils.symlink(
                    os.path.realpath(f"{params.run_dir}/iteration_{i}.model"),
                    f"{params.run_dir}/best.model",
                )

    return curr_model


if __name__ == "__main__":
    params = parse(Params)
    self_play_training(params)
