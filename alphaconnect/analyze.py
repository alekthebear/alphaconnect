"""Helper functions for analysis in jupyter notebook"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(loss_path):
    df_optim = pd.read_csv(loss_path)
    itr_changes = []
    for i in range(1, df_optim.iteration.max() + 1):
        itr_changes.append(df_optim[df_optim.iteration == i].index[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # plot loss
    ax1.set_title("Loss")
    ax1.plot(
        df_optim.index,
        df_optim[["value_loss", "policy_loss", "loss"]],
        label=["value_loss", "policy_loss", "loss"],
    )
    ax1.legend()
    ax1.vlines(x=itr_changes, ymin=0, ymax=3, color="r", ls=(0, (1, 10)))
    # plot grad norm
    ax2.set_title("Grad Norm")
    ax2.plot(df_optim.index, df_optim.grad_norm, label="grad_norm")
    ax2.legend()
    ax2.vlines(x=itr_changes, ymin=0, ymax=3, color="r", ls=(0, (1, 10)))
    return fig


def plot_itr(itr_metrics_path):
    df_itr = pd.read_csv(itr_metrics_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # plot samples
    ax1.set_title("Sample Sizes")
    ax1.plot(
        df_itr.index,
        df_itr[["sample_queue", "deduped_samples"]],
        label=["sample_queue", "deduped_samples"],
    )
    ax1.legend()

    # plot win rates
    ax2.set_title("Win Rates")
    df_itr["rate_vs_old"] = df_itr.wins_vs_old / (
        df_itr.wins_vs_old + df_itr.loss_vs_old
    )
    df_itr["rate_vs_mcts"] = df_itr.wins_vs_mcts / (
        df_itr.wins_vs_mcts + df_itr.loss_vs_mcts
    )
    ax2.plot(
        df_itr.index,
        df_itr[["rate_vs_old", "rate_vs_mcts"]],
        label=["win % vs prev", "win % vs MCTS"],
    )
    ax2.legend()
    return fig
