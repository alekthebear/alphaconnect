from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn

from alphaconnect.game import State, Env


class ResBlock(nn.Module):
    def __init__(self, filters, kernel=3, stride=1, padding="same"):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel, stride, padding)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel, stride, padding)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.ReLU()

    def forward(self, x) -> th.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = x + y
        y = self.relu2(y)
        return y


class ConnNet(nn.Module):
    def __init__(
        self,
        layers,
        in_filters=1,
        filters=128,
        kernel=3,
        stride=1,
        padding="same",
        policy_filters=1,
        value_filters=1,
        value_head_hidden_size=128,
    ):
        super().__init__()
        # save hyperparameters for saving/loading model
        self.hyperparameters = {
            k: v for k, v in locals().items() if k not in ["self", "__class__"]
        }

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_filters, filters, kernel, stride, padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList(
            [ResBlock(filters, kernel, stride, padding) for _ in range(layers)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, policy_filters, 1, 1, "same"),
            nn.BatchNorm2d(policy_filters),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(Env.columns * Env.rows * policy_filters, Env.columns),
            nn.Softmax(dim=1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, value_filters, 1, 1, "same"),
            nn.BatchNorm2d(value_filters),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(Env.columns * Env.rows * value_filters, value_head_hidden_size),
            nn.ReLU(),
            nn.Linear(value_head_hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Expected x dimensions: (batch_size, in_filters, rows, cols)"""
        x = x.to(self.device)
        x = self.conv_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value.squeeze(-1)

    def predict(self, state: State) -> tuple[np.ndarray, float]:
        x = state_to_tensor(state)
        x = x.unsqueeze(0).to(self.device)
        if self.training:
            self.eval()
        with th.no_grad():
            policy, value = self(x)
            return policy.squeeze(0).cpu().numpy(), value.squeeze(0).item()

    @property
    def device(self):
        return next(self.parameters()).device

    def save(self, file_path):
        th.save([self.hyperparameters, self.state_dict()], file_path)

    @classmethod
    def load(cls, file_path):
        hyperparameters, state_dict = th.load(file_path)
        model = cls(**hyperparameters)
        model.load_state_dict(state_dict)
        return model


def state_to_tensor(state: State) -> th.Tensor:
    """
    Returns a [1 x row x col] tensor of the game board with signed values based on the
    current mark.
    Each element in the board is either 0, 1, or -1.
    - If the position is empty, it is converted to 0.
    - If the position is occupied by the current player, it is converted to 1.
    - If the position is occupied by the opponent, it is converted to -1.
    """
    board = [0 if p == 0 else 1 if p == state.mark else -1 for p in state.board]
    return th.Tensor(board).reshape(Env.rows, Env.columns).unsqueeze(0)


def states_to_tensor(states: list[State]) -> th.Tensor:
    return th.stack([state_to_tensor(s) for s in states])
