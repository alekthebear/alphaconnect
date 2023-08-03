from copy import deepcopy
import logging
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import wandb


###################
# Pytorch Helpers #
###################
def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def seed_all(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def to_tensor(value, device=None, dtype=torch.float):
    device = device or get_device()
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.tensor(value, dtype=dtype).to(device)


def get_model_size(model: torch.nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_bytes = param_size + buffer_size
    print("model size: {:.3f}MB".format(size_all_bytes / (1024**2)))


def get_grad_norm(model: torch.nn.Module):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


###########
# Logging #
###########
class MetricLogger:
    def __init__(self, use_wandb=False, clear_on_step=False):
        self.curr_kv = {}
        self.all_kv = []
        self.wandb = use_wandb
        self.clear_on_step = clear_on_step

    def add_metric(self, key, value):
        self.curr_kv[key] = value

    def step(self):
        self.curr_kv["timestamp"] = pd.Timestamp.now()
        self.all_kv.append(deepcopy(self.curr_kv))
        if self.wandb:
            wandb.log(self.curr_kv)
        if self.clear_on_step:
            self.curr_kv = {}

    def to_df(self):
        return pd.DataFrame(self.all_kv)

    def dump(self, path):
        self.to_df().to_csv(path, index=False)

    @staticmethod
    def load(path):
        logger = MetricLogger()
        logger.all_kv = pd.read_csv(path).to_dict(orient="records")
        return logger


def set_log_level(log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)


def setup_logging(output_path=None, level=logging.INFO, stderr=False):
    # clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Setup logging
    logging.basicConfig(
        filename=output_path,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        encoding="utf-8",
        level=level,
    )
    if stderr:
        logging.getLogger().addHandler(logging.StreamHandler())


##############
# Filesystem #
##############
def symlink(src, dst, overwrite=True):
    if os.path.islink(dst) and overwrite is True:
        tmp_link = f"{dst}.tmp"
        os.symlink(src, tmp_link)
        os.rename(tmp_link, dst)
    else:
        os.symlink(src, dst)


def natural_sort(files):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(files, key=alphanum_key)
