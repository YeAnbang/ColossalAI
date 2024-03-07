#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper functions for IO save load checkpoints
"""

import json
import os
from typing import Any, Dict, Tuple, Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator


def load_json(file_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load file in JSON format
    """
    with open(file=file_path, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(data: Dict[str, Any], file_path: Union[str, os.PathLike]) -> None:
    """
    Save as JSON format
    """
    with open(file=file_path, mode="w", encoding="utf-8") as fp:
        json.dump(data, fp=fp, ensure_ascii=False, indent=4)


def save_checkpoint(
    save_dir: Union[str, os.PathLike],
    booster: Booster,
    model: torch.nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    step: int,
    batch_size: int,
    coordinator: DistCoordinator,
    max_number_of_checkpoints: int = 3,
) -> None:
    """
    Save model checkpoint, optimizer, LR scheduler and intermedidate running states.
    """
    
    save_dir = os.path.join(save_dir, f"epoch-{epoch}_step-{step}")
    os.makedirs(os.path.join(save_dir, "modeling"), exist_ok=True)

    booster.save_model(model, os.path.join(save_dir, "modeling"), shard=True)

    if max_number_of_checkpoints > 0:
        import glob
        existing_checkpoints = []
        for dir in glob.glob(os.path.join(save_dir, "epoch-*_step-*")):
            epoch_step = dir.split("/")[-1]
            epoch, step = epoch_step.split("_")
            existing_checkpoints.append((int(epoch.split("-")[-1]), int(step.split("-")[-1])))
        # remove the oldest checkpoint if the number of checkpoints exceeds the maximum
        if len(existing_checkpoints) > max_number_of_checkpoints:
            oldest_checkpoint = sorted(existing_checkpoints, key=lambda x: (x[0],x[1]))
            os.system(f"rm -r {os.path.join(save_dir, f'epoch-{oldest_checkpoint[0]}_step-{oldest_checkpoint[1]}')}")

    '''
    Temporary disable the following as save_optimizer causes all processes to hang in a multi-gpu environment, 
    working on fixing this bug
    '''

    booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True)
    booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    running_states = {
        "epoch": epoch,
        "step": step,
        "sample_start_index": step * batch_size,
    }
    if coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, "running_states.json"))


def load_checkpoint(
    load_dir: Union[str, os.PathLike],
    booster: Booster,
    model: torch.nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
) -> Tuple[int, int, int]:
    """
    Load model checkpoint, optimizer, LR scheduler and intermedidate running states.
    """

    # Update booster params states.
    booster.load_model(model=model, checkpoint=os.path.join(load_dir, "modeling"))
    booster.load_optimizer(optimizer=optimizer, checkpoint=os.path.join(load_dir, "optimizer"))
    booster.load_lr_scheduler(lr_scheduler=lr_scheduler, checkpoint=os.path.join(load_dir, "lr_scheduler"))

    running_states = load_json(file_path=os.path.join(load_dir, "running_states.json"))
    return (
        running_states["epoch"],
        running_states["step"],
        running_states["sample_start_index"],
    )
