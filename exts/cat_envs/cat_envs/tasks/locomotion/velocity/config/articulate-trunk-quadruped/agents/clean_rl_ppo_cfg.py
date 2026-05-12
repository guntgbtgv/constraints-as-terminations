# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from cat_envs.tasks.utils.cleanrl.rl_cfg import CleanRlPpoActorCriticCfg


@configclass
class ArticulateTrunkQuadrupedFlatPPORunnerCfg(CleanRlPpoActorCriticCfg):
    save_interval = 50

    learning_rate = 1.0e-3
    num_steps = 24
    num_iterations = 4000
    gamma = 0.99
    gae_lambda = 0.95
    updates_epochs = 5
    minibatch_size = 16384
    clip_coef = 0.2
    ent_coef = 0.001
    vf_coef = 0.2
    max_grad_norm = 1.0
    norm_adv = True
    clip_vloss = True
    anneal_lr = False

    experiment_name = "articulate-trunk-quadruped_flat"
    logger = "tensorboard"
    wandb_project = "articulate-trunk-quadruped_flat"

    load_run = ".*"
    load_checkpoint = "model_.*.pt"
