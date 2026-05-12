# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import pandas as pd
from typing import TYPE_CHECKING


import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

df = pd.read_csv("inference_log_20260506_141531.csv")

pos_cols = [col for col in df.columns if "joint_pos" in col]
vel_cols = [col for col in df.columns if "joint_vel"in col]

data_pos = torch.tensor(df[pos_cols].values, dtype=torch.float32)
data_vel = torch.tensor(df[vel_cols].values, dtype=torch.float32)


def randomize_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    max_displacement: float,
    asset_cfg: SceneEntityCfg,
):
    """Randomize the CoM of the bodies by adding a random value sampled from the given range.

    .. tip::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()[:, body_ids, :3]

    # Randomize the com in range -max displacement to max displacement
    coms += torch.rand_like(coms) * 2 * max_displacement - max_displacement

    # Set the new coms
    new_coms = asset.root_physx_view.get_coms().clone()
    new_coms[:, asset_cfg.body_ids, 0:3] = coms
    asset.root_physx_view.set_coms(new_coms, env_ids)


def push_by_setting_velocity_with_random_envs(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """pushing function ported from isaacgym CaT"""

    p_push = env.physics_dt / (
        env.max_episode_length_s * 2
    )  # <- time step / duration of X seconds
    # There will be a probability of 0.63 of having at least one swap after X seconds have elapsed
    # (1 / p) policy steps for X seconds, and the probability of having no swap at all is (1 - p)**(1 / p) = 0.37
    # The mean number of swaps for (1 / p) steps with probability p is 1.
    push_idx = (
        torch.bernoulli(torch.full((env.num_envs,), p_push, device=env.device))
        .nonzero(as_tuple=False)
        .flatten()
    )

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[push_idx]

    # sample random velocities
    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w[:] = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device
    )

    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=push_idx)

def randomize_joint_position_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # # cast env_ids to allow broadcasting
    # if asset_cfg.joint_ids != slice(None):
    #     iter_env_ids = env_ids[:, None]
    # else:
    #     iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos[0, asset_cfg.joint_ids] # [iter_env_ids, asset_cfg.joint_ids].clone()
    rand_pos = math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    env.action_manager.cfg.joint_pos.offset = joint_pos + rand_pos

def reset_joints_from_dataset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    sample_from_dataset: bool,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]


    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    if sample_from_dataset is False:
        # get default joint state
        joint_pos = asset.data.default_joint_pos[0, asset_cfg.joint_ids] # [iter_env_ids, asset_cfg.joint_ids].clone()
        joint_vel = asset.data.default_joint_vel[0, asset_cfg.joint_ids]

        # bias these values randomly
        joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
        joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

        # set into the physics simulation
        asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

    else:
        
        # data_pos = torch.tensor(
        #    [[-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, 1.0, -0.1, 0.5, -1.5, 0.1, 0.5, -1.5],   # crouched
        #    [-0.1, -0.6, -0.7, 0.1, -0.6, -0.7, 0.0, -0.1, 1.2, -0.7, 0.1, 1.2, -0.7],  # stretched 
        #    [-0.1, 0.4, -1.5, 0.1, 0.4, -1.5, 0.5, -0.1, 1.0, -2.0, 0.1, 1.0, -2.0],    # foreleg landing
        #    [-0.1, 0.7, -2.0, 0.1, 0.7, -2.0, 0.5, -0.1, 1.2, -1.5, 0.1, 1.2, -1.5]],   # hindleg landing
        #     # [[-0.1624112, 0.87693184, -1.87309827,  0.17673672,  1.21096103, -1.99519666, 0.17429623, -0.09178511,  1.42467002, -1.41669874,  0.17207183,  1.43940305, -1.6546099],
        #     # [ 0.06642797,  1.0939935,  -1.86104492, -0.05793188,  1.06444465, -1.78327756,  0.48264497, -0.32301549,  0.82283317, -1.64895404,  0.15263972,  0.90697448, -1.96417398],
        #     # [-0.25722814,  0.5237372,  -0.26222761,  0.26308583,  0.79660199, -0.60131166, 0.97865541, -0.01853016,  1.04324715, -1.76880212,  0.26597811,  1.44931495, -1.66382989],
        #     # [0.11417998, 1.14053238, -1.60934173, -0.11077342, 1.19959023, -1.64520811, 1.01634028, -0.2012674, 0.23723502, -1.75008844, 0.16435368, 0.65671855, -1.95483608]],
        #     device=asset.device
        # )
        # data_vel = torch.tensor(
        #     [[0.0, -2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0],        # crouched
        #     [0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, -2.0, 0.0],         # stretched 
        #     [0.0, 9.0, 0.0, 0.0, 9.0, 0.0, 9.0, 0.0, -9.0, 0.0, 0.0, -9.0, 0.0 ],      # foreleg landing
        #     [0.0, -9.0, 0.0, 0.0, -9.0, 0.0, -9.0, 0.0, 9.0, 0.0, 0.0, 9.0, 0.0 ]],    # hindleg landing
        #     # [[3.29014534, 1.17095511, -6.17838644, -3.1902238,  -3.01711325, -2.61997878, -1.22372721, -3.76939205, -7.77193545, -2.14695672,  1.09846187, -6.23008414, -6.40834376],
        #     # [2.31035856,  1.10325923,  5.88667664, -1.02297241,  2.58863774,  2.00665355,  7.42084898,  0.11992166, -8.66487777, -3.96042557, -0.51109221, -6.53305762, -1.76475165],
        #     # [1.34005205,   7.53602257,  -8.5629131,    1.67742256,  10.09278064, -10.47113429,  -6.96449862,  -0.1595156,   10.35210636,  -1.58605433, 0.08978622,   4.75888361,   6.66102322,],
        #     # [-0.67923843, -0.03169564, 7.04900771, -0.05435304, -0.7100362, 7.36513913, 9.43807574, 5.19793801, -8.58880731, 3.3028977, 0.50372767, -4.16085127, 1.93999803,]],   
        #     device=asset.device
        # )    

        idx = torch.randint(0, data_pos.shape[0], (1,))



        # print("joint_pos: ", joint_pos)
        # print("sampled: ", data_pos[idx].squeeze(0))
        asset.write_joint_state_to_sim(data_pos[idx].squeeze(0).to(asset.device)+math_utils.sample_uniform(*position_range, data_pos[0].shape, asset.device), data_vel[idx].squeeze(0).to(asset.device), joint_ids=asset_cfg.joint_ids, env_ids=env_ids)




        # joint_angles:  [ 4.92336945e+01 -5.14248315e+00  2.97231450e-01  9.73480036e-01
        # -2.30845866e-02  2.21086487e-01  5.40785315e-02  1.20737664e-01
        # 1.13101536e+00 -1.59079371e+00 -1.14052001e-01  1.15839809e+00
        # -1.60881009e+00  1.02984994e+00 -1.84414725e-01  1.97587035e-01
        # -1.75921256e+00  1.64890568e-01  6.95160990e-01 -1.98583038e+00]
        # joint_velocities:  [ 2.45503893  0.26659918  0.17451916  0.118404    2.86319757  0.39863799
        # -0.57051796 -0.82332736  7.88181603  0.24807584 -1.23609158  7.96598991
        # 9.69396239  5.40719267 -8.4441687   3.7630734   0.43969313 -3.72578966
        # 1.84832256]


        # joint_angles:  [ 4.89238258e+01 -5.18370080e+00  3.15904745e-01  9.97097197e-01
        # -4.34257182e-02  2.04730064e-02  5.90951944e-02 -1.06021329e-01
        # 9.11998053e-01 -1.98415867e+00  1.67598886e-01  1.15824963e+00
        # -2.04141262e+00  1.48664899e-01 -1.75304130e-01  1.36083316e+00
        # -1.43976075e+00  2.24771751e-01  1.36295593e+00 -1.70823472e+00]
        # joint_velocities:  [ 3.05735034  0.43966555 -0.22362145 -0.1668198   0.31944613  0.04179527
        # 4.677178    2.30917086 -5.98619831 -4.92200511 -3.04832689 -1.84714189
        # 0.32820637 -4.55866778 -8.79733963 -3.86116575  0.73931606 -6.33134305
        # -6.55882155]


