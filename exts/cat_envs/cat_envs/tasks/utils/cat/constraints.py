# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

condThe functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, quat_inv, quat_from_euler_xyz, quat_apply_inverse, yaw_quat, quat_error_magnitude
import pandas as pd

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# df = pd.read_csv("robot_config_data_edited.csv")
df = pd.read_csv("robot_config_data_reversed.csv")

quat_cols = [
    col for col in df.columns
    if col.startswith("base")
]
pos_cols = [
    col for col in df.columns
    if col.startswith("q") and not col.startswith("qdot")
]
# vel_cols = [
#     col for col in df.columns
#     if col.startswith("qdot")
# ]

data_quat = torch.tensor(df[quat_cols].values, dtype=torch.float32)
data_pos = torch.tensor(df[pos_cols].values, dtype=torch.float32)
# data_vel = torch.tensor(df[vel_cols].values, dtype=torch.float32)


def track_lin_vel_xy_yaw_frame(
    env, limit: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3],)
    lin_vel_error = torch.norm(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2], dim=1)
    # ang_vel_error = torch.norm(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]) 
    return lin_vel_error - limit

def joint_position(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = torch.abs(data.joint_pos[:, asset_cfg.joint_ids]) - limit
    return cstr


def joint_position_when_moving_forward(
    env: ManagerBasedRLEnv,
    limit: float,
    velocity_deadzone: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = (
        torch.abs(data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids])
        - limit
    )
    cstr *= (
        (
            torch.abs(data.root_lin_vel_b[:, 1])
            < velocity_deadzone
        )
        .float()
        .unsqueeze(1)
    )
    cstr *= (
        (
            torch.abs(data.root_ang_vel_b[:, 2])
            < velocity_deadzone
        )
        .float()
        .unsqueeze(1)
    )               
    return cstr


def joint_torque(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = torch.abs(data.applied_torque[:, asset_cfg.joint_ids]) - limit
    return cstr


def joint_velocity(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return torch.abs(data.joint_vel[:, asset_cfg.joint_ids]) - limit


def joint_acceleration(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return torch.abs(data.joint_acc[:, asset_cfg.joint_ids]) - limit


def upsidedown(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return data.projected_gravity_b[:, 2] > limit


def contact(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.any(
        torch.max(
            torch.norm(net_contact_forces[:, :, asset_cfg.body_ids], dim=-1),
            dim=1,
        )[0]
        > 1.0,
        dim=1,
    )


def base_orientation(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return torch.norm(data.projected_gravity_b[:, :2], dim=1) - limit

def base_orientation_2(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_link_quat_w[:,asset_cfg.body_ids,:]
    quat_02 = quat[:,0,:]
    rotation_angle = torch.tensor([-torch.pi/2, 0, 0], device=quat_02.device)
    rotation_angle = rotation_angle.unsqueeze(0).repeat(quat[:,0,:].size(0) , 1)  
    quat_22p = quat_from_euler_xyz(roll=rotation_angle[:,0], pitch=rotation_angle[:,1] , yaw=rotation_angle[:,2])

    quat_0p = quat_mul(quat_02, quat_22p)
    projected_gravity = quat_apply_inverse(quat_0p, asset.data.GRAVITY_VEC_W)

    # print("orientation: ", projected_gravity)
    return projected_gravity[:, 0] - limit

def air_time(
    env: ManagerBasedRLEnv,
    limit: float,
    velocity_deadzone: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, asset_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, asset_cfg.body_ids]
    # Like in CaT
    command_more_than_limit = (
        (
            torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
            > velocity_deadzone
        )
        .float()
        .unsqueeze(1)
    )
    cstr = (limit - last_air_time) * touchdown.float() * command_more_than_limit
    return cstr


def n_foot_contact(
    env: ManagerBasedRLEnv,
    number_of_desired_feet: int,
    min_command_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_cstr = torch.abs(
        (
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, asset_cfg.body_ids], dim=-1
                ),
                dim=1,
            )[0]
            > 1.0
        ).sum(1)
        - number_of_desired_feet
    )
    command_more_than_limit = (
        torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
        > min_command_value
    ).float()
    return contact_cstr * command_more_than_limit


def joint_range(
    env: ManagerBasedRLEnv,
    limit: float,
    velocity_deadzone: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = (
        torch.abs(data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids])
        - limit
    )
    cstr *= (
        (
            # torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
            torch.norm(data.root_lin_vel_b[:, :1], dim=1)
            < velocity_deadzone
        )
        .float()
        .unsqueeze(1)
    )    
    return cstr

def joint_range_contact(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    sensor = env.scene[sensor_cfg.name]

    foot_force_z = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]  # (N, num_bodies)
    in_contact = torch.any(foot_force_z > 1.0, dim=1).float()

    return in_contact * (
        torch.sum((data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids]) ** 2, dim=1) 
        - limit
    )


def action_rate(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    # print("asset_cfg.joint_ids: ", asset_cfg.joint_ids)
    # print("111: ", env.action_manager._action)
    # print("222: ", env.action_manager._action[:, asset_cfg.joint_ids])
    return (
        env.action_manager.cfg.joint_pos.scale * torch.abs(
            env.action_manager._action #[:, asset_cfg.joint_ids]
            - env.action_manager._prev_action #[:, asset_cfg.joint_ids]
        )
        / env.step_dt
        - limit
    )


def foot_contact_force(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return (
        torch.max(torch.norm(net_contact_forces[:, :, asset_cfg.body_ids], dim=-1), dim=1)[0]
        - limit
    )

def foot_touchdown_normal_force(
    env: ManagerBasedRLEnv,
    limit: float,
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    normal_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return first_contact * (normal_contact_forces - limit)


def min_base_height(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return limit - robot.data.root_pos_w[:, 2]

def swing_foot_height(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """ minimum foot height    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    no_contact = torch.abs(
        torch.max(
            torch.norm(
                net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1
            ),
            dim=1,
        )[0]
        < 1.0
    )    
    # print("body_link_pos_w: ", asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2])
    # print("root_pos_w: ", asset.data.root_pos_w[:, 2])
    # print("diff: ", asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - asset.data.root_pos_w[:, 2].unsqueeze(1))
    return no_contact * (limit - asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2])


def keyframe_distance (
    env,
    limit: float,
    # vel_threshold: float = 1e-3,
    # acc_threshold: float = 5.0,
    trunk_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    # front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    # hind_contact_cfg:  SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    Reward the robot for matching whichever posture (extended or flexed)
    is closer to the current joint configuration, but only when feet are
    not in contact with the ground.
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]

    # current joint configuration
    q = asset.data.joint_pos[:,asset_cfg.joint_ids]  # shape: (num_envs, num_joints)
    # print("asset_cfg.joint_ids: ", asset_cfg.joint_ids)
    # print("asset.data.joint_names: " , asset.data.joint_names)
    # print("trunk: ", asset.find_joints("trunk"))
    # print("q: ", q)
    # print("asset.data.joint_pos: ", asset.data.joint_pos)

    ext_q = data_pos[3].unsqueeze(0).to(asset.device)
    flex_q = data_pos[9].unsqueeze(0).to(asset.device)

    # squared distance to each reference posture
    d_ext = torch.abs(q - ext_q)
    d_flex = torch.abs(q - flex_q)

    trunk_id = asset.find_joints(["trunk"])

    # choose the closer posture
    mask = d_ext + trunk_threshold < d_flex
    # print("mask.shape: ", mask.shape)
    # print("d_ext: ", d_ext)
    d = torch.where(mask, d_ext, d_flex)
    d = torch.sum(d ** 2, dim=1)

    # if torch.any(mask):
    #     print("=========extenstion==========" , )
    # if not torch.any(mask):
    #     print("=========flextion==========" , )

    # no-contact gate: use the selected contact bodies from the contact sensor
    # if any selected foot/body has contact force above threshold, gate is 0
    foot_force_z = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]  # (N, num_bodies)
    in_contact = torch.any(foot_force_z > 1.0, dim=1)
    airborne = (~in_contact).float()

    # exponential shaping
    cnstr = d - limit

    return airborne * cnstr

    # """
    # front feet contact  -> compare current joint pose to flex_q
    # hind feet contact   -> compare current joint pose to ext_q

    # ext_q/flex_q should be full joint-reference vectors with the same joint
    # ordering as robot.data.joint_pos[:, robot_cfg.joint_ids].
    # """
    # asset = env.scene[asset_cfg.name]
    # front_sensor = env.scene.sensors[front_contact_cfg.name]
    # hind_sensor = env.scene.sensors[hind_contact_cfg.name]

    # # current joint configuration
    # q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # qdot = asset.data.joint_vel

    # # joint-name -> index map
    # joint_names = [str(x) for x in asset.data.joint_names]
    # joint_index = {name: i for i, name in enumerate(joint_names)}

    # # HFE joints to gate on
    # hfe_names = ["HFE_FR", "HFE_FL", "HFE_HR", "HFE_HL"]
    # hfe_ids = [joint_index[name] for name in hfe_names if name in joint_index]
    # if len(hfe_ids) == 0:
    #     raise RuntimeError("No HFE joints found in robot.data.joint_names")

    # # gate: only active when all HFE joints are near zero velocity
    # hfe_speed = qdot[:, hfe_ids].abs().min(dim=1).values
    # hfe_stopped = hfe_speed <= vel_threshold


    # # reference subsets in the same joint order
    # q_ext = data_pos[3].unsqueeze(0).to(asset.device)
    # q_flex = data_pos[9].unsqueeze(0).to(asset.device)

    # # posture distances
    # d_ext = torch.sum((q - q_ext) ** 2, dim=1)
    # d_flex = torch.sum((q - q_flex) ** 2, dim=1)

    # # contact gates
    # front_contact =  torch.any(front_sensor.data.net_forces_w[:, front_contact_cfg.body_ids, 2] > 1.0, dim=1  )# (N, num_bodies)
    # hind_contact =  torch.any(hind_sensor.data.net_forces_w[:, hind_contact_cfg.body_ids, 2] > 1.0, dim=1  )# (N, num_bodies)


    # front_only = front_contact & (~hind_contact)
    # hind_only = hind_contact & (~front_contact)

    # # gated posture reward
    # r_front = front_only.float() * (d_flex - limit)
    # r_hind = hind_only.float() * (d_ext - limit)

    # return hfe_stopped * (r_front + r_hind)

    # """
    # Reward the robot for matching whichever posture (extended or flexed)
    # is closer to the current joint configuration
    # """
    # asset = env.scene[asset_cfg.name]

    # # current joint configuration
    # q = asset.data.joint_pos  # shape: (num_envs, num_joints)
    # q_ext = data_pos[3].unsqueeze(0).to(asset.device)

    # # squared distance to each reference posture
    # # d = torch.sum((q[:,None,:] - data_pos.to(asset.device)[None, :, :]) ** 2, dim=-1)
    # d = torch.sum((q - q_ext) ** 2, dim=-1)

    # # choose the closer posture
    # # d_closest, _ = torch.min(d, dim=1)
    # # exponential shaping
    # cnstr = d - limit

    # return cnstr

    # """ Reward the robot for matching whichever posture (extended or flexed)
    # is closer to the current joint configuration, but only whenever v_HFE == 0
    # """
    # asset = env.scene[asset_cfg.name]

    # # current joint configuration
    # q = asset.data.joint_pos  # shape: (num_envs, num_joints)
    # qdot = asset.data.joint_vel
    

    # # joint-name -> index map
    # joint_names = [str(x) for x in asset.data.joint_names]
    # joint_index = {name: i for i, name in enumerate(joint_names)}

    # # HFE joints to gate on
    # hfe_names = ["HFE_FR", "HFE_FL", "HFE_HR", "HFE_HL"]
    # hfe_ids = [joint_index[name] for name in hfe_names if name in joint_index]
    # if len(hfe_ids) == 0:
    #     raise RuntimeError("No HFE joints found in robot.data.joint_names")

    # # gate: only active when all HFE joints are near zero velocity
    # hfe_speed = qdot[:, hfe_ids].abs().min(dim=1).values
    # hfe_stopped = hfe_speed <= vel_threshold

  
    # ext_q = data_pos[3].unsqueeze(0).to(asset.device)
    # flex_q = data_pos[9].unsqueeze(0).to(asset.device)

    # # squared distance to each reference posture
    # d_ext = torch.sum((q - ext_q) ** 2, dim=1)
    # d_flex = torch.sum((q - flex_q) ** 2, dim=1)

    # # choose the closer posture
    # d = torch.minimum(d_ext, d_flex)

    # # no-contact gate: use the selected contact bodies from the contact sensor
    # # if any selected foot/body has contact force above threshold, gate is 0
    # end_of_stroke = asset.data.joint_vel

    # # exponential shaping
    # cnstr = d - limit

    # return hfe_stopped * cnstr

    # """
    # Reward matching extended/flexed posture based on HFE acceleration sign
    # when HFE velocity is near zero.

    # Rules:
    #   - front HFE:  acc > 0 -> extended,  acc < 0 -> flexed
    #   - hind HFE:   acc < 0 -> extended,  acc > 0 -> flexed
    # """
    # asset = env.scene[asset_cfg.name]

    # # reference subsets in the same joint order
    # q_ext = data_pos[3].unsqueeze(0).to(asset.device)
    # quat_ext = data_quat[3].unsqueeze(0).to(asset.device)
    # q_flex = data_pos[9].unsqueeze(0).to(asset.device)
    # quat_flex = data_quat[9].unsqueeze(0).to(asset.device)

    # q = asset.data.joint_pos
    # qdot = asset.data.joint_vel
    # quat = asset.data.root_quat_w

    # joint_names = [str(x) for x in asset.data.joint_names]
    # joint_index = {name: i for i, name in enumerate(joint_names)}

    # # HFE joint groups
    # front_hfe = ["HFE_FR", "HFE_FL"]
    # hind_hfe  = ["HFE_HR", "HFE_HL"]

    # front_ids = [joint_index[n] for n in front_hfe if n in joint_index]
    # hind_ids  = [joint_index[n] for n in hind_hfe  if n in joint_index]

    # if len(front_ids) == 0 or len(hind_ids) == 0:
    #     raise RuntimeError("Could not find one or more HFE joints in asset.data.joint_names")

    # # current posture distances
    # d_ext = torch.sum((q - q_ext) ** 2, dim=1) + quat_error_magnitude(quat_ext.repeat(quat.size(0), 1) , quat)  
    # d_flex = torch.sum((q - q_flex) ** 2, dim=1) + quat_error_magnitude(quat_flex.repeat(quat.size(0), 1), quat)

    # # group averages
    # # front_vel = qdot[:, front_ids].abs().min(dim=1).values
    # hind_vel  = qdot[:, hind_ids].abs().min(dim=1).values

    # # front_acc = asset.data.joint_acc[:, front_ids].mean(dim=1)
    # hind_acc  = asset.data.joint_acc[:, hind_ids].mean(dim=1)

    # # front_stopped = front_vel <= vel_threshold
    # hind_stopped  = hind_vel <= vel_threshold

    # # choose posture based on your sign rules
    # # front_use_ext  = front_stopped & (front_acc > acc_threshold)
    # # front_use_flex = front_stopped & (front_acc < -acc_threshold)

    # hind_use_ext   = hind_stopped & (hind_acc < -acc_threshold)
    # hind_use_flex  = hind_stopped & (hind_acc > acc_threshold)

    # # score the chosen posture
    # # r_front = torch.zeros_like(d_ext)
    # r_hind = torch.zeros_like(d_ext)

    # # r_front = torch.where(front_use_ext, d_ext - limit , r_front)
    # # r_front = torch.where(front_use_flex, d_flex - limit , r_front)

    # r_hind = torch.where(hind_use_ext,  d_ext - limit , r_hind)
    # r_hind = torch.where(hind_use_flex, d_flex - limit , r_hind)

    # return r_hind #r_front +






def no_move(
    env: ManagerBasedRLEnv,
    velocity_deadzone: float,
    joint_vel_limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr_nomove = (torch.abs(data.joint_vel[:, asset_cfg.joint_ids]) - joint_vel_limit) * (
        torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
        < velocity_deadzone
    ).float().unsqueeze(1)
    return cstr_nomove

def slerp_torch(q1: torch.Tensor,
                q2: torch.Tensor,
                tau: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Vectorized, GPU-friendly SLERP for quaternions.

    Args:
      q1, q2: tensors with shape (..., 4) in (w,x,y,z) order
      tau: scalar float or tensor broadcastable to q1.shape[:-1]

    Returns:
      Tensor of shape (..., 4) (same leading shape as inputs), unit quaternions (w,x,y,z).
    """
    # Basic validation
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError("Last dimension of q1 and q2 must be 4 (w,x,y,z)")

    # Promote tau to tensor on same device/dtype as q1
    device = q1.device
    dtype = q1.dtype
    if not torch.is_tensor(tau):
        t = torch.tensor(float(tau), dtype=dtype, device=device)
    else:
        t = tau.to(device=device, dtype=dtype)

    # Broadcast q1 and q2 to the same shape for leading dims
    q1, q2 = torch.broadcast_tensors(q1, q2)  # same shape (...,4)
    leading_shape = q1.shape[:-1]

    # Flatten leading dims to (B,4) for efficient ops
    q1f = q1.reshape(-1, 4)
    q2f = q2.reshape(-1, 4)
    B = q1f.shape[0]

    # Prepare t to shape (B,) or broadcastable to (B,1)
    if t.ndim == 0:
        t_b = t.expand(B)
    else:
        # try to broadcast t to leading dims
        t_b = t.reshape(-1)
        if t_b.numel() == 1:
            t_b = t_b.expand(B)
        elif t_b.numel() != B:
            # let broadcasting rules apply (e.g., t had shape leading_shape)
            # so reshape to the leading_shape flattened
            try:
                t_b = t.reshape(*leading_shape).reshape(-1)
            except Exception as e:
                raise ValueError(f"tau shape {t.shape} is not broadcastable to inputs' leading shape {leading_shape}") from e

    # compute dot product (B,)
    dot = (q1f * q2f).sum(dim=-1)  # dot product per pair

    # shortest path: if dot < 0, negate q2
    neg_mask = dot < 0.0
    q2f = torch.where(neg_mask.unsqueeze(-1), -q2f, q2f)
    dot = torch.where(neg_mask, -dot, dot)

    # clamp dot to valid domain for acos
    dot = torch.clamp(dot, -1.0, 1.0)

    # compute angle and sin(angle)
    angle = torch.acos(dot)              # (B,)
    sin_angle = torch.sin(angle)         # (B,)

    # numeric thresholds
    eps = torch.finfo(dtype).eps
    tiny = eps * 4.0

    # compute coefficients safely; avoid divide-by-zero
    # s0 = sin((1-t)*angle) / sin(angle)
    # s1 = sin(t*angle) / sin(angle)
    # shape t_b to (B,1)
    t_b = t_b.to(device=device, dtype=dtype)
    s0 = torch.sin((1.0 - t_b).unsqueeze(-1) * angle.unsqueeze(-1)) / (sin_angle.unsqueeze(-1) + 1e-12)
    s1 = torch.sin(t_b.unsqueeze(-1) * angle.unsqueeze(-1)) / (sin_angle.unsqueeze(-1) + 1e-12)

    out = s0 * q1f + s1 * q2f   # (B,4)

    # For very small angles, sin(angle) ~ 0 -> fallback to linear interpolation (lerp)
    small = (sin_angle.abs() < tiny)
    if small.any():
        lerp = ((1.0 - t_b).unsqueeze(-1) * q1f) + (t_b.unsqueeze(-1) * q2f)
        out = torch.where(small.unsqueeze(-1), lerp, out)

    # normalize to unit length to avoid drift
    out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    # reshape back to original leading dims
    out = out.reshape(*leading_shape, 4)
    return out

def flat_orientation_articulate_trunk(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # print("asset: " , asset)
    # print("projected_gravity_b: ", asset.data.projected_gravity_b)

    quat = asset.data.body_link_quat_w[:,asset_cfg.body_ids,:]
    quat_01 = quat[:,0,:]
    quat_02 = quat[:,1,:]
    # quat_12 = quat_mul(quat_inv(quat_01), quat_02)

    rotation_angle = torch.tensor([-torch.pi/2, 0, 0], device=quat.device)
    rotation_angle = rotation_angle.unsqueeze(0).repeat(quat[:,1,:].size(0) , 1)  
    quat_22p = quat_from_euler_xyz(roll=rotation_angle[:,0], pitch=rotation_angle[:,1] , yaw=rotation_angle[:,2])

    # quat_02p =  quat_mul(quat_01, quat_mul(quat_12, quat_22p))
    quat_02p = quat_mul(quat_02, quat_22p)

    # print("rotation trunk: ",  euler_xyz_from_quat(quat_mul(quat_inv(quat_01), quat_02p)))




    # avg_quat = batch_quat_slerp(quat_slerp, quat_01, quat_02p, 0.5)
    avg_quat = slerp_torch(quat_01, quat_02p, 0.5)


    avg_projected_gravity = quat_apply_inverse(avg_quat, asset.data.GRAVITY_VEC_W)
    # print("asset.data.GRAVITY_VEC_W: ", asset.data.GRAVITY_VEC_W)
    # print("asset.data.projected_gravity_b: ", asset.data.projected_gravity_b)
    # print("avg_projected_gravity: ", avg_projected_gravity)
    # trunk = asset.data.joint_pos[:,asset_cfg.joint_ids]
    # print("trunk: ", trunk )
       
    return torch.norm(avg_projected_gravity[:, :2], dim=1) - limit