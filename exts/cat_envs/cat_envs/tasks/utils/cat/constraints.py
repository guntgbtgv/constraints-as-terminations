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
from isaaclab.utils.math import quat_mul, quat_inv, quat_from_euler_xyz, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def track_lin_vel_xy_yaw_frame(
    env, limit: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
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
            torch.abs(env.command_manager.get_command("base_velocity")[:, 1])
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
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return (
        torch.abs(data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids])
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


def min_base_height(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return limit - robot.data.root_pos_w[:, 2]

def foot_height(
    env: ManagerBasedRLEnv,
    # limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """ maximum foot height    """
    asset = env.scene[asset_cfg.name]
    # print("body_link_pos_w: ", asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2])
    # print("root_pos_w: ", asset.data.root_pos_w[:, 2])
    # print("diff: ", asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - asset.data.root_pos_w[:, 2].unsqueeze(1))
    return asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - asset.data.root_pos_w[:, 2].unsqueeze(1)



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
    quat_12 = quat_mul(quat_inv(quat_01), quat_02)

    rotation_angle = torch.tensor([-torch.pi/2, 0, 0], device=quat.device)
    rotation_angle = rotation_angle.unsqueeze(0).repeat(quat[:,1,:].size(0) , 1)  
    quat_22p = quat_from_euler_xyz(roll=rotation_angle[:,0], pitch=rotation_angle[:,1] , yaw=rotation_angle[:,2])

    quat_02p =  quat_mul(quat_01, quat_mul(quat_12, quat_22p))

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