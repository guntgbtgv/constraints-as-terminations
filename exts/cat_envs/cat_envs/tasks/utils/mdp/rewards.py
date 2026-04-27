from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, quat_mul, euler_xyz_from_quat, quat_inv, quat_apply_inverse, quat_from_euler_xyz, quat_slerp
from typing import Callable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def foot_power(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward penalizing sliding by multiplying contact force magnitudes with foot linear velocities.
    """
    # Get the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Latest net contact forces in world frame [num_envs, num_bodies, 3]
    contacts_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    # Corresponding asset body velocities in world frame [num_envs, num_bodies, 3]
    asset = env.scene[asset_cfg.name]
    feet_vel_w = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :]

    rew_feet_pow = torch.sum(torch.abs(torch.sum(contacts_w * feet_vel_w, dim=2)), dim=1)
    return rew_feet_pow


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)

def track_lin_vel_xy_yaw_frame_artuculate_trunk_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    quat = asset.data.body_link_quat_w[:,asset_cfg.body_ids,:]
    quat_01 = quat[:,0,:]
    quat_02 = quat[:,1,:]
    quat_12 = quat_mul(quat_inv(quat_01), quat_02)

    rotation_angle = torch.tensor([-torch.pi/2, 0, 0], device=quat.device)
    rotation_angle = rotation_angle.unsqueeze(0).repeat(quat[:,1,:].size(0) , 1)  
    quat_22p = quat_from_euler_xyz(roll=rotation_angle[:,0], pitch=rotation_angle[:,1] , yaw=rotation_angle[:,2])

    quat_02p =  quat_mul(quat_01, quat_mul(quat_12, quat_22p))

    vel_yaw_1 = quat_apply_inverse(yaw_quat(quat_01), asset.data.body_lin_vel_w[:, 0, :3])
    vel_yaw_2 = quat_apply_inverse(yaw_quat(quat_02p), asset.data.body_lin_vel_w[:, 1, :3])
    vel_yaw_avg = 0.5*(vel_yaw_1 + vel_yaw_2)

    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw_avg[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum((qvel * qfrc), dim=1)
    # return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

def energy_new(env: ManagerBasedRLEnv, std_lin: float, std_ang: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):

    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]

    energy_consume = torch.sum((qvel * qfrc), dim=1)
    divider_lin = std_lin * torch.abs(asset.data.root_lin_vel_b[:,0])
    divider_ang = std_ang * torch.abs(asset.data.root_ang_vel_b[:,2])
    divider = divider_ang + divider_lin
    # if (torch.exp(-energy_consume / divider) > 1.0e+5).any().item(): 
    #     print("reward: ", torch.exp(-energy_consume / divider) )
    return energy_consume / divider
    # return torch.exp(-energy_consume / divider)

def power_loss(env: ManagerBasedRLEnv, K: float, Coulomb: float, viscous: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 
    
    K = K_T^2 / R : motor coil resistance and torque constant (N m s)
    Coulomb    : Coulomb fricction (N m)
    viscous    : Viscous friction coefficient (N m s)
    
    """
    asset: Articulation = env.scene[asset_cfg.name]

    qdot = asset.data.joint_vel[:, asset_cfg.joint_ids]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]

    friction = Coulomb*torch.sign(qdot) + viscous*qdot

    loss_f = friction*qdot
    loss_J = 1/K*(tau + friction)**2

    divider_lin = 1000.0 * torch.abs(asset.data.root_lin_vel_b[:,0])
    divider_ang = 0.0 * torch.abs(asset.data.root_ang_vel_b[:,2])
    divider = divider_ang + divider_lin

    return torch.exp(-torch.sum(loss_f + loss_J, dim=1) / divider)

def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)

def trunk_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data

    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    pen = torch.linalg.norm((data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:,asset_cfg.joint_ids]), dim=1)
    return pen


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )

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



def flat_orientation_articulate_trunk_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
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

    return torch.sum(torch.square(avg_projected_gravity[:, :2]), dim=1)    