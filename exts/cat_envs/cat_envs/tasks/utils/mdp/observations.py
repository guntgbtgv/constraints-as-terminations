# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# from omni.isaac.lab.assets import Articulation
# from omni.isaac.lab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, quat_inv, quat_from_euler_xyz, quat_apply_inverse

if TYPE_CHECKING:
    # from omni.isaac.lab.envs import ManagerBasedEnv
    from isaaclab.envs import ManagerBasedEnv


def joint_pos(
    env: ManagerBasedEnv,
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset.find_joints(names, preserve_order=True)[0]]


def joint_vel(
    env: ManagerBasedEnv,
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset.find_joints(names, preserve_order=True)[0]]

def default_joint_pos(    
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    default_pos = asset.data.default_joint_pos[:,asset_cfg.joint_ids]
    offset = torch.zeros_like(default_pos)
    offset[:,:] = env.action_manager.cfg.joint_pos.offset 
    return offset


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



def avg_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.body_link_quat_w[:,asset_cfg.body_ids,:]
    quat_01 = quat[:,0,:]
    quat_02 = quat[:,1,:]
    # quat_12 = quat_mul(quat_inv(quat_01), quat_02)

    # rotation_angle = torch.tensor([-torch.pi/2, 0, 0], device=quat.device)
    # rotation_angle = rotation_angle.unsqueeze(0).repeat(quat[:,1,:].size(0) , 1)  
    # quat_22p = quat_from_euler_xyz(roll=rotation_angle[:,0], pitch=rotation_angle[:,1] , yaw=rotation_angle[:,2])

    # quat_02p =  quat_mul(quat_01, quat_mul(quat_12, quat_22p))
    # quat_02p = quat_mul(quat_02, quat_22p)

    avg_quat = slerp_torch(quat_01, quat_02, 0.5)

    avg_projected_gravity = quat_apply_inverse(avg_quat, asset.data.GRAVITY_VEC_W)

    return avg_projected_gravity


def trunk_range(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if not hasattr(env, "_trunk_limit_obs"):
        env._trunk_limit_obs = torch.zeros((env.scene.num_envs, 2), device=env.device)
    return env._trunk_limit_obs