from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
import isaaclab.utils.string as string_utils
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import JointPositionActionCfg

# If you already have a JointPositionActionCfg, inherit from it.
# This version adds the trunk-mode parameters only.
@configclass
class PerEnvJointPositionActionCfg(JointPositionActionCfg):

    trunk_mode_command_name: str = "trunk_mode"
    trunk_joint_name: str = "trunk_joint"
    fixed_trunk_scale: float = 1.0e-6
    articulated_trunk_scale: float = 0.3


class PerEnvJointPositionAction(ActionTerm):
    cfg: "PerEnvJointPositionActionCfg"    
    _asset: Articulation

    def __init__(self, cfg: PerEnvJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)

        self._env = env
        self._asset = env.scene[cfg.asset_name]

        # resolve joints for this action term
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)

        # trunk joint index inside the action vector
        self._trunk_joint_idx = self._joint_names.index(self.cfg.trunk_joint_name)

        # raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self._num_joints, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.scale, self._joint_names
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}")

        # offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.offset, self._joint_names
            )
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}")

        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # clip
        self._clip = None
        if self.cfg.clip is not None:
            self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self._num_joints, 1
            )
            if isinstance(cfg.clip, dict):
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}")

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # store raw actions
        self._raw_actions[:] = actions

        # per-env trunk mode comes from command manager
        trunk_mode = self._env.command_manager.get_command(self.cfg.trunk_mode_command_name)
        # expected shape: (num_envs, 1) or (num_envs,)
        if trunk_mode.ndim == 2:
            trunk_mode = trunk_mode[:, 0]

        trunk_scale = torch.where(
            trunk_mode < 0.5,
            torch.full((self.num_envs,), self.cfg.fixed_trunk_scale, device=self.device),
            torch.full((self.num_envs,), self.cfg.articulated_trunk_scale, device=self.device),
        )

        # update only the trunk column, per env
        if isinstance(self._scale, torch.Tensor):
            self._scale[:, self._trunk_joint_idx] = trunk_scale
        else:
            # if scale is scalar, promote to tensor here
            self._scale = torch.full((self.num_envs, self._num_joints), float(self._scale), device=self.device)
            self._scale[:, self._trunk_joint_idx] = trunk_scale

        # affine transform
        self._processed_actions = self._raw_actions * self._scale + self._offset

        # clip
        if self._clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        # send position targets to the robot
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

PerEnvJointPositionActionCfg.class_type = PerEnvJointPositionAction

# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedEnv

#     from isaaclab.envs.mdp import actions_cfg


# class JointRandomPositionAction(JointAction):
#     """Joint action term that applies the processed actions to the articulation's joints as position commands."""

#     cfg: actions_cfg.JointPositionActionCfg
#     """The configuration of the action term."""

#     def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
#         # initialize the action term
#         super().__init__(cfg, env)
#         # use default joint positions as offset
#         if cfg.use_default_offset:
#             self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

#     def apply_actions(self):
#         # set position targets
#         self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

