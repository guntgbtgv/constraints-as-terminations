# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from dataclasses import MISSING


class UniformVelocityCommandWithDeadzone(mdp.UniformVelocityCommand):
    """velocity command sampling class ported from isaacgym CaT"""

    cfg: "UniformVelocityCommandWithDeadzoneCfg"

    def __init__(
        self, cfg: "UniformVelocityCommandWithDeadzoneCfg", env: ManagerBasedEnv
    ):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)

        self.velocity_deadzone = cfg.velocity_deadzone
        self.dt = env.physics_dt
        self.max_episode_length_s = env.max_episode_length_s

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        # set small commands to zero
        self.vel_command_b *= (
            torch.any(
                torch.abs(self.vel_command_b[:, :3]) > self.velocity_deadzone, dim=1
            )
        ).unsqueeze(1)

        # Random velocity command resampling
        no_vel_command = (
            torch.norm(self.vel_command_b[:, :3], dim=1) < self.velocity_deadzone
        ).float()
        p_resample_command = 0.01 * no_vel_command + (
            self.dt / self.max_episode_length_s
        ) * (1 - no_vel_command)
        resample_command_idx = (
            torch.bernoulli(p_resample_command).nonzero(as_tuple=False).flatten()
        )
        if len(resample_command_idx) > 0:
            self._resample(resample_command_idx)

        # Random angular velocity inversion during the episode to avoid having the robot moving in circle
        p_ang_vel = (
            self.dt / self.max_episode_length_s
        )  # <- time step / duration of X seconds
        # There will be a probability of 0.63 of having at least one swap after X seconds have elapsed
        # (1 / p) policy steps for X seconds, and the probability of having no swap at all is (1 - p)**(1 / p) = 0.37
        # The mean number of swaps for (1 / p) steps with probability p is 1.
        self.vel_command_b[:, 2] *= (
            1
            - 2
            * torch.bernoulli(
                torch.full_like(self.vel_command_b[:, 2], p_ang_vel)
            ).float()
        )


@configclass
class UniformVelocityCommandWithDeadzoneCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = UniformVelocityCommandWithDeadzone
    velocity_deadzone: float = 0.1


# class UniformTrunkModeCommand(CommandTerm):

#     cfg: UniformTrunkModeCommandCfg
#     """The configuration of the command generator."""

#     def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
#         """Initialize the command generator.

#         Args:
#             cfg: The configuration of the command generator.
#             env: The environment.

#         Raises:
#             ValueError: If the heading command is active but the heading range is not provided.
#         """
#         # initialize the base class
#         super().__init__(cfg, env)


#         # obtain the robot asset
#         # -- robot
#         self.robot: Articulation = env.scene[cfg.asset_name]

#         # crete buffers to store the command
#         # -- command: x vel, y vel, yaw vel, heading
#         self.trunk_mode_command = torch.zeros(self.num_envs, 1, device=self.device)


#     def __str__(self) -> str:
#         """Return a string representation of the command generator."""
#         msg = "UniformVelocityCommand:\n"
#         msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
#         msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
#         return msg

#     """
#     Properties
#     """

#     @property
#     def command(self) -> torch.Tensor:
#         """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
#         return self.trunk_mode_command

#     """
#     Implementation specific functions.
#     """


#     def _resample_command(self, env_ids: Sequence[int]):
#         # sample velocity commands
#         r = torch.empty(len(env_ids), device=self.device)
#         # -- linear velocity - x direction
#         self.trunk_mode_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)



class UniformTrunkModeCommand(CommandTerm):
    cfg: UniformTrunkModeCommandCfg

    def __init__(self, cfg: UniformTrunkModeCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 1), device=self.device)
        self._metrics = {}
        print("TRUNK CFG RESAMPLING:", self.cfg.resampling_time_range)
        
    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_command(self):
        # fallback used by Isaac Lab versions that expect _update_command
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)

    def _resample_command(self, env_ids: torch.Tensor):
        # fallback used by Isaac Lab versions that expect _resample_command
        low, high = self.cfg.ranges
        n = env_ids.numel()
        self._command[env_ids, 0] = torch.rand(n, device=self.device) * (high - low) + low

    def _update_metrics(self):
        self._metrics["trunk_mode_mean"] = self._command.mean()


@configclass
class UniformTrunkModeCommandCfg(mdp.NullCommandCfg):
    class_type: type = UniformTrunkModeCommand
    ranges: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        self.resampling_time_range = (10.0, 10.0)