# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from CleanRL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with CleanRL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--camera_eye_b",
    type=float,
    nargs=3,
    default=(0.0, -2.6, 1.1),
    metavar=("X", "Y", "Z"),
    help="Camera eye offset in the robot yaw frame (body-like frame).",
)
parser.add_argument(
    "--camera_lookat_b",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.35),
    metavar=("X", "Y", "Z"),
    help="Camera look-at offset in the robot yaw frame.",
)
# append CleanRL cli arguments
cli_args.add_clean_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os

from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


from cat_envs.tasks.utils.cleanrl.ppo import Agent

# Import extensions to set up environment tasks
import cat_envs.tasks  # noqa: F401

import torch
import numpy as np
import math
import csv


class SideFollowCamera:
    """Camera that tracks the robot root and keeps a side view in the robot yaw frame."""

    def __init__(self, raw_env, env_index: int = 0, eye_b=(0.0, -2.6, 1.1), lookat_b=(0.0, 0.0, 0.35)):
        self.raw_env = raw_env
        self.env_index = int(env_index)
        self.eye_b = np.asarray(eye_b, dtype=np.float64)
        self.lookat_b = np.asarray(lookat_b, dtype=np.float64)
        self.asset = raw_env.scene["robot"]

    @staticmethod
    def _quat_wxyz_to_yaw(quat_wxyz: torch.Tensor) -> float:
        w, x, y, z = [float(v) for v in quat_wxyz]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _yaw_rot(yaw: float) -> np.ndarray:
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def update(self):
        root_pos_w = self.asset.data.root_pos_w[self.env_index].detach().cpu().numpy().astype(np.float64)
        root_quat_w = self.asset.data.root_quat_w[self.env_index].detach().cpu()
        yaw = self._quat_wxyz_to_yaw(root_quat_w)
        rot = self._yaw_rot(yaw)

        eye_w = root_pos_w + rot @ self.eye_b
        lookat_w = root_pos_w + rot @ self.lookat_b
        self.raw_env.sim.set_camera_view(eye=eye_w, target=lookat_w)



def main():
    """Play with CleanRL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_clean_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "clean_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    print(f"[INFO] Loading model: {resume_path}")
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos_play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    actor_sd = torch.load(resume_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = Agent(env).to(device)
    actor.load_state_dict(actor_sd)
    actor.eval()

    obs = env.reset()[0]["policy"]

    # export model to onnx and .pt
    exported_path = os.path.join(log_dir, "exported")
    Path(exported_path).mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, obs.shape[-1]).to(device)
    onnx_path = os.path.join(exported_path, "model.onnx")
    torch.onnx.export(
        actor,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        verbose=True
    )
    print(f"[INFO] Exported ONNX model to {onnx_path}")
    
    pt_path = os.path.join(exported_path, "model.pt")
    torch.jit.trace(actor, dummy_input).save(pt_path)
    print(f"[INFO] Exported .pt model to {pt_path}")

    camera = SideFollowCamera(
        env.unwrapped,
        env_index=0,
        eye_b=tuple(map(float, args_cli.camera_eye_b)),
        lookat_b=tuple(map(float, args_cli.camera_lookat_b)),
    )
    camera.update()

    for step in range(args_cli.video_length):
        with torch.no_grad():
            actions, _, _, _ = actor.get_action_and_value(
                actor.obs_rms(obs, update=False)
            )

        next_obs, rewards, next_done, timeouts, info = env.step(actions)
        obs = next_obs["policy"]


        camera.update()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
