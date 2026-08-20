# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from cat_envs.tasks.utils.cat.manager_constraint_cfg import (
    ConstraintTermCfg as ConstraintTerm,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import cat_envs.tasks.utils.cat.constraints as constraints
import cat_envs.tasks.utils.cat.curriculums as curriculums

import cat_envs.tasks.utils.mdp.terminations as terminations
import cat_envs.tasks.utils.mdp.events as events
import cat_envs.tasks.utils.mdp.commands as commands
import cat_envs.tasks.utils.mdp.rewards as rewards

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG, UNITREE_GO2_ARMATURE_CFG  # isort: skip


JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = UNITREE_GO2_ARMATURE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base", offset=ImuCfg.OffsetCfg(pos=(-0.02557, 0, 0.04232))
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands.UniformVelocityCommandWithDeadzoneCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        velocity_deadzone=0.1,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES, 
        scale=0.3,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        base_acceleration = ObsTerm(
            func=mdp.imu_lin_acc, 
            params={
                "asset_cfg": SceneEntityCfg("imu")
            },
            noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 3.0),
            "dynamic_friction_range": (0.3, 3.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 100,
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )    
    randomize_rigid_body_collider_offsets = EventTerm(
        func=mdp.randomize_rigid_body_collider_offsets,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "contact_offset_distribution_params": (0.0, 0.03)
        }
    )
    # randomize_joint_parameters = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
    #         "friction_distribution_params": (0.05, 0.05),
    #         "armature_distribution_params": (0.0001*40.0689, 0.0007*40.0689),
    #         "operation": "abs",
    #         "distribution": "uniform",            
    #     }
    # )

    # randomize_joint_position_offset = EventTerm(
    #     func=events.randomize_joint_position_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
    #         "position_range": (-0.5, 0.5),
    #     }
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                # "z": (-0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # reset_trunk_joint = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["trunk"], preserve_order=True),
    #         "position_range": (-0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )


    # reset_configuration_from_dataset = EventTerm(
    #     func=events.reset_configuration_from_dataset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
    #         "position_range": (-0.1, 0.1),
    #         "velocity_range": (-1.0, 1.0),
    #         "pose_range": {
    #             # "roll": (-0.5, 0.5),
    #             # "pitch": (0.5, 0.5),
    #             "yaw": (-0.5, 0.5),
    #         },
    #         "root_velocity_range": {
    #             "x": (-0.5, 2.0),
    #             "y": (-0.5, 0.5),
    #             "z": (-0.5, 0.5),
    #             "roll": (-0.5, 0.5),
    #             "pitch": (-0.5, 0.5),
    #             "yaw": (-0.5, 0.5),
    #         },
    #         "sample_from_dataset": True,
    #     }
    # )


    # interval

    anchoring_force = EventTerm(
        func=events.anchoring_force,
        mode="interval",
        interval_range_s=(0.005, 0.005),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),  
            "front_contact_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot.*", "FR_foot.*"]),
            "hind_contact_cfg": SceneEntityCfg("contact_forces", body_names=[ "RL_foot.*", "RR_foot.*"]),
            "force_mag": 20.0,
            "contact_threshold": 1.0,
            "touchdown_margin": 0.005
        },
    )

    lifting_force_hind = EventTerm(
        func=events.lifting_force_hind,
        mode="interval",
        interval_range_s=(0.005, 0.005),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),  
            "front_contact_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot.*", "FR_foot.*"]),
            "hind_contact_cfg": SceneEntityCfg("contact_forces", body_names=[ "RL_foot.*", "RR_foot.*"]),
            "force_mag": 0.0,
            "contact_threshold": 1.0,
            "touchdown_margin": 0.005
        },
    )    
    lifting_force_front = EventTerm(
        func=events.lifting_force_front,
        mode="interval",
        interval_range_s=(0.005, 0.005),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),  
            "front_contact_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot.*", "FR_foot.*"]),
            "hind_contact_cfg": SceneEntityCfg("contact_forces", body_names=[ "RL_foot.*", "RR_foot.*"]),
            "force_mag": 0.0,
            "contact_threshold": 1.0,
            "touchdown_margin": 0.005
        },
    )   



    # set pushing every step, as only some of the environments are chosen
    # as in the isaacgym cat version
    # push_robot = EventTerm(
    #     # Standard push_by_setting_velocity also works, but interestingly results
    #     # in a different gait
    #     func=events.push_by_setting_velocity_with_random_envs,
    #     mode="interval",
    #     is_global_time=True,
    #     interval_range_s=(1.0, 3.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5), "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.78, 0.78),}},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # track_lin_vel_xy_yaw_frame_exp = RewTerm(
    #     func=rewards.track_lin_vel_xy_yaw_frame_exp,
    #     weight=1,
    #     params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    # )        
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_torque_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.5e-5)
    # joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    touchdown_forces = RewTerm(
        func=rewards.foot_touchdown_normal_force, 
        weight=-0.01, 
        params={"threshold": 50.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"])}
    )
    # contact_forces = RewTerm(
    #     func=mdp.contact_forces, 
    #     weight=-0.001, 
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"])}
    # )    

    # track_ang_vel_z_world_exp = RewTerm(
    #     func=rewards.track_ang_vel_z_world_exp,
    #     weight=0.5,
    #     params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    # )
    # root_lin_vel_z = RewTerm(
    #     func=rewards.root_lin_vel_z,
    #     weight=0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])}
    # )
    # joint_position_penalty = RewTerm(
    #     func = rewards.joint_position_penalty,
    #     weight = -0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES), "stand_still_scale": 0.1, "velocity_threshold": 0.1}

    # )
    variable_posture = RewTerm(
        func=rewards.variable_posture,
        weight = 0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES), "std_standing": 0.1, "std_walking": 0.4, "std_running": 0.5, "walking_threshold": 0.3, "running_threshold": 1.5}
    )


    # is_terminated_term = RewTerm(
    #     func=mdp.is_terminated,
    #     weight=-100.0,
    # )
    # leg_extension = RewTerm(
    #     func=rewards.leg_extension,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot.*")
    #     }     
    # )
    # leg_retraction = RewTerm(
    #     func=rewards.leg_retraction,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot.*")
    #     }     
    # )
    # leg_work = RewTerm(
    #     func=rewards.leg_work,
    #     weight=0.001,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot.*")
    #     }
    # )
    # power_loss = RewTerm(
    #     func=rewards.power_loss,
    #     weight=0.05,
    #     params={
    #         "K": 0.05640625,
    #         "Coulomb": 0.05,
    #         "viscous": 0.0,
    #     }
    # )
    # stance_time = RewTerm(
    #     func=rewards.feet_stance_time,
    #     weight=2,
    #     params={
    #         "threshold": 0.1, 
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot.*"])},
    # )
    # energy_new = RewTerm(
    #     func=rewards.energy_new,
    #     weight=-0.01,
    #     params={"std_lin": 1000.0, "std_ang": 0.0,}
    # )


@configclass
class ConstraintsCfg:
    # Safety Soft constraints
    joint_torque = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={"limit": 35.0, 
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    joint_velocity = ConstraintTerm(
        func=constraints.joint_velocity,
        max_p=0.25,
        params={"limit": 51.0, 
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    # joint_acceleration = ConstraintTerm(
    #     func=constraints.joint_acceleration,
    #     max_p=0.25,
    #     params={"limit": 2000.0, 
    #             "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    # )
    # action_rate = ConstraintTerm(
    #     func=constraints.action_rate,
    #     max_p=0.25,
    #     params={"limit": 10.0, 
    #             "asset_cfg": SceneEntityCfg("robot")}, #joint_names=JOINT_NAMES
    # )

    # Safety Hard constraints
    # Knee and base
    contact = ConstraintTerm(
        func=constraints.contact,
        max_p=0.25,
        params={
                "asset_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*thigh", ".*calf"])},
    )
    track_lin_vel_xy_yaw_frame = ConstraintTerm(
        func=constraints.track_lin_vel_xy_yaw_frame,
        max_p=0.25,
        params={"command_name": "base_velocity", 
                "limit": 2.0,
                "asset_cfg": SceneEntityCfg("robot")}
    )
    # foot_touchdown_normal_force = ConstraintTerm(
    #     func=constraints.foot_touchdown_normal_force,
    #     max_p=1.0,
    #     params={"limit": 300.0, 
    #             "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot.*")},
    # )
    # front_hfe_position = ConstraintTerm(
    #     func=constraints.joint_position,
    #     max_p=1.0,
    #     params={"limit": 1.3, 
    #             "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_HFE", "FR_HFE"])},
    # )
    # upsidedown = ConstraintTerm(
    #     func=constraints.upsidedown, 
    #     max_p=1.0,
    #     params={"limit": 0.0,
    #             "asset_cfg": SceneEntityCfg("robot")}
    # )

    # Style constraints
    HAA_position = ConstraintTerm(
        func=constraints.joint_range,
        max_p=0.25,
        params={
            "limit": 0.5, 
            "velocity_deadzone": 3.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    HFE_position = ConstraintTerm(
        func=constraints.joint_range,
        max_p=0.25,
        params={
            "limit": 1.5, 
            "velocity_deadzone": 3.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint"])},
    )
    KFE_position = ConstraintTerm(
        func=constraints.joint_range,
        max_p=0.25,
        params={
            "limit": 0.9, 
            "velocity_deadzone": 3.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"])},
    )        
    # base_orientation = ConstraintTerm(
    #     func=constraints.base_orientation, 
    #     max_p=0.25, 
    #     params={
    #         "limit": 0.5,
    #         "asset_cfg": SceneEntityCfg("robot")}
    # )
    # air_time = ConstraintTerm(
    #     func=constraints.air_time,
    #     max_p=0.25,
    #     params={
    #         "limit": 0.1, 
    #         "velocity_deadzone": 0.1,
    #         "asset_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"])},
    # )
    # no_move = ConstraintTerm(
    #     func=constraints.no_move,
    #     max_p=0.25,
    #     params={
    #         "velocity_deadzone": 0.1,
    #         "joint_vel_limit": 4.0,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)
    #     },
    # )
    # posture_deviation = ConstraintTerm(
    #     func=constraints.joint_range,
    #     max_p=0.25,
    #     params={
    #         "limit": 1.0,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)
    #     },
    # )
    # two_foot_contact = ConstraintTerm(
    #     func=constraints.n_foot_contact,
    #     max_p=0.25,
    #     params={
    #         "number_of_desired_feet": 2,
    #         "min_command_value": 0.5,
    #         "asset_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"])
    #     },
    # )
    # Other constraints:
    # base_height = ConstraintTerm(
    #     func=constraints.min_base_height,
    #     max_p=0.25,
    #     params={
    #         "limit": 0.25,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
    #     },
    # )
    # foot_height = ConstraintTerm(
    #     func=constraints.foot_height,
    #     max_p=0.25,
    #     params={
    #         # "limit": 0.2,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"]),
    #     },
    # )        

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=["base", "Head_upper", "Head_lower",]
    #         ),
    #         "threshold": 1.0,
    #     },
    # )
    upside_down = DoneTerm(
        func=terminations.upside_down,
        params={
            "limit": 0.8,
        },
    )


MAX_CURRICULUM_ITERATIONS = 1000


@configclass
class CurriculumCfg:
    # Safety Soft constraints
    track_lin_vel_xy_yaw_frame = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "track_lin_vel_xy_yaw_frame",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },   
    )
    joint_torque = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_torque",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_velocity = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_velocity",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    # joint_acceleration = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "joint_acceleration",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )
    # action_rate = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "action_rate",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )

    # Style constraints
    contact = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "contact",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )    
    HAA_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "HAA_position",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    HFE_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "HFE_position",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    KFE_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "KFE_position",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )        
    # base_orientation = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "base_orientation",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )
    # keyframe_distance = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "keyframe_distance",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )    
    # base_orientation_1 = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "base_orientation_1",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )
    # base_orientation_2 = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "base_orientation_2",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )
    # air_time = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "air_time",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )
    # two_foot_contact = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "two_foot_contact",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )
    # no_move = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "no_move",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.1,
    #     }
    # )
    # posture_deviation = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "posture_deviation",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.1,
    #     }
    # )    
    # base_height = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "base_height",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.1,
    #     }
    # )
    # swing_foot_height = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "swing_foot_height",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.1,
    #     }
    # )   

    # Curriculum for other than constraint probability

    range_override_x_1 = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
        "address": "commands.base_velocity.ranges.lin_vel_x",
        "modify_fn": curriculums.override_command_range,
        "modify_params": {
            "value": (-1.0, 2.5),
            "num_steps": 12 * MAX_CURRICULUM_ITERATIONS,
        }
    }
    )

    range_override_y = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
        "address": "commands.base_velocity.ranges.lin_vel_y",
        "modify_fn": curriculums.override_command_range,
        "modify_params": {
            "value": (-1.0, 1.0),
            "num_steps": 12 * MAX_CURRICULUM_ITERATIONS,
        }
    }
    )   

    range_override_x_2 = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
        "address": "commands.base_velocity.ranges.lin_vel_x",
        "modify_fn": curriculums.override_command_range,
        "modify_params": {
            "value": (-1.0, 3.0),
            "num_steps": 18 * MAX_CURRICULUM_ITERATIONS,
        }
    }
    )

    # range_override_ang = CurrTerm(
    # func=mdp.modify_term_cfg,
    # params={
    #     "address": "commands.base_velocity.ranges.ang_vel_z",
    #     "modify_fn": curriculums.override_command_range,
    #     "modify_params": {
    #         "value": (-0.1, 0.1),
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #     }
    # }
    # )       


    # range_override_x_3 = CurrTerm(
    # func=mdp.modify_term_cfg,
    # params={
    #     "address": "commands.base_velocity.ranges.lin_vel_x",
    #     "modify_fn": curriculums.override_command_range,
    #     "modify_params": {
    #         "value": (4.5, 5.0),
    #         "num_steps": 120 * MAX_CURRICULUM_ITERATIONS,
    #     }
    # }
    # )

    # reset_config = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "event_manager.cfg.reset_configuration_from_dataset.params.sample_from_dataset",
    #         "modify_fn": curriculums.override_joints_reset,
    #         "modify_params": {
    #             "value": True,
    #             "num_steps": 0 * MAX_CURRICULUM_ITERATIONS,
    #         }
    #     }
    # )

    # reset_base_vx = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "event_manager.cfg.reset_configuration_from_dataset.params.root_velocity_range.x",
    #         "modify_fn": curriculums.override_joints_reset,
    #         "modify_params": {
    #             "value": (-0.0, 1.0),
    #             "num_steps": 0 * MAX_CURRICULUM_ITERATIONS,
    #         }
    #     }
    # )

    # reset_base_wpitch = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "event_manager.cfg.reset_base.params.velocity_range.pitch",
    #         "modify_fn": curriculums.override_joints_reset,
    #         "modify_params": {
    #             "value": (-1.0, 1.0),
    #             "num_steps": 0 * MAX_CURRICULUM_ITERATIONS,
    #         }
    #     }
    # )

    # reset_base_wyaw = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "event_manager.cfg.reset_base.params.velocity_range.yaw",
    #         "modify_fn": curriculums.override_joints_reset,
    #         "modify_params": {
    #             "value": (0.0, 0.0),
    #             "num_steps": 0 * MAX_CURRICULUM_ITERATIONS,
    #         }
    #     }
    # )

    # reset_base_vx_1 = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "event_manager.cfg.reset_configuration_from_dataset.params.root_velocity_range.x",
    #         "modify_fn": curriculums.override_joints_reset,
    #         "modify_params": {
    #             "value": (-0.0, 2.0),
    #             "num_steps": 10 * MAX_CURRICULUM_ITERATIONS,
    #         }
    #     }
    # )


    # reset_base_vx_2 = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "event_manager.cfg.reset_configuration_from_dataset.params.root_velocity_range.x",
    #         "modify_fn": curriculums.override_joints_reset,
    #         "modify_params": {
    #             "value": (0.0, 3.0),
    #             "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         }
    #     }
    # )    

    # reset_base_vx_3 = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "event_manager.cfg.reset_base.params.velocity_range.x",
    #         "modify_fn": curriculums.override_joints_reset,
    #         "modify_params": {
    #             "value": (4.5, 5.0),
    #             "num_steps": 120 * MAX_CURRICULUM_ITERATIONS,
    #         }
    #     }
    # )        


    anchoring_force = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.anchoring_force.params.force_mag",
            "modify_fn": curriculums.override_joints_reset,
            "modify_params": {
                "value": 10,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            }
        }
    )
    anchoring_force_1 = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.anchoring_force.params.force_mag",
            "modify_fn": curriculums.override_joints_reset,
            "modify_params": {
                "value":5,
                "num_steps": 30 * MAX_CURRICULUM_ITERATIONS,
            }
        }
    )
    anchoring_force_2 = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.anchoring_force.params.force_mag",
            "modify_fn": curriculums.override_joints_reset,
            "modify_params": {
                "value":0,
                "num_steps": 36 * MAX_CURRICULUM_ITERATIONS,
            }
        }
    )   

    lifting_force_hind = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.lifting_force_hind.params.force_mag",
            "modify_fn": curriculums.override_joints_reset,
            "modify_params": {
                "value": 20,
                "num_steps": 3 * MAX_CURRICULUM_ITERATIONS,
            }
        }
    )
    lifting_force_hind_1 = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.lifting_force_hind.params.force_mag",
            "modify_fn": curriculums.override_joints_reset,
            "modify_params": {
                "value":10,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            }
        }
    )
    lifting_force_hind_2 = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.lifting_force_hind.params.force_mag",
            "modify_fn": curriculums.override_joints_reset,
            "modify_params": {
                "value":5,
                "num_steps": 30 * MAX_CURRICULUM_ITERATIONS,
            }
        }
    )   
    lifting_force_hind_3 = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.lifting_force_hind.params.force_mag",
            "modify_fn": curriculums.override_joints_reset,
            "modify_params": {
                "value":0,
                "num_steps": 36 * MAX_CURRICULUM_ITERATIONS,
            }
        }
    )       
    lifting_force_front = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.lifting_force_front.params.force_mag",
            "modify_fn": curriculums.override_joints_reset,
            "modify_params": {
                "value": 20,
                "num_steps": 3 * MAX_CURRICULUM_ITERATIONS,
            }
        }
    )
    lifting_force_front_1 = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.lifting_force_front.params.force_mag",
            "modify_fn": curriculums.override_joints_reset,
            "modify_params": {
                "value":10,
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            }
        }
    )
    lifting_force_front_2 = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.lifting_force_front.params.force_mag",
            "modify_fn": curriculums.override_joints_reset,
            "modify_params": {
                "value":5,
                "num_steps": 30 * MAX_CURRICULUM_ITERATIONS,
            }
        }
    )   
    lifting_force_front_3 = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.lifting_force_front.params.force_mag",
            "modify_fn": curriculums.override_joints_reset,
            "modify_params": {
                "value":0,
                "num_steps": 36 * MAX_CURRICULUM_ITERATIONS,
            }
        }
    )   


    # anchoring_force_3 = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "event_manager.cfg.anchoring_force.params.force_mag",
    #         "modify_fn": curriculums.override_joints_reset,
    #         "modify_params": {
    #             "value":0,
    #             "num_steps": 15 * MAX_CURRICULUM_ITERATIONS,
    #         }
    #     }
    # )         



##
# Environment configuration
##


@configclass
class Go2FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.solver_type = 0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.max_position_iteration_count = 4
        # self.sim.max_velocity_iteration_count = 1
        # self.sim.bounce_threshold_velocity = 0.2
        # self.sim.gpu_max_rigid_contact_count = 33554432
        # self.sim.disable_contact_processing = True
        # self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class Go2FlatEnvCfg_PLAY(Go2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # set velocity command
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        # self.events.anchoring_force.params["force_mag"] = 0.0
        self.events.lifting_force_front.params["force_mag"] = 20.0
        self.events.lifting_force_hind.params["force_mag"] = 20.0

        # self.episode_length_s = 1.0


