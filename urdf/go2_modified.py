# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`UNITREE_A1_CFG`: Unitree A1 robot with DC motor model for the legs
* :obj:`UNITREE_GO1_CFG`: Unitree Go1 robot with actuator net model for the legs
* :obj:`UNITREE_GO2_CFG`: Unitree Go2 robot with DC motor model for the legs
* :obj:`H1_CFG`: H1 humanoid robot
* :obj:`H1_MINIMAL_CFG`: H1 humanoid robot with minimal collision bodies
* :obj:`G1_CFG`: G1 humanoid robot
* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies
* :obj:`G1_29DOF_CFG`: G1 humanoid robot configured for locomanipulation tasks
* :obj:`G1_INSPIRE_FTP_CFG`: G1 29DOF humanoid robot with Inspire 5-finger hand

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Configuration - Actuators.
##

GO1_ACTUATOR_CFG = ActuatorNetMLPCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/Unitree/unitree_go1.pt",
    pos_scale=-1.0,
    vel_scale=1.0,
    torque_scale=1.0,
    input_order="pos_vel",
    input_idx=[0, 1, 2],
    effort_limit=23.7,  # taken from spec sheet
    velocity_limit=30.0,  # taken from spec sheet
    saturation_effort=23.7,  # same as effort limit
)
"""Configuration of Go1 actuators using MLP model.

Actuator specifications: https://shop.unitree.com/products/go1-motor

This model is taken from: https://github.com/Improbable-AI/walk-these-ways
"""


##
# Configuration
##


UNITREE_GO2_ARMATURE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        # usd_path=f"/home/gpark/Documents/first_IsaacLab_project/IsaacLab-Go2-Gaits/usd/go2_description_adjusted.usd",
        # usd_path=f"/home/gpark/Documents/first_IsaacLab_project/constraints-as-terminations/urdf/go2_description/go2_description.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            # ".*L_hip_joint": 0.1,
            # ".*R_hip_joint": -0.1,
            # "F[L,R]_thigh_joint": 0.8,
            # "R[L,R]_thigh_joint": 1.0,
            # ".*_calf_joint": -1.5,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,            
            ".*calf_joint": -1.5,
            # ".*R_hip_joint": 0.2,
            # ".*L_hip_joint": -0.2,
            "FR_hip_joint": 0.1,
            "FL_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,            
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            # joint_names_expr=[
            #     "FR_hip_joint",
            #     "FR_thigh_joint",
            #     "FR_calf_joint",
            #     "FL_hip_joint",
            #     "FL_thigh_joint",
            #     "FL_calf_joint",
            #     "RR_hip_joint",
            #     "RR_thigh_joint",
            #     "RR_calf_joint",
            #     "RL_hip_joint",
            #     "RL_thigh_joint",
            #     "RL_calf_joint",
            # ],
            effort_limit_sim=50, #23.5,
            # saturation_effort=50, #23.5,
            velocity_limit_sim=30.0,
            stiffness=20.0,
            damping= 0.5,
            friction=0.05,   #0.0,
            dynamic_friction=0.05,
            armature=0.0007*40.0689
        ),
    },
)
"""Configuration of Unitree Go2 using DC-Motor actuator model."""



UNITREE_GO2_MODIFIED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        # usd_path=f"/home/gpark/Documents/first_IsaacLab_project/constraints-as-terminations/urdf/go2-modified/go2-modified.usda",
        usd_path=f"/home/gpark/Documents/IsaacLab_v3/IsaacLab/go2_modified_8/go2_modified.usda",
        #  usd_path=f"/home/gpark/Documents/first_IsaacLab_project/IsaacLab/go2-modified/go2-modified.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            # ".*L_hip_joint": 0.1,
            # ".*R_hip_joint": -0.1,
            # "F[L,R]_thigh_joint": 0.8,
            # "R[L,R]_thigh_joint": 1.0,
            # ".*_calf_joint": -1.5,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,            
            ".*calf_joint": -1.5,
            # ".*R_hip_joint": 0.2,
            # ".*L_hip_joint": -0.2,
            "FR_hip_joint": 0.1,
            "FL_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,      
            "trunk_joint": -0.6,       
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_joint", 
                ".*_thigh_joint", 
                ".*_calf_joint", 
                "trunk_joint"
            ],
            # joint_names_expr=[
            #     "FR_hip_joint",
            #     "FR_thigh_joint",
            #     "FR_calf_joint",
            #     "FL_hip_joint",
            #     "FL_thigh_joint",
            #     "FL_calf_joint",
            #     "RR_hip_joint",
            #     "RR_thigh_joint",
            #     "RR_calf_joint",
            #     "RL_hip_joint",
            #     "RL_thigh_joint",
            #     "RL_calf_joint",
            # ],
            effort_limit_sim=50, #23.5,
            # saturation_effort=50, #23.5,
            velocity_limit_sim=30.0,
            stiffness=20.0,
            damping= 0.5,
            friction=0.05,   #0.0,
            dynamic_friction=0.05,
            armature=0.0007*40.0689
        ),
        # "trunk": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         "trunk_joint", 
        #     ],
        #     effort_limit_sim=1e+9,
        #     # velocity_limit_sim=1.0,
        #     stiffness = 2.0e+2,
        #     damping = 0.5,
        #     friction=0.05,
        #     dynamic_friction=0.05,
        #     armature=0.0007*40.0689
        # )

    },
)
"""Configuration of Unitree Go2 using DC-Motor actuator model."""
