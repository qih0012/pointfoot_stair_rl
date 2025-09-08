# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER OR CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.base_config import BaseConfig
from legged_gym import LEGGED_GYM_ROOT_DIR

import os
robot_type = os.getenv("ROBOT_TYPE")

class BipedCfgStairs(BaseConfig):
    class env:
        num_envs = 8192
        num_observations = 47  # 完全匹配网络期望的维度
        num_critic_observations = 3 + num_observations
        num_height_samples = 117
        num_actions = 6  # 6个可动关节：abad_L, hip_L, knee_L, abad_R, hip_R, knee_R
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 30  # 增加episode长度以完成爬楼梯
        obs_history_length = 10  # number of observations stacked together
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5

    class terrain:
        mesh_type = "trimesh"  # 使用trimesh以支持复杂楼梯地形
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.4
        dynamic_friction = 0.4
        restitution = 0.8
        # rough terrain only:
        measure_heights = True  # 启用高度测量
        critic_measure_heights = True
        measured_points_x = [
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # 从简单楼梯开始
        terrain_length = 8.0  # 保持与 width 一致以避免维度问题
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # 专注于楼梯地形: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 0.0, 0.7, 0.0, 0.3]  # 70%楼梯，30%离散障碍
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )

    class commands:
        curriculum = True
        smooth_max_lin_vel_x = 1.5  # 降低速度以适应楼梯
        smooth_max_lin_vel_y = 0.5
        non_smooth_max_lin_vel_x = 1.0
        non_smooth_max_lin_vel_y = 0.5
        max_ang_vel_yaw = 2.0  # 降低转向速度
        curriculum_threshold = 0.75
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)

        resampling_time = 10.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [0.0, 1.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-2.0, 2.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class gait:
        num_gait_params = 4
        resampling_time = 5  # time before command are changed[s]

        class ranges:
            frequencies = [1.5, 2.5]
            offsets = [0, 1]  # offset is hard to learn
            # durations = [0.3, 0.8]  # small durations(<0.4) is hard to learn
            # frequencies = [2, 2]
            # offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            swing_height = [0.0, 0.1]

    class init_state:
        pos = [0.0, 0.0, 0.8]  # x,y,z [m] 参考原始配置的高度
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0 - 直立姿态
            "abad_L_Joint": 0.0,      # 左侧髋关节外展
            "hip_L_Joint": 0.0,       # 左侧髋关节
            "knee_L_Joint": 0.0,      # 左侧膝关节
            "foot_L_Joint": 0.0,      # 左侧足部关节 (fixed)
            "abad_R_Joint": 0.0,      # 右侧髋关节外展
            "hip_R_Joint": 0.0,       # 右侧髋关节
            "knee_R_Joint": 0.0,      # 右侧膝关节
            "foot_R_Joint": 0.0,      # 右侧足部关节 (fixed)
        }

    class control:
        action_scale = 0.25
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters - 参考原始配置
        stiffness = {
            "abad_L_Joint": 42,       # 左侧髋关节外展刚度
            "hip_L_Joint": 42,        # 左侧髋关节刚度
            "knee_L_Joint": 42,       # 左侧膝关节刚度
            "foot_L_Joint": 0.0,      # 左侧足部关节刚度 (fixed joint)
            "abad_R_Joint": 42,       # 右侧髋关节外展刚度
            "hip_R_Joint": 42,        # 右侧髋关节刚度
            "knee_R_Joint": 42,       # 右侧膝关节刚度
            "foot_R_Joint": 0.0,      # 右侧足部关节刚度 (fixed joint)
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 2.5,      # 左侧髋关节外展阻尼
            "hip_L_Joint": 2.5,       # 左侧髋关节阻尼
            "knee_L_Joint": 2.5,      # 左侧膝关节阻尼
            "foot_L_Joint": 0.0,      # 左侧足部关节阻尼 (fixed joint)
            "abad_R_Joint": 2.5,      # 右侧髋关节外展阻尼
            "hip_R_Joint": 2.5,       # 右侧髋关节阻尼
            "knee_R_Joint": 2.5,      # 右侧膝关节阻尼
            "foot_R_Joint": 0.0,      # 右侧足部关节阻尼 (fixed joint)
        }  # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        user_torque_limit = 33.5  # limits torque range
        max_power = 1000.0  # [W]

    class asset:
        file = "{}/resources/robots/{}/urdf/robot.urdf".format(LEGGED_GYM_ROOT_DIR, robot_type)
        name = "biped"
        foot_name = "foot"  # 会匹配 foot_L_Link, foot_R_Link
        foot_radius = 0.032  # 根据URDF中sphere radius="0.032"
        penalize_contacts_on = ["hip_L_Link", "hip_R_Link", "knee_L_Link", "knee_R_Link"]  # 根据URDF链接名称
        terminate_after_contacts_on = ["base_Link"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fix the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.02

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        randomize_base_com = True
        rand_com_vec = [0.03, 0.02, 0.03]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        rand_force = False
        force_resampling_time_s = 15
        max_force = 50.0
        rand_force_curriculum_level = 0
        randomize_Kp = True
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        randomize_imu_offset = True
        randomize_imu_offset_range = [-1.2, 1.2]
        delay_ms_range = [0, 20]

    class rewards:
        class scales:
            # 基础奖励
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.0
            base_height = -0.0
            joint_acc = -0.0001
            action_rate = -0.01
            power = -0.0001
            collision = -1.0
            # 爬楼梯特定奖励
            height_progress = 2.0  # 高度进展奖励
            stair_contact = 0.5    # 楼梯接触奖励
            balance_on_stairs = 1.0  # 楼梯平衡奖励
            forward_progress = 1.5  # 前进进展奖励
            energy_efficiency = -0.001  # 能效奖励

        only_positive_rewards = (
            True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        )
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        ang_tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = 1.0  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 1.0
        feet_height_target = 0.10
        min_feet_distance = 0.115
        about_landing_threshold = 0.08
        max_contact_force = 100.0  # forces above this value are penalized
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            contact_forces = 0.01
            torque = 0.05
            # 楼梯相关观察缩放
            stair_height = 2.0

        clip_observations = 100.0
        clip_actions = 100.0

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            friction_offset_threshold = 0.001  # [m]
            friction_correlation_distance = 0.005  # [m]
            num_gauss_seidel_iterations = 4
            default_buffer_size_multiplier = 5.0
            max_gpu_contact_pairs = 8 * 1024 * 1024  # 8M contact pairs
            num_position_iterations = 8
            num_velocity_iterations = 0
            contact_offset = 0.002  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.2  # [m/s]
            friction_offset_threshold = 0.001  # [m]
            friction_correlation_distance = 0.005  # [m]
            num_gauss_seidel_iterations = 4

    class noise:
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
            imu = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11, 5, 3]  # [m]


class BipedCfgPPOStairs(BaseConfig):
    seed = 1

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        experiment_name = "pointfoot_stairs"
        run_name = ""
        max_iterations = 1500  # number of policy updates
        save_interval = 50  # check for potential saves every this many iterations
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and checkpoint
        use_wandb = False
        wandb_project = "legged_gym"
        wandb_entity = None
        wandb_group = None
        wandb_name = None
        wandb_job_type = "train"
        exptid = ""

    class MLP_Encoder:
        output_detach = True
        num_input_dim = BipedCfgStairs.env.num_observations * BipedCfgStairs.env.obs_history_length
        num_output_dim = 3
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        orthogonal_init = False

    class algorithm:
        # algorithm
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.0e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        
        # Extra training params
        est_learning_rate = 1.0e-3
        ts_learning_rate = 1.0e-4
        critic_take_latent = True

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1