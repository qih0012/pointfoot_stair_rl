import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from isaacgym.torch_utils import quat_rotate_inverse, quat_apply, torch_rand_float
from .pointfoot_stairs_config import BipedCfgStairs

class BipedStairs(BaseTask):
    
    def __init__(
        self, cfg: BipedCfgStairs, sim_params, physics_engine, sim_device, headless
    ):
        """爬楼梯环境类
        
        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        self.group_idx = torch.arange(0, self.cfg.env.num_envs)

        # 爬楼梯特定的缓冲区
        self._init_stairs_buffers()
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        
        # 初始化feet相关的缓冲区（在feet_indices定义之后）
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        
        self._prepare_reward_function()
        self.init_done = True

    def _init_stairs_buffers(self):
        """初始化爬楼梯相关的缓冲区"""
        # 记录初始高度和目标高度
        self.initial_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.current_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.height_progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # 楼梯接触状态
        self.stair_contact_buf = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)  # 左右脚
        self.on_stairs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # 前进距离跟踪
        self.initial_pos_x = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.forward_progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # 为奖励函数添加必要的缓冲区 (将在_init_buffers中初始化feet相关缓冲区)

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)

        Returns:
            obs (torch.Tensor): Tensor of shape (num_envs, num_observations_per_env)
            rewards (torch.Tensor): Tensor of shape (num_envs)
            dones (torch.Tensor): Tensor of shape (num_envs)
        """
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf
        )

    def compute_observations(self):
        """Computes observations with stairs-specific information"""
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )

        # 更新爬楼梯相关状态
        self._update_stairs_state()

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch_rand_float(
            self.command_ranges["heading"][0],
            self.command_ranges["heading"][1],
            (self.num_envs, 1),
            device=self.device,
        ).squeeze(1)

        # 基础观察：只包含6个可动关节
        obs = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,           # 3维
                self.base_ang_vel * self.obs_scales.ang_vel,           # 3维
                self.projected_gravity,                                # 3维
                self.commands[:, :3] * self.commands_scale,            # 3维
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 6维
                self.dof_vel * self.obs_scales.dof_vel,                # 6维
                self.actions,                                          # 6维
            ),
            dim=-1,
        )  # 总计30维

        # 添加楼梯特定观察 (5维)
        stairs_obs = self._get_stairs_observations()  # 5维
        obs = torch.cat((obs, stairs_obs), dim=-1)     # 30 + 5 = 35维
        
        # 为了匹配网络维度，暂时移除高度测量
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            obs = torch.cat((obs, heights), dim=-1)

        # 观察历史
        self.obs_buf = obs
        # 不处理观察历史，让基类处理

        # Critic observations (包含特权信息)
        if self.cfg.env.num_critic_observations is not None:
            critic_obs = torch.cat(
                (
                    obs,
                    self.contact_forces[:, self.feet_indices, 2].flatten(start_dim=1)
                    * self.obs_scales.contact_forces,
                    self.root_states[:, 2:3]
                    - self.env_origins[:, 2:3],
                ),
                dim=1,
            )
            self.critic_obs_buf = critic_obs

    def _get_stairs_observations(self):
        """获取楼梯相关观察"""
        stairs_obs = torch.cat([
            self.height_progress.unsqueeze(1) * self.obs_scales.stair_height,  # 高度进展
            self.forward_progress.unsqueeze(1),  # 前进进展
            self.stair_contact_buf,  # 足部接触状态
            self.on_stairs.float().unsqueeze(1),  # 是否在楼梯上
        ], dim=-1)
        return stairs_obs

    def _update_stairs_state(self):
        """更新爬楼梯状态"""
        # 计算当前高度
        self.current_height = self.root_states[:, 2] - self.env_origins[:, 2]
        
        # 计算高度进展 (相对于初始高度)
        self.height_progress = self.current_height - self.initial_height
        
        # 计算前进进展
        current_pos_x = self.root_states[:, 0] - self.env_origins[:, 0]
        self.forward_progress = current_pos_x - self.initial_pos_x
        
        # 检测足部与楼梯的接触
        self._detect_stair_contact()

    def _detect_stair_contact(self):
        """检测足部与楼梯的接触"""
        # 获取足部接触力
        foot_contacts = self.contact_forces[:, self.feet_indices, 2] > 1.0  # z方向接触力
        
        # 简化的楼梯检测：基于当前位置和高度
        # 在实际实现中可以更精确地检测楼梯接触
        base_height = self.current_height
        on_ground = base_height < 0.2  # 接近地面
        elevated = base_height > 0.1   # 有一定高度
        
        self.on_stairs = elevated & foot_contacts.any(dim=1)
        self.stair_contact_buf = foot_contacts.float()

    def reset_idx(self, env_ids):
        """Reset specific environments"""
        if len(env_ids) == 0:
            return
            
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        self.obs_history[env_ids] = self.obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["max_terrain_level"] = torch.max(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            
        # 重置楼梯相关状态
        self.initial_height[env_ids] = self.root_states[env_ids, 2] - self.env_origins[env_ids, 2]
        self.initial_pos_x[env_ids] = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
        self.height_progress[env_ids] = 0.0
        self.forward_progress[env_ids] = 0.0
        self.stair_contact_buf[env_ids] = 0.0
        self.on_stairs[env_ids] = False
        self.last_contacts[env_ids] = False
        self.feet_air_time[env_ids] = 0.0

    #------------ 标准基础奖励函数 ----------------
    def _reward_lin_vel_z(self):
        # 惩罚z轴基础线性速度 - 在楼梯上时允许适当的垂直速度
        lin_vel_z = self.base_lin_vel[:, 2]
        
        # 在楼梯上时，允许适当的向上速度
        on_stairs_mask = self.on_stairs.float()
        
        # 在楼梯上：惩罚过快的垂直速度，但允许适当的向上运动
        stairs_penalty = torch.where(
            torch.abs(lin_vel_z) > 0.5,  # 垂直速度过快
            torch.square(lin_vel_z),
            torch.zeros_like(lin_vel_z)  # 适当的垂直速度
        )
        
        # 在平地上：惩罚所有垂直运动
        ground_penalty = torch.square(lin_vel_z)
        
        return on_stairs_mask * stairs_penalty + (1 - on_stairs_mask) * ground_penalty
    
    def _reward_ang_vel_xy(self):
        # 惩罚xy轴基础角速度
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # 惩罚非平坦的基础方向
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # 惩罚基础高度偏离目标
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # 惩罚力矩
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # 惩罚关节速度
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # 惩罚关节加速度
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # 惩罚动作变化
        return torch.sum(torch.square(self.last_actions[:, :, 0] - self.actions), dim=1)
    
    def _reward_collision(self):
        # 惩罚碰撞
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # 终止奖励
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # 惩罚关节位置接近极限
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # 惩罚关节速度接近极限
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # 惩罚力矩接近极限
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # 跟踪线性速度命令 (xy轴)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # 跟踪角速度命令 (yaw轴)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # 奖励长步幅
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # 惩罚足部撞击垂直表面
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # 在零命令时惩罚运动
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # 惩罚高接触力
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    #------------ 双足机器人特定奖励函数 ----------------
    def _reward_no_fly(self):
        # 防止双脚同时离地 (参考Cassie)
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact

    #------------ 爬楼梯特定奖励函数 ----------------
    def _reward_height_progress(self):
        # 奖励高度进展
        return torch.clamp(self.height_progress, 0.0, 2.0)  # 限制最大奖励

    def _reward_stair_contact(self):
        # 奖励楼梯接触
        contact_reward = torch.zeros(self.num_envs, device=self.device)
        
        # 双脚都接触时给予更高奖励
        both_feet_contact = self.stair_contact_buf.sum(dim=1) > 1.5
        single_foot_contact = (self.stair_contact_buf.sum(dim=1) > 0.5) & (~both_feet_contact)
        
        contact_reward[both_feet_contact] = 1.0
        contact_reward[single_foot_contact] = 0.5
        
        return contact_reward

    def _reward_forward_progress(self):
        # 奖励前进进展
        return torch.clamp(self.forward_progress, 0.0, 3.0)  # 限制最大奖励

    def _reward_stair_climbing_efficiency(self):
        # 爬楼梯效率：结合高度和前进进展
        height_component = torch.clamp(self.height_progress, 0.0, 1.0)
        forward_component = torch.clamp(self.forward_progress, 0.0, 1.0)
        # 只有在楼梯上时才给奖励
        efficiency = (height_component + forward_component) * 0.5 * self.on_stairs.float()
        return efficiency

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # 基础观察噪声 (30维)
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel      # base lin vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel    # base ang vel
        noise_vec[6:9] = noise_scales.gravity * noise_level                              # projected gravity
        noise_vec[9:12] = 0.0                                                            # commands
        noise_vec[12:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # dof pos (6 joints)
        noise_vec[18:24] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # dof vel (6 joints)
        noise_vec[24:30] = 0.0                                                           # actions (6 joints)
        
        # 楼梯特定观察噪声 (5维)
        noise_vec[30:35] = 0.05 * noise_level                                            # stairs observations
        
        # 高度测量噪声 (如果启用)
        if self.cfg.terrain.measure_heights:
            height_start = 35
            height_end = height_start + len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y)
            noise_vec[height_start:height_end] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        
        return noise_vec


    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _resample_commands(self, env_ids):
        """Randomly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_y"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "ang_vel_yaw"
        ][
            env_ids, 0
        ]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)