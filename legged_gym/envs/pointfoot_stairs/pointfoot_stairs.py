import torch
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

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
            (self.num_envs,),
            device=self.device,
        ).squeeze(1)

        # 基础观察 
        obs = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        # 添加楼梯特定观察
        stairs_obs = self._get_stairs_observations()
        obs = torch.cat((obs, stairs_obs), dim=-1)

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
        if self.obs_history_length > 1:
            self.obs_history = torch.cat(
                [self.obs_buf.unsqueeze(1), self.obs_history[:, :-1]], dim=1
            ).flatten(1)

        # Critic observations (包含特权信息)
        if self.cfg.env.num_critic_observations is not None:
            critic_obs = torch.cat(
                (
                    obs,
                    self.contact_forces[:, self.feet_indices, 2].flatten(dim=1)
                    * self.obs_scales.contact_forces,
                    self.rigid_body_states[:, self.base_idx, 2:3]
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
        super().reset_idx(env_ids)
        
        # 重置楼梯相关状态
        self.initial_height[env_ids] = self.root_states[env_ids, 2] - self.env_origins[env_ids, 2]
        self.initial_pos_x[env_ids] = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
        self.height_progress[env_ids] = 0.0
        self.forward_progress[env_ids] = 0.0
        self.stair_contact_buf[env_ids] = 0.0
        self.on_stairs[env_ids] = False

    # ======== 爬楼梯特定奖励函数 ========
    def _reward_height_progress(self):
        """奖励高度进展"""
        return self.height_progress

    def _reward_stair_contact(self):
        """奖励楼梯接触"""
        # 当足部正确接触楼梯时给予奖励
        contact_reward = torch.zeros(self.num_envs, device=self.device)
        
        # 双脚都接触时给予更高奖励
        both_feet_contact = self.stair_contact_buf.sum(dim=1) > 1.5
        single_foot_contact = (self.stair_contact_buf.sum(dim=1) > 0.5) & (~both_feet_contact)
        
        contact_reward[both_feet_contact] = 1.0
        contact_reward[single_foot_contact] = 0.5
        
        return contact_reward

    def _reward_balance_on_stairs(self):
        """奖励在楼梯上保持平衡"""
        # 基于角速度和姿态稳定性
        ang_vel_penalty = torch.norm(self.base_ang_vel, dim=1)
        orientation_penalty = torch.norm(self.projected_gravity[:, :2], dim=1)
        
        balance_reward = torch.exp(-2 * (ang_vel_penalty + orientation_penalty))
        
        # 只在楼梯上时应用平衡奖励
        balance_reward = balance_reward * self.on_stairs.float()
        
        return balance_reward

    def _reward_forward_progress(self):
        """奖励前进进展"""
        # 鼓励在爬楼梯的同时前进
        return torch.clamp(self.forward_progress, 0.0, 5.0)  # 限制最大奖励

    def _reward_energy_efficiency(self):
        """能效奖励 - 惩罚过度的动作"""
        return -torch.norm(self.torques, dim=1)

    def _reward_lin_vel_z_stairs(self):
        """修改的垂直速度奖励 - 在楼梯上时允许适当的垂直速度"""
        lin_vel_z = self.base_lin_vel[:, 2]
        
        # 在楼梯上时，允许适当的向上速度
        on_stairs_mask = self.on_stairs.float()
        
        # 在楼梯上：惩罚过快的垂直速度，但允许适当的向上运动
        stairs_penalty = torch.where(
            lin_vel_z > 0.5,  # 向上速度过快
            -lin_vel_z,
            torch.where(
                lin_vel_z < -0.5,  # 向下速度过快
                lin_vel_z,
                0.0  # 适当的垂直速度
            )
        )
        
        # 在平地上：惩罚所有垂直运动
        ground_penalty = -torch.abs(lin_vel_z)
        
        return on_stairs_mask * stairs_penalty + (1 - on_stairs_mask) * ground_penalty
