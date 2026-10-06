# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import to_torch, unscale, quat_apply, tensor_clamp, torch_rand_float, quat_conjugate, quat_mul
from glob import glob
from hora.utils.misc import tprint
from .base.vec_task import VecTask
from .tactile import PiezoresistiveArray, VTRefineTactileArray
import torch

class SimpleHandHora(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config
        # before calling init in VecTask, need to do
        # 1. setup randomization
        self._setup_domain_rand_config(config['env']['randomization'])
        # 2. setup privileged information
        self._setup_priv_option_config(config['env']['privInfo'])
        # 3. setup object assets
        self._setup_object_info(config['env']['object'])
        # 4. setup reward
        self._setup_reward_config(config['env']['reward'])
        # 5. setup fingertip piezoresistive arrays
        self._setup_tactile_config(config['env'].get('tactile', {}))
        self.base_obj_scale = config['env']['baseObjScale']
        self.save_init_pose = config['env']['genGrasps']
        self.aggregate_mode = self.config['env']['aggregateMode']
        self.up_axis = 'z'
        self.reset_z_threshold = self.config['env']['reset_height_threshold']
        self.grasp_cache_name = self.config['env']['grasp_cache_name']
        self.evaluate = self.config['on_evaluation']
        self.priv_info_dict = {
            'obj_position': (0, 3),
            'obj_scale': (3, 4),
            'obj_mass': (4, 5),
            'obj_friction': (5, 6),
            'obj_com': (6, 9),
        }

        super().__init__(config, sim_device, graphics_device_id, headless)

        self.debug_viz = self.config['env']['enableDebugVis']
        self.max_episode_length = self.config['env']['episodeLength']
        self.dt = self.sim_params.dt

        if self.viewer:
            cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.SimpleHand_default_dof_pos = torch.zeros(self.num_SimpleHand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.SimpleHand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_SimpleHand_dofs]
        self.SimpleHand_dof_pos = self.SimpleHand_dof_state[..., 0]
        self.SimpleHand_dof_vel = self.SimpleHand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self._refresh_gym()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # object apply random forces parameters
        self.force_scale = self.config['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.config['env'].get('randomForceProbScalar', 0.0)
        self.force_decay = self.config['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.config['env'].get('forceDecayInterval', 0.08)
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        if self.randomize_scale and self.scale_list_init:
            self.saved_grasping_states = {}
            for s in self.randomize_scale_list:
                self.saved_grasping_states[str(s)] = torch.from_numpy(np.load(
                    f'cache/{self.grasp_cache_name}_grasp_50k_s{str(s).replace(".", "")}.npy'
                )).float().to(self.device)
        else:
        # 在生成模式下，创建一个空的 saved_grasping_states 或者设置标志
            if self.save_init_pose:
                self.saved_grasping_states = None  # 或者 {}
            else:
                assert self.save_init_pose

        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        # useful buffers
        self.object_rot_prev = self.object_rot.clone()
        self.object_pos_prev = self.object_pos.clone()
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.torques = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [int, float], 'assume p_gain and d_gain are only scalars'
        self.p_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.d_gain

        # debug and understanding statistics
        self.env_timeout_counter = to_torch(np.zeros((len(self.envs)))).long().to(self.device)  # max 10 (10000 envs)
        self.stat_sum_rewards = 0
        self.stat_sum_rotate_rewards = 0
        self.stat_sum_episode_length = 0
        self.stat_sum_obj_linvel = 0
        self.stat_sum_torques = 0
        self.env_evaluated = 0
        self.max_evaluate_envs = 50000

    def _before_prepare_sim(self):
        self.tactile_layout = None
        self.tactile_sensor = None
        self.tactile_backend_active = 'disabled'
        if not self.tactile_enabled:
            return

        layout = PiezoresistiveArray(
            rows=self.tactile_rows,
            columns=self.tactile_columns,
            surface_centers=self.tactile_surface_centers,
            row_axes=self.tactile_row_axes,
            column_axes=self.tactile_column_axes,
            normal_axes=self.tactile_normal_axes,
            surface_size=self.tactile_surface_size,
            contact_sigma=self.tactile_contact_sigma,
            force_threshold=self.tactile_force_threshold,
            max_force=self.tactile_max_force,
            noise_std=self.tactile_noise_std,
            device=self.device,
        )
        self.tactile_layout = layout
        use_vt_refine = (
            self.tactile_sdf_api_available
            and self.tactile_backend_requested in ('auto', 'vt_refine')
        )
        if self.tactile_backend_requested == 'vt_refine' and not use_vt_refine:
            if not self.tactile_allow_contact_fallback:
                raise RuntimeError(
                    'tactile.backend=vt_refine requires the TacSL Isaac Gym binary'
                )
            print(
                'TacSL SDF tensor API is unavailable; using the explicit '
                'contact-force tactile fallback.'
            )

        if use_vt_refine:
            shape_ids = to_torch(
                self.tactile_sdf_proxy_shape_ids,
                dtype=torch.int32,
                device=self.device,
            )
            self.tactile_sensor = VTRefineTactileArray(
                gym=self.gym,
                gymtorch=gymtorch,
                sim=self.sim,
                shape_ids=shape_ids,
                local_taxel_positions=layout.local_taxel_positions,
                normal_axes=layout.normal_axes,
                stiffness_n_per_m=self.tactile_stiffness_n_per_m,
                normal_sign=self.tactile_sdf_normal_sign,
                force_threshold=self.tactile_force_threshold,
                max_force=self.tactile_max_force,
                noise_std=self.tactile_noise_std,
                mesh_pos_local=self.tactile_indenter_mesh_pos,
                mesh_orn_local=self.tactile_indenter_mesh_orn,
                sdf_bounds_min=self.tactile_sdf_bounds_min,
                sdf_bounds_max=self.tactile_sdf_bounds_max,
                sdf_query_margin_m=self.tactile_sdf_query_margin_m,
            )
            self.tactile_sensor.acquire_sdf_view_tensor()
            self.tactile_backend_active = self.tactile_sensor.backend_name
        else:
            self.tactile_sensor = layout
            self.tactile_backend_active = 'contact_force'

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.tactile_sdf_api_available = (
            hasattr(self.gym, 'acquire_sdf_view_tensor')
            and hasattr(self.gym, 'refresh_sdf_view_tensor')
        )
        self.tactile_use_sdf_backend = (
            self.tactile_enabled
            and self.tactile_sdf_api_available
            and self.tactile_backend_requested in ('auto', 'vt_refine')
        )
        self._create_object_asset()

        # set SimpleHand dof properties
        self.num_SimpleHand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        SimpleHand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.SimpleHand_dof_lower_limits = []
        self.SimpleHand_dof_upper_limits = []

        for i in range(self.num_SimpleHand_dofs):
            self.SimpleHand_dof_lower_limits.append(SimpleHand_dof_props['lower'][i])
            self.SimpleHand_dof_upper_limits.append(SimpleHand_dof_props['upper'][i])
            SimpleHand_dof_props['effort'][i] = 0.5 # 限制最大力矩
            if self.torque_control:
                SimpleHand_dof_props['stiffness'][i] = 0. # 降低刚度
                SimpleHand_dof_props['damping'][i] = 0. # 降低阻尼
                SimpleHand_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            else:
                SimpleHand_dof_props['stiffness'][i] = self.config['env']['controller']['pgain']
                SimpleHand_dof_props['damping'][i] = self.config['env']['controller']['dgain']
            SimpleHand_dof_props['friction'][i] = 0.01
            SimpleHand_dof_props['armature'][i] = 0.001

        self.SimpleHand_dof_lower_limits = to_torch(self.SimpleHand_dof_lower_limits, device=self.device)
        self.SimpleHand_dof_upper_limits = to_torch(self.SimpleHand_dof_upper_limits, device=self.device)

        hand_pose, obj_pose = self._init_object_pose()

        # compute aggregate size
        self.num_SimpleHand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_SimpleHand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        max_agg_bodies = self.num_SimpleHand_bodies + 2
        # hand + manipulated object + an isolated SDF proxy shape
        max_agg_shapes = self.num_SimpleHand_shapes + 2

        self.envs = []

        self.object_init_state = []
        self.object_scales = []

        self.hand_indices = []
        self.object_indices = []
        self.tactile_sdf_proxy_shape_ids = []

        SimpleHand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        object_rb_count = 1
        self.object_rb_handles = list(range(SimpleHand_rb_count, SimpleHand_rb_count + object_rb_count))

        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, SimpleHand_dof_props)
            if (
                self.tactile_enabled
                and self.tactile_sdf_api_available
                and self.tactile_backend_requested in ('auto', 'vt_refine')
            ):
                hand_shape_props = self.gym.get_actor_rigid_shape_properties(
                    env_ptr, hand_actor
                )
                body_shape_indices = self.gym.get_actor_rigid_body_shape_indices(
                    env_ptr, hand_actor
                )
                for body_id in self.tactile_body_indices_list:
                    shape_range = body_shape_indices[body_id]
                    for shape_id in range(
                        shape_range.start, shape_range.start + shape_range.count
                    ):
                        hand_shape_props[shape_id].compliance = (
                            self.tactile_elastomer_compliance
                        )
                self.gym.set_actor_rigid_shape_properties(
                    env_ptr, hand_actor, hand_shape_props
                )
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # add object
            object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            object_asset = self.object_asset_list[object_type_id]

            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object', i, 0, 0)
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            if self.tactile_use_sdf_backend:
                object_shape_props = self.gym.get_actor_rigid_shape_properties(
                    env_ptr, object_handle
                )
                if len(object_shape_props) != 1:
                    raise ValueError(
                        'VT-Refine currently requires exactly one rigid shape per object'
                    )

            obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                obj_scale = np.random.uniform(self.randomize_scale_list[i % num_scales] - 0.025, self.randomize_scale_list[i % num_scales] + 0.025)
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
            self.object_scales.append(obj_scale)
            self._update_priv_buf(env_id=i, name='obj_scale', value=obj_scale)

            if self.tactile_use_sdf_backend:
                # Query an isolated copy of the cooked mesh. The TacSL kernel
                # can fail when its target shape is in an active contact pair.
                proxy_asset = self.object_sdf_proxy_asset_list[object_type_id]
                proxy_pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, -10.0))
                proxy_handle = self.gym.create_actor(
                    env_ptr, proxy_asset, proxy_pose, 'sdf_proxy', i, 0, 0
                )
                self.gym.set_actor_scale(env_ptr, proxy_handle, obj_scale)
                proxy_props = self.gym.get_actor_rigid_shape_properties(
                    env_ptr, proxy_handle
                )
                if len(proxy_props) != 1:
                    raise ValueError('VT-Refine SDF proxy requires one rigid shape')
                self.tactile_sdf_proxy_shape_ids.append(proxy_props[0].global_index)

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name='obj_com', value=obj_com)

            obj_friction = 1.0
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_friction = rand_friction
            self._update_priv_buf(env_id=i, name='obj_friction', value=obj_friction)

            if self.randomize_mass:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)
            else:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.object_scales = to_torch(
            self.object_scales, dtype=torch.float, device=self.device
        )

    def reset_idx(self, env_ids):
        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0
        
        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[(env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)]
            if len(s_ids) == 0:
                continue
            obj_scale = self.randomize_scale_list[n_s]
            scale_key = str(obj_scale)
            sampled_pose_idx = np.random.randint(self.saved_grasping_states[scale_key].shape[0], size=len(s_ids))
            sampled_pose = self.saved_grasping_states[scale_key][sampled_pose_idx].clone()
            self.root_state_tensor[self.object_indices[s_ids], :7] = sampled_pose[:, 16:]
            self.root_state_tensor[self.object_indices[s_ids], 7:13] = 0
            pos = sampled_pose[:, :16]
            
            self.SimpleHand_dof_pos[s_ids, :] = pos
            self.SimpleHand_dof_vel[s_ids, :] = 0
            self.prev_targets[s_ids, :self.num_SimpleHand_dofs] = pos
            self.cur_targets[s_ids, :self.num_SimpleHand_dofs] = pos
            self.init_pose_buf[s_ids, :] = pos.clone()

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.tactile_forces[env_ids] = 0
        self.tactile_depth[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def _read_tactile_forces(self):
        """Read the current tactile array without changing policy buffers."""
        if not self.tactile_enabled:
            return None

        fingertip_states = self.rigid_body_states[:, self.tactile_body_indices, :]
        fingertip_contact_forces = self.contact_forces[
            :, self.tactile_body_indices, :
        ]
        if self.tactile_backend_active == 'vt_refine_sdf':
            object_state = self.root_state_tensor[self.object_indices]
            tactile_forces = self.tactile_sensor.sense(
                fingertip_states, object_state, self.object_scales
            )
            self.tactile_depth[:] = self.tactile_sensor.last_depth
        else:
            tactile_forces = self.tactile_sensor.sense(
                fingertip_states, self.object_pos, fingertip_contact_forces
            )
            self.tactile_depth.zero_()
        return tactile_forces

    def compute_observations(self):
        self._refresh_gym()
        # deal with normal observation, do sliding window
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = (torch.rand(self.SimpleHand_dof_pos.shape) * 2.0 - 1.0) * self.joint_noise_scale
        cur_obs_buf = unscale(
            joint_noise_matrix.to(self.device) + self.SimpleHand_dof_pos, self.SimpleHand_dof_lower_limits, self.SimpleHand_dof_upper_limits
        ).clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:16] = unscale(
            self.SimpleHand_dof_pos[at_reset_env_ids], self.SimpleHand_dof_lower_limits,
            self.SimpleHand_dof_upper_limits
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 16:32] = self.SimpleHand_dof_pos[at_reset_env_ids].unsqueeze(1)
        t_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()

        self.obs_buf[:, :t_buf.shape[1]] = t_buf
        if self.tactile_enabled:
            tactile_forces = self._read_tactile_forces()
            tactile_forces[at_reset_env_ids] = 0.0
            self.tactile_depth[at_reset_env_ids] = 0.0
            self.tactile_forces[:] = tactile_forces
            tactile_obs = torch.clamp(
                tactile_forces / self.tactile_force_normalization, min=0.0, max=1.0
            )
            self.obs_buf[:, t_buf.shape[1]:] = tactile_obs.flatten(start_dim=1)
        self.at_reset_buf[at_reset_env_ids] = 0

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone())

    def compute_reward(self, actions):
        self.rot_axis_buf[:, -1] = -1
        # pose diff penalty
        pose_diff_penalty = ((self.SimpleHand_dof_pos - self.init_pose_buf) ** 2).sum(-1)
        # work and torque penalty
        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = ((self.torques * self.dof_vel_finite_diff).sum(-1)) ** 2
        # Compute offset in radians. Radians -> radians / sec
        angdiff = quat_to_axis_angle(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / (self.control_freq_inv * self.dt)
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
        rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)
        # linear velocity: use position difference instead of self.object_linvel
        object_linvel = ((self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)).clone()
        object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)

        # 新增：接触稳定性奖励（针对Digit传感器）
        contact_forces_norm = torch.norm(self.contact_forces, p=2, dim=-1)
        has_contact = contact_forces_norm > 0.1  # 接触力阈值
        num_contact_points = has_contact.sum(dim=-1).float()

        # 鼓励保持足够的接触点
        contact_stability_reward = torch.clamp(num_contact_points / 4.0, 0.0, 1.0)

        self.rew_buf[:] = compute_hand_reward(
            object_linvel_penalty, self.object_linvel_penalty_scale,
            rotate_reward, self.rotate_reward_scale,
            pose_diff_penalty, self.pose_diff_penalty_scale,
            torque_penalty, self.torque_penalty_scale,
            work_penalty, self.work_penalty_scale,
            contact_stability_reward, 0.3,  # 新增接触稳定性奖励
        )
        self.reset_buf[:] = self.check_termination(self.object_pos)
        self.extras['rotation_reward'] = rotate_reward.mean()
        self.extras['object_linvel_penalty'] = object_linvel_penalty.mean()
        self.extras['pose_diff_penalty'] = pose_diff_penalty.mean()
        self.extras['work_done'] = work_penalty.mean()
        self.extras['torques'] = torque_penalty.mean()
        self.extras['roll'] = object_angvel[:, 0].mean()
        self.extras['pitch'] = object_angvel[:, 1].mean()
        self.extras['yaw'] = object_angvel[:, 2].mean()

        if self.evaluate:
            finished_episode_mask = self.reset_buf == 1
            self.stat_sum_rewards += self.rew_buf.sum()
            self.stat_sum_rotate_rewards += rotate_reward.sum()
            self.stat_sum_torques += self.torques.abs().sum()
            self.stat_sum_obj_linvel += (self.object_linvel ** 2).sum(-1).sum()
            self.stat_sum_episode_length += (self.reset_buf == 0).sum()
            self.env_evaluated += (self.reset_buf == 1).sum()
            self.env_timeout_counter[finished_episode_mask] += 1
            info = f'progress {self.env_evaluated} / {self.max_evaluate_envs} | ' \
                   f'reward: {self.stat_sum_rewards / self.env_evaluated:.2f} | ' \
                   f'eps length: {self.stat_sum_episode_length / self.env_evaluated:.2f} | ' \
                   f'rotate reward: {self.stat_sum_rotate_rewards / self.env_evaluated:.2f} | ' \
                   f'lin vel (x100): {self.stat_sum_obj_linvel * 100 / self.stat_sum_episode_length:.4f} | ' \
                   f'command torque: {self.stat_sum_torques / self.stat_sum_episode_length:.2f}'
            tprint(info)
            if self.env_evaluated >= self.max_evaluate_envs:
                exit()

    def post_physics_step(self):
        # print("object z:", self.object_pos[0, 2].item())
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        self.compute_reward(self.actions)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            draw_env_count = min(self.num_envs, self.tactile_debug_max_envs)
            for i in range(draw_env_count):
                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

            self._draw_tactile_debug(draw_env_count)

    def _draw_tactile_debug(self, draw_env_count):
        if self.tactile_layout is None or draw_env_count == 0:
            return

        fingertip_states = self.rigid_body_states[
            :draw_env_count, self.tactile_body_indices, :
        ]
        points, row_axes, column_axes, normal_axes = (
            self.tactile_layout.taxel_geometry_world(fingertip_states)
        )
        row_axes = row_axes[:, :, None, None, :]
        column_axes = column_axes[:, :, None, None, :]
        normal_axes = normal_axes[:, :, None, None, :]

        # Lift the marks just above the collision surface to avoid z-fighting.
        points = points + normal_axes * self.tactile_debug_surface_offset_m
        half_size = 0.5 * self.tactile_debug_point_size_m
        row_segments = torch.stack(
            (points - row_axes * half_size, points + row_axes * half_size), dim=-2
        )
        column_segments = torch.stack(
            (
                points - column_axes * half_size,
                points + column_axes * half_size,
            ),
            dim=-2,
        )
        normal_segments = torch.stack(
            (points, points + normal_axes * self.tactile_debug_normal_length_m),
            dim=-2,
        )
        vertices = torch.stack(
            (row_segments, column_segments, normal_segments), dim=-3
        )

        intensity = torch.clamp(
            self.tactile_forces[:draw_env_count]
            / self.tactile_force_normalization,
            min=0.0,
            max=1.0,
        )
        colors = torch.stack(
            (
                intensity,
                0.8 * (1.0 - intensity),
                1.0 - intensity,
            ),
            dim=-1,
        )
        colors = colors.unsqueeze(-2).expand(-1, -1, -1, -1, 3, -1)

        for env_id in range(draw_env_count):
            env_vertices = (
                vertices[env_id].reshape(-1, 2, 3).detach().cpu().numpy()
            )
            env_colors = colors[env_id].reshape(-1, 3).detach().cpu().numpy()
            self.gym.add_lines(
                self.viewer,
                self.envs[env_id],
                env_vertices.shape[0],
                env_vertices,
                env_colors,
            )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.prev_targets + 1 / 24 * self.actions
        self.cur_targets[:] = tensor_clamp(targets, self.SimpleHand_dof_lower_limits, self.SimpleHand_dof_upper_limits)
        self.prev_targets[:] = self.cur_targets.clone()
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # apply new forces
            obj_mass = to_torch(
                [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for
                 env in self.envs], device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

    def reset(self):
        super().reset()
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict['tactile_forces'] = self.tactile_forces.to(self.rl_device)
        self.obs_dict['tactile_depth'] = self.tactile_depth.to(self.rl_device)
        self.obs_dict['tactile_backend'] = self.tactile_backend_active
        return self.obs_dict

    def step(self, actions):
        super().step(actions)
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict['tactile_forces'] = self.tactile_forces.to(self.rl_device)
        self.obs_dict['tactile_depth'] = self.tactile_depth.to(self.rl_device)
        self.obs_dict['tactile_backend'] = self.tactile_backend_active
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def update_low_level_control(self):
        previous_dof_pos = self.SimpleHand_dof_pos.clone()
        self._refresh_gym()
        if self.torque_control:
            dof_pos = self.SimpleHand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff = dof_vel.clone()
            torques = self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
            self.torques = torch.clip(torques, -0.5, 0.5).clone()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def check_termination(self, object_pos):
        resets = torch.logical_or(
            torch.less(object_pos[:, -1], self.reset_z_threshold),
            torch.greater_equal(self.progress_buf, self.max_episode_length),
        )
        return resets

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config['randomizeMass']
        # 调整质量范围，避免太重物体
        self.randomize_mass_lower = rand_config.get('randomizeMassLower', 0.02)  # 从0.01提高
        self.randomize_mass_upper = rand_config.get('randomizeMassUpper', 0.15)  # 从0.25降低
        # self.randomize_mass_lower = rand_config['randomizeMassLower']
        # self.randomize_mass_upper = rand_config['randomizeMassUpper']

        self.randomize_com = rand_config['randomizeCOM']
        self.randomize_com_lower = rand_config['randomizeCOMLower']
        self.randomize_com_upper = rand_config['randomizeCOMUpper']
        # 增加摩擦力范围，帮助Digit传感器更好地抓握
        self.randomize_friction = rand_config['randomizeFriction']
        # self.randomize_friction_lower = rand_config['randomizeFrictionLower']
        # self.randomize_friction_upper = rand_config['randomizeFrictionUpper']
        self.randomize_friction_lower = rand_config.get('randomizeFrictionLower', 0.8)  # 从0.3提高
        self.randomize_friction_upper = rand_config.get('randomizeFrictionUpper', 2.5)  # 从3.0降低但保持较高
    
        self.randomize_scale = rand_config['randomizeScale']
        self.scale_list_init = rand_config['scaleListInit']
        self.randomize_scale_list = rand_config['randomizeScaleList']
        self.randomize_scale_lower = rand_config['randomizeScaleLower']
        self.randomize_scale_upper = rand_config['randomizeScaleUpper']
        self.randomize_pd_gains = rand_config['randomizePDGains']
        self.randomize_p_gain_lower = rand_config['randomizePGainLower']
        self.randomize_p_gain_upper = rand_config['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_config['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_config['randomizeDGainUpper']
        self.joint_noise_scale = rand_config['jointNoiseScale']

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config['enableObjPos']
        self.enable_priv_obj_mass = p_config['enableObjMass']
        self.enable_priv_obj_scale = p_config['enableObjScale']
        self.enable_priv_obj_com = p_config['enableObjCOM']
        self.enable_priv_obj_friction = p_config['enableObjFriction']

    def _setup_tactile_config(self, tactile_config):
        self.tactile_enabled = tactile_config.get('enabled', False)
        self.tactile_link_names = tactile_config.get('linkNames', [])
        self.tactile_rows = int(tactile_config.get('rows', 6))
        self.tactile_columns = int(tactile_config.get('columns', 6))
        self.tactile_surface_centers = tactile_config.get('surfaceCenters', [])
        self.tactile_row_axes = tactile_config.get('rowAxes', [])
        self.tactile_column_axes = tactile_config.get('columnAxes', [])
        self.tactile_normal_axes = tactile_config.get('normalAxes', [])
        self.tactile_surface_size = tactile_config.get('surfaceSize', [0.024, 0.018])
        self.tactile_contact_sigma = tactile_config.get('contactSigma', 0.004)
        self.tactile_force_threshold = tactile_config.get('forceThresholdN', 0.0)
        self.tactile_max_force = tactile_config.get('maxForceN', 20.0)
        self.tactile_noise_std = tactile_config.get('noiseStdN', 0.0)
        self.tactile_force_normalization = tactile_config.get('forceNormalizationN', 5.0)
        self.tactile_cache_min_active_fingers = int(
            tactile_config.get('cacheMinActiveFingers', 2)
        )
        self.tactile_cache_activation_threshold = float(
            tactile_config.get(
                'cacheTaxelActivationThresholdN', self.tactile_force_threshold
            )
        )
        self.tactile_debug_point_size_m = float(
            tactile_config.get('debugPointSizeM', 0.001)
        )
        self.tactile_debug_normal_length_m = float(
            tactile_config.get('debugNormalLengthM', 0.003)
        )
        self.tactile_debug_surface_offset_m = float(
            tactile_config.get('debugSurfaceOffsetM', 0.0004)
        )
        self.tactile_debug_max_envs = int(tactile_config.get('debugMaxEnvs', 4))
        self.tactile_backend_requested = tactile_config.get('backend', 'auto')
        self.tactile_allow_contact_fallback = tactile_config.get(
            'allowContactForceFallback', True
        )
        self.tactile_stiffness_n_per_m = tactile_config.get('stiffnessNPerM', 1.0)
        self.tactile_elastomer_compliance = tactile_config.get(
            'elastomerCompliance', 30.0
        )
        self.tactile_sdf_normal_sign = tactile_config.get('sdfNormalSign', 1.0)
        self.tactile_indenter_mesh_pos = tactile_config.get(
            'indenterMeshPosition', [0.0, 0.0, 0.0]
        )
        self.tactile_indenter_mesh_orn = tactile_config.get(
            'indenterMeshOrientation', [0.0, 0.0, 0.0, 1.0]
        )
        self.tactile_sdf_bounds_min = tactile_config.get('sdfBoundsMin', None)
        self.tactile_sdf_bounds_max = tactile_config.get('sdfBoundsMax', None)
        self.tactile_sdf_query_margin_m = tactile_config.get(
            'sdfQueryMarginM', 0.0005
        )
        if (self.tactile_sdf_bounds_min is None) != (
            self.tactile_sdf_bounds_max is None
        ):
            raise ValueError('tactile.sdfBoundsMin and sdfBoundsMax must be paired')

        if not self.tactile_enabled:
            self.tactile_obs_dim = 0
            return
        if self.tactile_force_normalization <= 0.0:
            raise ValueError('tactile.forceNormalizationN must be positive')
        if self.tactile_debug_point_size_m <= 0.0:
            raise ValueError('tactile.debugPointSizeM must be positive')
        if self.tactile_debug_normal_length_m < 0.0:
            raise ValueError('tactile.debugNormalLengthM cannot be negative')
        if self.tactile_debug_surface_offset_m < 0.0:
            raise ValueError('tactile.debugSurfaceOffsetM cannot be negative')
        if self.tactile_debug_max_envs <= 0:
            raise ValueError('tactile.debugMaxEnvs must be positive')
        if self.tactile_backend_requested not in ('auto', 'vt_refine', 'contact_force'):
            raise ValueError(
                'tactile.backend must be auto, vt_refine, or contact_force'
            )
        if self.tactile_stiffness_n_per_m <= 0.0:
            raise ValueError('tactile.stiffnessNPerM must be positive')
        if self.tactile_elastomer_compliance < 0.0:
            raise ValueError('tactile.elastomerCompliance cannot be negative')
        num_sensors = len(self.tactile_link_names)
        if not 1 <= self.tactile_cache_min_active_fingers <= num_sensors:
            raise ValueError(
                'tactile.cacheMinActiveFingers must be between 1 and the '
                'number of tactile sensors'
            )
        if self.tactile_cache_activation_threshold < 0.0:
            raise ValueError(
                'tactile.cacheTaxelActivationThresholdN cannot be negative'
            )
        vector_fields = {
            'surfaceCenters': self.tactile_surface_centers,
            'rowAxes': self.tactile_row_axes,
            'columnAxes': self.tactile_column_axes,
            'normalAxes': self.tactile_normal_axes,
        }
        for name, values in vector_fields.items():
            if len(values) != num_sensors:
                raise ValueError(f'tactile.{name} must contain one entry per link')
        self.tactile_obs_dim = num_sensors * self.tactile_rows * self.tactile_columns
        proprio_dim = self.config['env'].get('hora', {}).get('proprioDim', 32)
        expected_num_observations = 3 * proprio_dim + self.tactile_obs_dim
        configured_num_observations = self.config['env']['numObservations']
        if configured_num_observations != expected_num_observations:
            raise ValueError(
                'env.numObservations must be '
                f'{expected_num_observations} when tactile sensing is enabled, got '
                f'{configured_num_observations}'
            )

    def _update_priv_buf(self, env_id, name, value, lower=None, upper=None):
        # normalize to -1, 1
        s, e = self.priv_info_dict[name]
        if eval(f'self.enable_priv_{name}'):
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            if type(lower) is list or upper is list:
                lower = to_torch(lower, dtype=torch.float, device=self.device)
                upper = to_torch(upper, dtype=torch.float, device=self.device)
            if lower is not None and upper is not None:
                value = (2.0 * value - upper - lower) / (upper - lower)
            self.priv_info_buf[env_id, s:e] = value
        else:
            self.priv_info_buf[env_id, s:e] = 0

    def _setup_object_info(self, o_config):
        self.object_type = o_config['type']
        raw_prob = o_config['sampleProb']
        assert (sum(raw_prob) == 1)

        primitive_list = self.object_type.split('+')
        print('---- Primitive List ----')
        print(primitive_list)
        self.object_type_prob = []
        self.object_type_list = []
        self.asset_files_dict = {
            'simple_tennis_ball': 'assets/ycb/056_tennis_ball.urdf',
            'ball':'assets/ball_sdf.urdf',
            'cube':'assets/cube_sdf.urdf',
            'cube25':'assets/cube25_sdf.urdf',
            'cube35':'assets/cube35_sdf.urdf',
            'cube66':'assets/cube66_sdf.urdf',
            'cylinder':'assets/cylinder_sdf.urdf',
        }
        self.non_sdf_asset_files_dict = {
            'ball': 'assets/ball.urdf',
            'cube': 'assets/cube.urdf',
            'cube25': 'assets/cube25.urdf',
            'cube35': 'assets/cube35.urdf',
            'cube66': 'assets/cube66.urdf',
            'cylinder': 'assets/cylinder.urdf',
        }
        for p_id, prim in enumerate(primitive_list):
            if prim == 'cylinder':
                self.object_type_list += ['cylinder']
                self.object_type_prob += [raw_prob[p_id]]
            elif 'cuboid' in prim:
                subset_name = self.object_type.split('_')[-1]
                cuboids = sorted(glob(f'../assets/cuboid/{subset_name}/*.urdf'))
                cuboid_list = [f'cuboid_{i}' for i in range(len(cuboids))]
                self.object_type_list += cuboid_list
                for i, name in enumerate(cuboids):
                    self.asset_files_dict[f'cuboid_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cuboid_list) for _ in cuboid_list]
            elif 'cylinder' in prim:
                subset_name = self.object_type.split('_')[-1]
                cylinders = sorted(glob(f'assets/cylinder/{subset_name}/*.urdf'))
                cylinder_list = [f'cylinder_{i}' for i in range(len(cylinders))]
                self.object_type_list += cylinder_list
                for i, name in enumerate(cylinders):
                    self.asset_files_dict[f'cylinder_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cylinder_list) for _ in cylinder_list]
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]
        print('---- Object List ----')
        print(self.object_type_list)
        assert (len(self.object_type_list) == len(self.object_type_prob))

    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        self.prop_hist_len = self.config['env']['hora']['propHistoryLen']
        self.num_env_factors = self.config['env']['hora']['privInfoDim']
        self.priv_info_buf = torch.zeros((num_envs, self.num_env_factors), device=self.device, dtype=torch.float)
        self.proprio_hist_buf = torch.zeros(
            (num_envs, self.prop_hist_len, self.obs_history_dim),
            device=self.device,
            dtype=torch.float,
        )
        tactile_shape = (
            num_envs,
            len(self.tactile_link_names),
            self.tactile_rows,
            self.tactile_columns,
        )
        self.tactile_forces = torch.zeros(
            tactile_shape, device=self.device, dtype=torch.float
        )
        self.tactile_depth = torch.zeros(
            tactile_shape, device=self.device, dtype=torch.float
        )

    # def _setup_reward_config(self, r_config):
    #     self.angvel_clip_min = r_config['angvelClipMin']
    #     self.angvel_clip_max = r_config['angvelClipMax']
    #     self.rotate_reward_scale = r_config['rotateRewardScale']
    #     self.object_linvel_penalty_scale = r_config['objLinvelPenaltyScale']
    #     self.pose_diff_penalty_scale = r_config['poseDiffPenaltyScale']
    #     self.torque_penalty_scale = r_config['torquePenaltyScale']
    #     self.work_penalty_scale = r_config['workPenaltyScale']

    def _setup_reward_config(self, r_config):
        # 针对Digit传感器调整奖励权重
        self.angvel_clip_min = r_config['angvelClipMin']
        self.angvel_clip_max = r_config['angvelClipMax']

        # 降低旋转奖励权重，先保证稳定性
        self.rotate_reward_scale = r_config.get('rotateRewardScale', 2.0)  # 线速度奖励系数

        # 增加稳定性相关惩罚的权重
        self.object_linvel_penalty_scale = r_config.get('objLinvelPenaltyScale', 2.0)  # 增加
        self.pose_diff_penalty_scale = r_config.get('poseDiffPenaltyScale', 0.8)  # 增加
        self.torque_penalty_scale = r_config.get('torquePenaltyScale', 0.2)  # 调整
        self.work_penalty_scale = r_config.get('workPenaltyScale', 0.1)  # 调整

    def _create_object_asset(self):
        # object file to asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        hand_asset_file = self.config['env']['asset']['handAsset']
        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = True
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01

        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        else:
            hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_asset_options)
        if self.tactile_enabled:
            hand_body_dict = self.gym.get_asset_rigid_body_dict(self.hand_asset)
            missing_links = [name for name in self.tactile_link_names if name not in hand_body_dict]
            if missing_links:
                raise ValueError(f'Tactile links not found in hand asset: {missing_links}')
            self.tactile_body_indices_list = [
                hand_body_dict[name] for name in self.tactile_link_names
            ]
            self.tactile_body_indices = to_torch(
                self.tactile_body_indices_list,
                dtype=torch.long,
                device=self.device,
            )

        # load object asset
        self.object_asset_list = []
        self.object_sdf_proxy_asset_list = []
        for object_type in self.object_type_list:
            object_asset_file = self.asset_files_dict[object_type]
            # Standard CPU PhysX uses the non-SDF primitive assets. TacSL uses
            # the mesh assets above so it can expose their SDF query tensors.
            if not self.tactile_use_sdf_backend:
                object_asset_file = self.non_sdf_asset_files_dict.get(
                    object_type, object_asset_file
                )
            object_asset_options = gymapi.AssetOptions()
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            self.object_asset_list.append(object_asset)
            if self.tactile_use_sdf_backend:
                proxy_options = gymapi.AssetOptions()
                proxy_options.fix_base_link = True
                proxy_options.disable_gravity = True
                proxy_asset = self.gym.load_asset(
                    self.sim, asset_root, object_asset_file, proxy_options
                )
            else:
                proxy_asset = None
            self.object_sdf_proxy_asset_list.append(proxy_asset)

    def _init_object_pose(self):
        SimpleHand_start_pose = gymapi.Transform()
        SimpleHand_start_pose.p = gymapi.Vec3(0, 0, 0.5)
        # SimpleHand_start_pose.r = gymapi.Quat.from_axis_angle(
        #     gymapi.Vec3(0, 1, 0), -np.pi / 2) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi / 2)
        SimpleHand_start_pose.r = gymapi.Quat(0, 0, 0, 1)
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = SimpleHand_start_pose.p.x
        # Move the initial object 1 cm away from the wrist (+x) and 2 cm
        # toward the little-finger side (+y) in the hand frame.
        pose_dx, pose_dy, pose_dz = 0.020, -0.020, 0.12 # meters

        object_start_pose.p.x = SimpleHand_start_pose.p.x + pose_dx
        object_start_pose.p.y = SimpleHand_start_pose.p.y + pose_dy
        object_start_pose.p.z = SimpleHand_start_pose.p.z + pose_dz

        # object_start_pose.p.y = SimpleHand_start_pose.p.y - 0.01
        # for grasp pose generation, it is used to initialize the object
        # it should be slightly higher than the fingertip
        # so it is set to be 0.66 for internal SimpleHand and 0.64 for the public SimpleHand
        # ----
        # for in-hand object rotation, the initialization of z is only used in the first step
        # it is set to be 0.65 for backward compatibility
        object_z = 0.58 if self.save_init_pose else 0.57 #
        if 'internal' not in self.grasp_cache_name:
            object_z += 0.03 #
        object_start_pose.p.z = object_z
        return SimpleHand_start_pose, object_start_pose


def compute_hand_reward(
    object_linvel_penalty, object_linvel_penalty_scale: float,
    rotate_reward, rotate_reward_scale: float,
    pose_diff_penalty, pose_diff_penalty_scale: float,
    torque_penalty, torque_pscale: float,
    work_penalty, work_pscale: float,
    contact_stability_reward=None, contact_stability_scale=0.0,
):
    reward = rotate_reward_scale * rotate_reward
    # Distance from the hand to the object
    reward = reward + object_linvel_penalty * object_linvel_penalty_scale
    reward = reward + pose_diff_penalty * pose_diff_penalty_scale
    reward = reward + torque_penalty * torque_pscale
    reward = reward + work_penalty * work_pscale
    # 新增奖励项
    if contact_stability_reward is not None:
        reward = reward + contact_stability_reward * contact_stability_scale

    return reward


def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Adapted from PyTorch3D:
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_axis_angle
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., :3], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., 3:])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., :3] / sin_half_angles_over_angles
