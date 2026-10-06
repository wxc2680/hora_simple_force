#!/usr/bin/env python3

"""Run the full SimpleHand task with the strict TacSL VT-Refine backend."""

import argparse
from pathlib import Path
import sys

import isaacgym  # noqa: F401 - must be imported before torch
from isaacgym.torch_utils import quat_apply
from hydra import compose, initialize_config_dir
import torch

# Importing train registers the OmegaConf resolvers used by the config tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: F401
from hora.tasks import isaacgym_task_map
from hora.utils.reformat import omegaconf_to_dict
from hora.utils.misc import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--sim-device', default='cuda:0')
    parser.add_argument('--normal-sign', type=float, default=-1.0)
    parser.add_argument(
        '--grasp-cache-name',
        default='internal_SimpleHand3_force',
        help='Cache prefix to load for reset poses.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError('num-envs and steps must be positive')

    config_dir = str(Path(__file__).resolve().parents[1] / 'configs')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name='config',
            overrides=[
                'task=SimpleHandHora',
                'headless=True',
                'pipeline=gpu',
                f'sim_device={args.sim_device}',
                f'rl_device={args.sim_device}',
                f'num_envs={args.num_envs}',
                'task.env.tactile.backend=vt_refine',
                'task.env.tactile.allowContactForceFallback=False',
                f'task.env.tactile.sdfNormalSign={args.normal_sign}',
                f'task.env.grasp_cache_name={args.grasp_cache_name}',
            ],
        )

    set_seed(config.seed)
    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=True,
    )
    try:
        obs = env.reset()
        expected_tactile_shape = (args.num_envs, 4, 6, 6)
        expected_policy_shape = (args.num_envs, 240)
        force_max_seen = 0.0
        depth_max_seen = 0.0
        for _ in range(args.steps):
            actions = torch.zeros(
                (args.num_envs, env.num_actions),
                dtype=torch.float32,
                device=env.rl_device,
            )
            obs, _, _, _ = env.step(actions)
            if obs['tactile_backend'] != 'vt_refine_sdf':
                raise RuntimeError(
                    f"unexpected tactile backend: {obs['tactile_backend']}"
                )
            for name in ('tactile_forces', 'tactile_depth'):
                tensor = obs[name]
                if tuple(tensor.shape) != expected_tactile_shape:
                    raise RuntimeError(
                        f'{name} has shape {tuple(tensor.shape)}, expected '
                        f'{expected_tactile_shape}'
                    )
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f'{name} contains non-finite values')
                if torch.any(tensor < 0.0):
                    raise RuntimeError(f'{name} contains negative values')
            if tuple(obs['obs'].shape) != expected_policy_shape:
                raise RuntimeError(
                    f"policy observation has shape {tuple(obs['obs'].shape)}, "
                    f'expected {expected_policy_shape}'
                )
            tactile_policy_obs = torch.clamp(
                obs['tactile_forces'] / env.tactile_force_normalization,
                min=0.0,
                max=1.0,
            ).flatten(start_dim=1)
            if not torch.allclose(
                obs['obs'][:, -env.tactile_obs_dim:], tactile_policy_obs
            ):
                raise RuntimeError(
                    'the tactile tensor is not present at the end of the '
                    'policy observation'
                )
            force_max_seen = max(
                force_max_seen, obs['tactile_forces'].max().item()
            )
            depth_max_seen = max(depth_max_seen, obs['tactile_depth'].max().item())

        forces = obs['tactile_forces']
        depth = obs['tactile_depth']
        # Natural grasp caches do not guarantee contact on the array's +Z
        # active face. Place a sphere 1 mm into one known taxel to verify the
        # configured metric force conversion deterministically.
        env._refresh_gym()
        fingertip_states = env.rigid_body_states[:, env.tactile_body_indices, :]
        taxel_local = env.tactile_sensor.local_taxel_positions[0, 2, 2]
        taxel_local = taxel_local.unsqueeze(0).expand(args.num_envs, -1)
        fingertip_pos = fingertip_states[:, 0, :3]
        fingertip_orn = fingertip_states[:, 0, 3:7]
        taxel_world = fingertip_pos + quat_apply(fingertip_orn, taxel_local)
        normal_local = env.tactile_sensor.normal_axes[0]
        normal_local = normal_local.unsqueeze(0).expand(args.num_envs, -1)
        normal_world = quat_apply(fingertip_orn, normal_local)
        controlled_state = env.root_state_tensor[env.object_indices].clone()
        controlled_depth_m = 0.001
        sphere_radius_m = 0.03 * env.object_scales
        controlled_state[:, :3] = taxel_world + normal_world * (
            sphere_radius_m - controlled_depth_m
        ).unsqueeze(-1)
        controlled_state[:, 3:7] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=env.device
        )
        controlled_forces = env.tactile_sensor.sense(
            fingertip_states, controlled_state, env.object_scales
        )
        controlled_depth = env.tactile_sensor.last_depth
        if not torch.isfinite(controlled_forces).all():
            raise RuntimeError('controlled tactile forces contain non-finite values')
        if controlled_forces.max().item() <= 0.0:
            raise RuntimeError('controlled 1 mm compression produced no normal force')
        if controlled_depth.max().item() <= 0.0:
            raise RuntimeError('controlled 1 mm compression produced no SDF depth')

        print(f'backend: {obs["tactile_backend"]}')
        print(f'tactile_shape: {tuple(forces.shape)}')
        print(f'policy_observation_shape: {tuple(obs["obs"].shape)}')
        print(f'natural_force_n_max_seen: {force_max_seen:.8f}')
        print(f'natural_depth_m_max_seen: {depth_max_seen:.8f}')
        print(f'controlled_force_n_max: {controlled_forces.max().item():.8f}')
        print(f'controlled_depth_m_max: {controlled_depth.max().item():.8f}')
        print('SIMPLEHAND_VT_REFINE_TASK_SMOKE_OK')
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
