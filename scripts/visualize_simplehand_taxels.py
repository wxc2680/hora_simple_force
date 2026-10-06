#!/usr/bin/env python3

"""Visualize SimpleHand taxel locations and live force readings."""

import argparse
from pathlib import Path
import sys

import isaacgym  # noqa: F401 - must be imported before torch
from isaacgym import gymapi
from hydra import compose, initialize_config_dir
import torch

# Importing train registers the OmegaConf resolvers used by the config tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: F401
from hora.tasks import isaacgym_task_map
from hora.utils.misc import set_seed
from hora.utils.reformat import omegaconf_to_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim-device', default='cuda:0')
    parser.add_argument('--graphics-device-id', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = str(Path(__file__).resolve().parents[1] / 'configs')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name='config',
            overrides=[
                'task=SimpleHandHora',
                'headless=False',
                'pipeline=gpu',
                f'sim_device={args.sim_device}',
                f'rl_device={args.sim_device}',
                f'graphics_device_id={args.graphics_device_id}',
                'num_envs=1',
                'task.env.enableDebugVis=True',
                'task.env.tactile.backend=vt_refine',
                'task.env.tactile.allowContactForceFallback=False',
                'task.env.randomization.randomizeMass=False',
                'task.env.randomization.randomizeCOM=False',
                'task.env.randomization.randomizeFriction=False',
                'task.env.randomization.randomizePDGains=False',
            ],
        )

    set_seed(config.seed)
    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=False,
    )
    try:
        env.gym.viewer_camera_look_at(
            env.viewer,
            None,
            gymapi.Vec3(0.18, 0.22, 0.72),
            gymapi.Vec3(0.0, 0.0, 0.52),
        )
        obs = env.reset()
        if obs['tactile_backend'] != 'vt_refine_sdf':
            raise RuntimeError(
                f"unexpected tactile backend: {obs['tactile_backend']}"
            )
        print('Taxels: cyan = no force, red = forceNormalizationN or above')
        print('Press V to toggle viewer sync; press Escape to exit.')
        actions = torch.zeros(
            (1, env.num_actions), dtype=torch.float32, device=env.rl_device
        )
        while True:
            obs, _, _, _ = env.step(actions)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if env.viewer:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
