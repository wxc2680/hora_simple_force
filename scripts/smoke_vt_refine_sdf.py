#!/usr/bin/env python3

"""Exercise the real TacSL SDF tensor API with four 6x6 pressure arrays."""

import argparse
import importlib.util
from pathlib import Path
import sys
import types

from isaacgym import gymapi, gymtorch
import torch


def _load_tactile_classes():
    module_dir = Path(__file__).resolve().parents[1] / 'hora' / 'tasks' / 'tactile'
    package_name = '_vt_refine_smoke_tactile'
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package

    piezo_name = f'{package_name}.piezoresistive_array'
    piezo_spec = importlib.util.spec_from_file_location(
        piezo_name, module_dir / 'piezoresistive_array.py'
    )
    piezo_module = importlib.util.module_from_spec(piezo_spec)
    sys.modules[piezo_name] = piezo_module
    piezo_spec.loader.exec_module(piezo_module)

    vt_name = f'{package_name}.vt_refine_array'
    vt_spec = importlib.util.spec_from_file_location(
        vt_name, module_dir / 'vt_refine_array.py'
    )
    vt_module = importlib.util.module_from_spec(vt_spec)
    sys.modules[vt_name] = vt_module
    vt_spec.loader.exec_module(vt_module)
    return piezo_module.PiezoresistiveArray, vt_module.VTRefineTactileArray


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute-device-id', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--stiffness-n-per-m', type=float, default=1000.0)
    return parser.parse_args()


def _require_tacsl(gym):
    required = ('acquire_sdf_view_tensor', 'refresh_sdf_view_tensor')
    missing = [name for name in required if not hasattr(gym, name)]
    has_global_index = hasattr(gymapi.RigidShapeProperties(), 'global_index')
    if missing or not has_global_index:
        raise RuntimeError(
            'TacSL Isaac Gym binding is not active. Run '
            'scripts/check_vt_refine_backend.py and put the TacSL '
            'isaacgym/python directory first on PYTHONPATH.'
        )


def main():
    args = _parse_args()
    gym = gymapi.acquire_gym()
    _require_tacsl(gym)
    PiezoresistiveArray, VTRefineTactileArray = _load_tactile_classes()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 120.0
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.use_gpu = not args.cpu
    sim_params.use_gpu_pipeline = not args.cpu

    sim = gym.create_sim(
        args.compute_device_id,
        -1,
        gymapi.SIM_PHYSX,
        sim_params,
    )
    if sim is None:
        raise RuntimeError('failed to create Isaac Gym simulation')

    try:
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_root = str(Path(__file__).resolve().parents[1])
        ball_asset = gym.load_asset(
            sim, asset_root, 'assets/ball_sdf.urdf', asset_options
        )
        env = gym.create_env(
            sim,
            gymapi.Vec3(-0.1, -0.1, -0.1),
            gymapi.Vec3(0.1, 0.1, 0.1),
            1,
        )
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        actor = gym.create_actor(env, ball_asset, pose, 'sdf_ball', 0, 0)
        shape_properties = gym.get_actor_rigid_shape_properties(env, actor)
        if len(shape_properties) != 1:
            raise RuntimeError('smoke-test ball must have exactly one rigid shape')
        shape_id = shape_properties[0].global_index

        device = torch.device(
            f'cuda:{args.compute_device_id}' if not args.cpu else 'cpu'
        )
        layout = PiezoresistiveArray(
            rows=6,
            columns=6,
            surface_centers=[[0.0, 0.0, 0.0]] * 4,
            row_axes=[[1.0, 0.0, 0.0]] * 4,
            column_axes=[[0.0, 1.0, 0.0]] * 4,
            normal_axes=[[0.0, 0.0, 1.0]] * 4,
            surface_size=[0.006, 0.006],
            contact_sigma=0.004,
            device=device,
        )
        sensor = VTRefineTactileArray(
            gym=gym,
            gymtorch=gymtorch,
            sim=sim,
            shape_ids=torch.tensor([shape_id], dtype=torch.int32, device=device),
            local_taxel_positions=layout.local_taxel_positions,
            normal_axes=layout.normal_axes,
            stiffness_n_per_m=args.stiffness_n_per_m,
        )

        # TacSL requires this allocation after actors exist and before prepare_sim.
        sdf_view = sensor.acquire_sdf_view_tensor()
        if tuple(sdf_view.shape) != (1, 1, 144, 4):
            raise RuntimeError(f'unexpected SDF view shape: {tuple(sdf_view.shape)}')
        gym.prepare_sim(sim)
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        fingertip_states = torch.zeros(1, 4, 13, device=device)
        fingertip_states[..., 2] = 0.029
        fingertip_states[..., 6] = 1.0
        object_state = torch.zeros(1, 13, device=device)
        object_state[..., 6] = 1.0
        readings = sensor.sense(fingertip_states, object_state)

        tensors = {
            'signed_distance': sensor.last_signed_distance,
            'depth': sensor.last_depth,
            'normal_mesh': sensor.last_normal_mesh,
            'normal_world': sensor.last_normal_world,
            'force': readings,
        }
        for name, tensor in tensors.items():
            if tensor is None or not torch.isfinite(tensor).all():
                raise RuntimeError(f'{name} contains non-finite values')
        if tuple(readings.shape) != (1, 4, 6, 6):
            raise RuntimeError(f'unexpected force shape: {tuple(readings.shape)}')
        if sensor.last_depth.max().item() <= 0.0:
            raise RuntimeError('controlled inside-ball queries produced no penetration depth')
        if readings.max().item() <= 0.0:
            raise RuntimeError(
                'controlled inside-ball queries produced no force; check sdfNormalSign'
            )

        print(f'shape_global_index: {shape_id}')
        print(f'sdf_view_shape: {tuple(sdf_view.shape)}')
        print(
            'depth_m_min_max: '
            f'{sensor.last_depth.min().item():.8f} '
            f'{sensor.last_depth.max().item():.8f}'
        )
        print(
            'force_n_min_max: '
            f'{readings.min().item():.8f} {readings.max().item():.8f}'
        )
        print('VT_REFINE_4X6X6_SDF_SMOKE_OK')
    finally:
        gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
