#!/usr/bin/env python3

"""Check TacSL SDF queries when the target shape has a non-zero global index."""

import argparse
from pathlib import Path

from isaacgym import gymapi, gymtorch
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute-device-id', type=int, default=0)
    parser.add_argument('--prefix-shapes', type=int, default=18)
    parser.add_argument('--actor-scale', type=float, default=1.0)
    parser.add_argument('--actor-z', type=float, default=0.0)
    parser.add_argument('--use-hand', action='store_true')
    parser.add_argument('--out-of-bounds', action='store_true')
    parser.add_argument('--hora-queries', action='store_true')
    parser.add_argument('--query-file', type=str, default=None)
    parser.add_argument('--dynamic-ball', action='store_true')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--ground', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    gym = gymapi.acquire_gym()

    if not all(hasattr(gym, name) for name in (
        'acquire_sdf_view_tensor', 'refresh_sdf_view_tensor'
    )):
        raise RuntimeError('TacSL SDF methods are unavailable')

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 120.0
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True
    sim = gym.create_sim(args.compute_device_id, -1, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError('failed to create simulation')

    try:
        env = gym.create_env(
            sim,
            gymapi.Vec3(-0.2, -0.2, -0.2),
            gymapi.Vec3(0.2, 0.2, 0.2),
            1,
        )
        if args.ground:
            gym.add_ground(sim, gymapi.PlaneParams())
        if args.aggregate:
            gym.begin_aggregate(env, 2000, 2000, True)
        fixed = gymapi.AssetOptions()
        fixed.fix_base_link = True
        fixed.disable_gravity = True
        # Each box contributes one rigid shape before the SDF target.
        box = gym.create_box(sim, 0.001, 0.001, 0.001, fixed)
        root = str(Path(__file__).resolve().parents[1])
        if args.use_hand:
            hand_options = gymapi.AssetOptions()
            hand_options.fix_base_link = True
            hand_options.collapse_fixed_joints = True
            hand_options.disable_gravity = True
            hand_options.thickness = 0.001
            hand_options.angular_damping = 0.01
            hand = gym.load_asset(
                sim, root, 'assets/simplehand/Simple_Hand_V3.urdf', hand_options
            )
            hand_pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.5))
            hand_actor = gym.create_actor(env, hand, hand_pose, 'hand', 0, 0, 0)
            # Match HORA's pre-prepare compliance update on the fingertip shapes.
            hand_props = gym.get_actor_rigid_shape_properties(env, hand_actor)
            for prop in hand_props:
                prop.compliance = 30.0
            gym.set_actor_rigid_shape_properties(env, hand_actor, hand_props)
        else:
            for index in range(args.prefix_shapes):
                pose = gymapi.Transform(p=gymapi.Vec3(0.15 + index * 0.003, 0.0, 0.0))
                gym.create_actor(env, box, pose, f'prefix_{index}', index, 0)

        ball_options = fixed
        if args.dynamic_ball:
            ball_options = gymapi.AssetOptions()
            ball_options.disable_gravity = True
        ball = gym.load_asset(sim, root, 'assets/ball_sdf.urdf', ball_options)
        ball_actor = gym.create_actor(
            env, ball, gymapi.Transform(p=gymapi.Vec3(0.006464, -0.031448, 0.565430)),
            'sdf_ball', args.prefix_shapes, 0
        )
        gym.set_actor_scale(env, ball_actor, args.actor_scale)
        if args.actor_z != 0.0:
            root_state = gym.get_actor_rigid_body_states(
                env, ball_actor, gymapi.STATE_POS
            )
            root_state['pose']['p']['z'] = args.actor_z
            gym.set_actor_rigid_body_states(env, ball_actor, root_state, gymapi.STATE_POS)
        props = gym.get_actor_rigid_shape_properties(env, ball_actor)
        if len(props) != 1:
            raise RuntimeError('target ball must have one rigid shape')
        target_global_index = int(props[0].global_index)
        if args.aggregate:
            gym.end_aggregate(env)

        device = torch.device(f'cuda:{args.compute_device_id}')
        shape_ids = torch.tensor(
            [[target_global_index]], dtype=torch.int32, device=device
        ).contiguous()
        queries = torch.zeros((1, 1, 144, 3), dtype=torch.float32, device=device)
        if args.query_file:
            captured = torch.load(args.query_file, map_location='cpu')
            queries[0, 0] = captured['query_points_mesh'][0]
        elif args.hora_queries:
            # Captured from HORA's first reset, after fingertip/object transforms.
            low = torch.tensor([-0.045049, -0.017731, -0.053306], device=device)
            high = torch.tensor([0.027146, 0.061983, 0.017430], device=device)
            flat = torch.linspace(0.0, 1.0, 144, device=device).unsqueeze(-1)
            queries[0, 0] = low + (high - low) * flat
        else:
            # Keep every query strictly inside the cooked [-0.03, 0.03]^3 volume.
            offsets = torch.linspace(-0.005, 0.005, 12, device=device)
            queries[0, 0, :, 0] = offsets.repeat_interleave(12)
            queries[0, 0, :, 1] = offsets.repeat(12)
        if args.out_of_bounds:
            queries[0, 0, 22:, 0] = 0.04
            queries[0, 0, 22:, 1] = -0.04
            valid = (
                (queries >= -0.0295) & (queries <= 0.0295)
            ).all(dim=-1, keepdim=True)
            center = torch.zeros((1, 1, 1, 3), device=device)
            queries = torch.where(valid, queries, center)
        queries = queries.contiguous()

        sdf_view = gymtorch.wrap_tensor(gym.acquire_sdf_view_tensor(sim, 1, 144))
        gym.prepare_sim(sim)
        for _ in range(args.steps):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        torch.cuda.synchronize(device)
        gym.refresh_sdf_view_tensor(
            sim,
            gymtorch.unwrap_tensor(shape_ids),
            gymtorch.unwrap_tensor(queries),
        )
        torch.cuda.synchronize(device)
        result = sdf_view[:, 0]
        if not torch.isfinite(result).all():
            raise RuntimeError('SDF result contains non-finite values')
        print(f'prefix_shapes: {args.prefix_shapes}')
        print(f'target_global_index: {target_global_index}')
        print(f'sdf_shape: {tuple(sdf_view.shape)}')
        print(
            'signed_distance_min_max: '
            f'{result[..., 3].min().item():.8f} '
            f'{result[..., 3].max().item():.8f}'
        )
        print('VT_REFINE_NONZERO_GLOBAL_INDEX_OK')
    finally:
        gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
