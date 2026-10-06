#!/usr/bin/env python3
"""Record a finite SimpleHand PPO rollout with an Isaac Gym camera sensor."""

import argparse
import importlib
import os
from pathlib import Path
import sys
from xml.etree import ElementTree

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument(
        '--warmup-frames', type=int, default=0,
        help='Policy frames to simulate before writing the first video frame.',
    )
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--object-type', choices=('ball', 'cube', 'cube25', 'cube35', 'cube66', 'cylinder'), default='ball')
    parser.add_argument('--sim-device', default='cuda:0')
    parser.add_argument('--graphics-device-id', type=int, default=0)
    parser.add_argument('--tactile-panel', action='store_true')
    parser.add_argument(
        '--offline-render', action='store_true',
        help='Render recorded rigid-body states through EGL instead of an Isaac Gym camera.',
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def import_repo_modules(repo_root):
    repo_root = repo_root.resolve()
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    importlib.invalidate_caches()

    import isaacgym  # noqa: F401 - Isaac Gym must load before torch.
    from isaacgym import gymapi
    import torch

    # Importing train registers the OmegaConf resolvers used by the task config.
    import train  # noqa: F401
    from hydra import compose, initialize_config_dir
    from hora.algo.models.models import ActorCritic
    from hora.algo.models.running_mean_std import RunningMeanStd
    from hora.tasks import isaacgym_task_map
    from hora.utils.misc import set_seed
    from hora.utils.reformat import omegaconf_to_dict

    return {
        'ActorCritic': ActorCritic,
        'RunningMeanStd': RunningMeanStd,
        'compose': compose,
        'gymapi': gymapi,
        'initialize_config_dir': initialize_config_dir,
        'isaacgym_task_map': isaacgym_task_map,
        'omegaconf_to_dict': omegaconf_to_dict,
        'set_seed': set_seed,
        'torch': torch,
    }


def build_config(args, modules):
    config_dir = str(args.repo.resolve() / 'configs')
    with modules['initialize_config_dir'](config_dir=config_dir, version_base=None):
        return modules['compose'](
            config_name='config',
            overrides=[
                'task=SimpleHandHora',
                'headless=True',
                'pipeline=gpu',
                f'sim_device={args.sim_device}',
                f'rl_device={args.sim_device}',
                f'graphics_device_id={args.graphics_device_id}',
                'num_envs=1',
                f'seed={args.seed}',
                'test=True',
            f'task.env.object.type={args.object_type}',
                'task.env.enableDebugVis=False',
                'task.env.forceScale=0.0',
                'task.env.randomForceProbScalar=0.0',
                'task.env.randomization.randomizeMass=False',
                'task.env.randomization.randomizeCOM=False',
                'task.env.randomization.randomizeFriction=False',
                'task.env.randomization.randomizePDGains=False',
                'task.env.randomization.randomizeScale=True',
                'train.algo=PPO',
                'train.ppo.priv_info=True',
                'train.ppo.proprio_adapt=False',
            ],
        )


def build_recording_env(
    config, modules, tactile_panel, width, height, offline_render=False
):
    gymapi = modules['gymapi']
    task_config = modules['omegaconf_to_dict'](config.task)
    # Keep the simulator graphics-free when rendering offline. This avoids the
    # unstable Vulkan camera path while preserving the exact physics rollout.
    task_config['enableCameraSensors'] = not offline_render
    if tactile_panel:
        task_config['env']['tactile']['backend'] = 'contact_force'

    env = modules['isaacgym_task_map'][config.task_name](
        config=task_config,
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=True,
    )
    if offline_render:
        return env

    # The stock SimpleHand task calls gym.prepare_sim() inside its constructor.
    # Creating the camera after that call is the stable path for its binding.
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = width
    camera_properties.height = height
    camera_properties.horizontal_fov = 60.0
    env.record_camera_width = camera_properties.width
    env.record_camera_height = camera_properties.height
    env.record_camera = env.gym.create_camera_sensor(env.envs[0], camera_properties)
    if env.record_camera < 0:
        raise RuntimeError('Isaac Gym could not create the recording camera')
    env.gym.set_camera_location(
        env.record_camera,
        env.envs[0],
        gymapi.Vec3(0.18, 0.22, 0.72),
        gymapi.Vec3(0.0, 0.0, 0.52),
    )
    return env


def quat_xyzw_to_matrix(quaternion):
    x, y, z, w = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rigid_body_pose(state):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = quat_xyzw_to_matrix(state[3:7])
    pose[:3, 3] = state[:3]
    return pose


def look_at_pose(camera_position, target, up=(0.0, 0.0, 1.0)):
    camera_position = np.asarray(camera_position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = camera_up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_position
    return pose


def load_object_collision_mesh(repo_root, env, trimesh):
    """Load the same collision geometry selected by the recording environment."""
    object_types = list(getattr(env, 'object_type_list', []))
    if len(object_types) != 1:
        raise RuntimeError(
            f'offline recording requires one object type, got {object_types}'
        )
    object_type = object_types[0]
    asset_file = env.asset_files_dict[object_type]
    if not getattr(env, 'tactile_use_sdf_backend', False):
        # Match SimpleHandHora._create_object_asset() when TacSL is unavailable.
        asset_file = getattr(env, 'non_sdf_asset_files_dict', {}).get(
            object_type, asset_file
        )
    asset_path = repo_root / asset_file
    if not asset_path.is_file():
        raise FileNotFoundError(f'object asset not found: {asset_path}')

    root = ElementTree.parse(asset_path).getroot()
    geometry = root.find('./link/collision/geometry')
    if geometry is None:
        raise RuntimeError(f'no collision geometry in {asset_path}')
    sphere = geometry.find('sphere')
    box = geometry.find('box')
    cylinder = geometry.find('cylinder')
    mesh_element = geometry.find('mesh')
    if sphere is not None:
        radius = float(sphere.attrib['radius'])
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
        description = f'{asset_file} (sphere radius {radius:.4f} m)'
    elif box is not None:
        size = np.fromstring(box.attrib.get('size', ''), sep=' ')
        if size.shape != (3,):
            raise RuntimeError(f'invalid box size in {asset_path}: {size}')
        mesh = trimesh.creation.box(extents=size)
        description = f'{asset_file} (box {size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f} m)'
    elif cylinder is not None:
        radius = float(cylinder.attrib['radius'])
        length = float(cylinder.attrib['length'])
        mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=64)
        description = f'{asset_file} (cylinder radius {radius:.4f} m, length {length:.4f} m)'
    elif mesh_element is not None:
        mesh_path = asset_path.parent / mesh_element.attrib['filename']
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh = mesh.copy()
        scale = np.fromstring(mesh_element.attrib.get('scale', '1 1 1'), sep=' ')
        if scale.size == 1:
            scale = np.repeat(scale, 3)
        if scale.shape != (3,):
            raise RuntimeError(f'invalid mesh scale in {asset_path}: {scale}')
        mesh.apply_scale(scale)
        description = f'{asset_file} ({mesh_element.attrib["filename"]})'
    else:
        raise RuntimeError(f'unsupported collision geometry in {asset_path}')

    origin = root.find('./link/collision/origin')
    if origin is not None:
        translation = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
        rpy = np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')
        if translation.shape != (3,) or rpy.shape != (3,):
            raise RuntimeError(f'invalid collision origin in {asset_path}')
        transform = trimesh.transformations.euler_matrix(*rpy)
        transform[:3, 3] = translation
        mesh.apply_transform(transform)
    return mesh, description


def get_object_scale(env):
    object_scale = getattr(env, 'object_scales', None)
    if object_scale is not None:
        return float(object_scale[0].detach().cpu())
    scale_slice = getattr(env, 'priv_info_dict', {}).get('obj_scale')
    priv_info = getattr(env, 'priv_info_buf', None)
    if scale_slice is not None and priv_info is not None:
        return float(priv_info[0, scale_slice[0]].detach().cpu())
    return 1.0


class OfflineSimpleHandRenderer:
    """EGL renderer fed from actual Isaac Gym rigid-body state tensors."""

    def __init__(self, repo_root, width, height, env):
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        os.environ.setdefault(
            '__EGL_VENDOR_LIBRARY_FILENAMES',
            '/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
        )
        import pyrender
        import trimesh

        self.pyrender = pyrender
        self.width = width
        self.height = height
        self.scene = pyrender.Scene(
            bg_color=np.array([0.18, 0.19, 0.21, 1.0]),
            ambient_light=np.array([0.13, 0.14, 0.16]),
        )
        visual_root = repo_root / 'assets' / 'Simple_Hand_V4' / 'meshes' / 'visual'
        mesh_paths = {
            'palm': visual_root / 'palm_link.STL',
            'mcp': visual_root / 'mcp_link.STL',
            'pip': visual_root / 'pip_link.STL',
            'dip': visual_root / 'dip_link.STL',
            'tip': visual_root / 'fingertip_link.STL',
            'thumb_mcp': visual_root / 'thumb_mcp_link.STL',
            'thumb_pip': visual_root / 'thumb_pip_link.STL',
            'thumb_dip': visual_root / 'thumb_dip_link.STL',
            'thumb_tip': visual_root / 'thumb_fingertip_link.STL',
        }
        body_meshes = [
            'palm', 'mcp', 'pip', 'dip', 'tip',
            'mcp', 'pip', 'dip', 'tip',
            'mcp', 'pip', 'dip', 'tip',
            'thumb_mcp', 'thumb_pip', 'thumb_dip', 'thumb_tip',
        ]
        hand_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.31, 0.42, 0.54, 1.0],
            metallicFactor=0.10,
            roughnessFactor=0.76,
        )
        self.hand_nodes = []
        loaded_meshes = {}
        for body_index, mesh_key in enumerate(body_meshes):
            if mesh_key not in loaded_meshes:
                mesh = trimesh.load_mesh(mesh_paths[mesh_key], process=False)
                loaded_meshes[mesh_key] = pyrender.Mesh.from_trimesh(
                    mesh, material=hand_material, smooth=True
                )
            self.hand_nodes.append(self.scene.add(loaded_meshes[mesh_key]))

        ball_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.93, 0.42, 0.14, 1.0],
            metallicFactor=0.02,
            roughnessFactor=0.46,
        )
        object_mesh, object_description = load_object_collision_mesh(
            repo_root, env, trimesh
        )
        print(f'Offline object collision mesh: {object_description}')
        self.object_node = self.scene.add(
            pyrender.Mesh.from_trimesh(
                object_mesh, material=ball_material, smooth=True
            )
        )

        ground = trimesh.creation.box(extents=(0.40, 0.40, 0.004))
        ground.apply_translation((0.0, 0.0, 0.37))
        ground_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.09, 0.11, 0.14, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.88,
        )
        self.scene.add(
            pyrender.Mesh.from_trimesh(ground, material=ground_material, smooth=False)
        )

        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(48.0))
        self.scene.add(
            camera,
            pose=look_at_pose((0.23, 0.34, 0.78), (0.02, -0.01, 0.52)),
        )
        for position, intensity in [
            ((0.22, 0.18, 0.86), 0.75),
            ((-0.25, -0.08, 0.70), 0.45),
        ]:
            light_pose = np.eye(4, dtype=np.float32)
            light_pose[:3, 3] = position
            self.scene.add(
                pyrender.PointLight(color=np.ones(3), intensity=intensity),
                pose=light_pose,
            )
        self.renderer = pyrender.OffscreenRenderer(width, height)

    def render(self, rigid_body_states, object_scale):
        states = np.asarray(rigid_body_states, dtype=np.float32)
        if states.shape != (18, 13):
            raise RuntimeError(f'expected rigid body states (18, 13), got {states.shape}')
        for body_index, node in enumerate(self.hand_nodes):
            self.scene.set_pose(node, rigid_body_pose(states[body_index]))
        object_pose = rigid_body_pose(states[17])
        object_pose[:3, :3] *= float(object_scale)
        self.scene.set_pose(self.object_node, object_pose)
        color, _ = self.renderer.render(self.scene)
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    def close(self):
        self.renderer.delete()


def build_policy(config, env, checkpoint_path, modules):
    torch = modules['torch']
    ppo_config = config.train.ppo
    network_config = config.train.network
    model = modules['ActorCritic'](
        {
            'actor_units': list(network_config.mlp.units),
            'priv_mlp_units': list(network_config.priv_mlp.units),
            'actions_num': env.action_space.shape[0],
            'input_shape': env.observation_space.shape,
            'priv_info': bool(ppo_config.priv_info),
            'proprio_adapt': bool(ppo_config.proprio_adapt),
            'priv_info_dim': int(ppo_config.priv_info_dim),
        }
    ).to(config.rl_device)
    observation_normalizer = modules['RunningMeanStd'](
        env.observation_space.shape
    ).to(config.rl_device)

    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    observation_normalizer.load_state_dict(checkpoint['running_mean_std'])
    model.eval()
    observation_normalizer.eval()
    return model, observation_normalizer


def capture_scene(env, gymapi):
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    image = np.asarray(
        env.gym.get_camera_image(
            env.sim, env.envs[0], env.record_camera, gymapi.IMAGE_COLOR
        )
    )
    height = env.record_camera_height
    width = env.record_camera_width
    if image.ndim == 1:
        image = image.view(np.uint8).reshape(height, width, 4)
    elif image.shape == (height, width * 4):
        image = image.reshape(height, width, 4)
    if image.shape != (height, width, 4):
        raise RuntimeError(
            f'expected camera image {(height, width, 4)}, got {image.shape}'
        )
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)


def add_scene_caption(frame, frame_index, total_frames, reward, tactile=False):
    caption = 'Proprioception + tactile' if tactile else 'Proprioception only'
    cv2.rectangle(frame, (14, 14), (430, 76), (16, 16, 16), thickness=-1)
    cv2.putText(
        frame, caption, (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (240, 240, 240), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f'Frame {frame_index:03d}/{total_frames}   Reward {reward:+.3f}',
        (28, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (195, 220, 235), 1, cv2.LINE_AA,
    )
    return frame


def make_tactile_panel(forces, force_normalization, backend, height):
    panel_width = 400
    panel = np.full((height, panel_width, 3), (18, 18, 18), dtype=np.uint8)
    cv2.putText(
        panel, 'TACTILE FORCE (N)', (22, 36), cv2.FONT_HERSHEY_SIMPLEX,
        0.78, (242, 242, 242), 2, cv2.LINE_AA,
    )
    cv2.putText(
        panel, f'Backend: {backend}', (22, 62), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (185, 205, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        panel, f'0 to {force_normalization:.1f} N (red = high)',
        (22, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 205, 220),
        1, cv2.LINE_AA,
    )

    sensor_names = ['THUMB', 'INDEX', 'MIDDLE', 'RING']
    cell_size = 27
    grid_size = cell_size * 6
    x_positions = [24, 210]
    y_positions = [126, 420]
    values = np.asarray(forces, dtype=np.float32)
    if values.shape != (4, 6, 6):
        raise RuntimeError(f'expected tactile shape (4, 6, 6), got {values.shape}')

    for sensor_index, sensor_name in enumerate(sensor_names):
        row, column = divmod(sensor_index, 2)
        x0 = x_positions[column]
        y0 = y_positions[row]
        readings = values[sensor_index]
        scaled = np.clip(readings / force_normalization, 0.0, 1.0)
        heatmap = cv2.applyColorMap(
            np.rint(scaled * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        cv2.putText(
            panel, sensor_name, (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX,
            0.53, (235, 235, 235), 1, cv2.LINE_AA,
        )
        for grid_row in range(6):
            for grid_column in range(6):
                x1 = x0 + grid_column * cell_size
                y1 = y0 + grid_row * cell_size
                color = tuple(int(channel) for channel in heatmap[grid_row, grid_column])
                cv2.rectangle(
                    panel, (x1, y1), (x1 + cell_size, y1 + cell_size),
                    color, thickness=-1,
                )
                cv2.rectangle(
                    panel, (x1, y1), (x1 + cell_size, y1 + cell_size),
                    (42, 42, 42), thickness=1,
                )
        cv2.rectangle(
            panel, (x0 - 1, y0 - 1), (x0 + grid_size + 1, y0 + grid_size + 1),
            (210, 210, 210), thickness=1,
        )
        cv2.putText(
            panel, f'max {readings.max():.2f} N', (x0, y0 + grid_size + 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 210, 225), 1, cv2.LINE_AA,
        )
    return panel


def open_video_writer(output_path, frame, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError('OpenCV could not open an MP4 video writer')
    return writer


def main():
    args = parse_args()
    if (
        args.frames <= 0
        or args.warmup_frames < 0
        or args.fps <= 0
        or args.width <= 0
        or args.height <= 0
    ):
        raise ValueError(
            '--frames, --fps, --width, and --height must be positive; '
            '--warmup-frames cannot be negative'
        )
    args.repo = args.repo.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if not args.repo.is_dir():
        raise FileNotFoundError(f'repository not found: {args.repo}')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'output exists: {args.output}; pass --overwrite')

    modules = import_repo_modules(args.repo)
    config = build_config(args, modules)
    modules['set_seed'](config.seed)
    env = build_recording_env(
        config,
        modules,
        args.tactile_panel,
        args.width,
        args.height,
        offline_render=args.offline_render,
    )
    offline_renderer = (
        OfflineSimpleHandRenderer(args.repo, args.width, args.height, env)
        if args.offline_render
        else None
    )
    writer = None
    try:
        model, observation_normalizer = build_policy(
            config, env, args.checkpoint, modules
        )
        torch = modules['torch']
        obs = env.reset()
        print(
            f'Recording {args.frames} frames after {args.warmup_frames} warmup frames '
            f'from {args.checkpoint.name}; '
            f'tactile backend: {getattr(env, "tactile_backend_active", "disabled")}'
        )
        for _ in range(args.warmup_frames):
            with torch.no_grad():
                policy_input = {
                    'obs': observation_normalizer(obs['obs']),
                    'priv_info': obs['priv_info'],
                }
                actions = torch.clamp(
                    model.act_inference(policy_input), min=-1.0, max=1.0
                )
            obs, _, _, _ = env.step(actions)
        for frame_index in range(1, args.frames + 1):
            with torch.no_grad():
                policy_input = {
                    'obs': observation_normalizer(obs['obs']),
                    'priv_info': obs['priv_info'],
                }
                actions = torch.clamp(
                    model.act_inference(policy_input), min=-1.0, max=1.0
                )
            obs, reward, _, _ = env.step(actions)
            if offline_renderer is not None:
                frame = offline_renderer.render(
                    env.rigid_body_states[0].detach().cpu().numpy(),
                    get_object_scale(env),
                )
            else:
                frame = capture_scene(env, modules['gymapi'])
            reward_value = float(reward[0].detach().cpu())
            frame = add_scene_caption(
                frame, frame_index, args.frames, reward_value, args.tactile_panel
            )
            if args.tactile_panel:
                tactile_panel = make_tactile_panel(
                    obs['tactile_forces'][0].detach().cpu().numpy(),
                    env.tactile_force_normalization,
                    obs['tactile_backend'],
                    frame.shape[0],
                )
                frame = np.concatenate((frame, tactile_panel), axis=1)
            if writer is None:
                writer = open_video_writer(args.output, frame, args.fps)
            writer.write(frame)
            if frame_index % 50 == 0 or frame_index == args.frames:
                print(f'  wrote frame {frame_index}/{args.frames}')
    finally:
        if writer is not None:
            writer.release()
        if offline_renderer is not None:
            offline_renderer.close()
        if getattr(env, 'viewer', None):
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)

    capture = cv2.VideoCapture(str(args.output))
    recorded_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if recorded_frames != args.frames:
        raise RuntimeError(
            f'video frame count is {recorded_frames}, expected {args.frames}'
        )
    print(f'Wrote {args.output} ({recorded_frames} frames at {args.fps} fps)')


if __name__ == '__main__':
    main()
