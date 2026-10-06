import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


def _load_tactile_modules():
    package_name = '_hora_tactile_test_package'
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    module_dir = Path(__file__).resolve().parents[1] / 'hora' / 'tasks' / 'tactile'

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


PiezoresistiveArray, VTRefineTactileArray = _load_tactile_modules()


class FakeGymTorch:
    @staticmethod
    def wrap_tensor(tensor):
        return tensor

    @staticmethod
    def unwrap_tensor(tensor):
        return tensor


class SphereSDFGym:
    def __init__(self, batch_size, radius):
        self.batch_size = batch_size
        self.radius = radius
        self.sdf_view = None
        self.acquire_args = None
        self.last_shape_ids = None
        self.last_queries = None

    def acquire_sdf_view_tensor(self, sim, num_shapes, num_queries):
        self.acquire_args = (sim, num_shapes, num_queries)
        self.sdf_view = torch.zeros(self.batch_size, num_shapes, num_queries, 4)
        return self.sdf_view

    def refresh_sdf_view_tensor(self, sim, shape_ids, queries):
        self.last_shape_ids = shape_ids.clone()
        self.last_queries = queries.clone()
        points = queries[:, 0]
        radius = torch.norm(points, dim=-1).clamp_min(1e-12)
        self.sdf_view[:, 0, :, :3] = points / radius.unsqueeze(-1)
        self.sdf_view[:, 0, :, 3] = radius - self.radius


def make_states(batch_size, sensors):
    states = torch.zeros(batch_size, sensors, 13)
    states[..., 6] = 1.0
    return states


class VTRefineTactileArrayTest(unittest.TestCase):
    def test_sdf_query_shape_transform_depth_and_force(self):
        layout = PiezoresistiveArray(
            rows=2,
            columns=3,
            surface_centers=[[0.0, 0.0, 0.02]],
            row_axes=[[1.0, 0.0, 0.0]],
            column_axes=[[0.0, 1.0, 0.0]],
            normal_axes=[[0.0, 0.0, 1.0]],
            surface_size=[0.002, 0.003],
            contact_sigma=0.001,
        )
        gym = SphereSDFGym(batch_size=2, radius=0.03)
        sensor = VTRefineTactileArray(
            gym=gym,
            gymtorch=FakeGymTorch,
            sim='fake_sim',
            shape_ids=torch.tensor([4, 9], dtype=torch.int32),
            local_taxel_positions=layout.local_taxel_positions,
            normal_axes=layout.normal_axes,
            stiffness_n_per_m=100.0,
        )
        sensor.acquire_sdf_view_tensor()

        object_state = torch.zeros(2, 13)
        object_state[:, 6] = 1.0
        readings = sensor.sense(make_states(2, 1), object_state)

        points = layout.local_taxel_positions[0]
        point_radius = torch.norm(points, dim=-1)
        expected_depth = 0.03 - point_radius
        expected = 100.0 * expected_depth * points[..., 2] / point_radius
        self.assertEqual(tuple(readings.shape), (2, 1, 2, 3))
        self.assertTrue(torch.allclose(readings[0, 0], expected, atol=1e-6))
        self.assertTrue(torch.allclose(readings[1, 0], expected, atol=1e-6))
        self.assertTrue(torch.allclose(sensor.last_depth[0, 0], expected_depth, atol=1e-6))
        self.assertEqual(gym.acquire_args, ('fake_sim', 1, 6))
        self.assertEqual(tuple(gym.last_queries.shape), (2, 1, 6, 3))
        self.assertTrue(torch.equal(gym.last_shape_ids, torch.tensor([[4], [9]], dtype=torch.int32)))

    def test_four_6x6_sensors_use_one_144_query_sdf_view(self):
        centers = [[0.0, 0.0, 0.02]] * 4
        row_axes = [[1.0, 0.0, 0.0]] * 4
        column_axes = [[0.0, 1.0, 0.0]] * 4
        normal_axes = [[0.0, 0.0, 1.0]] * 4
        layout = PiezoresistiveArray(
            rows=6,
            columns=6,
            surface_centers=centers,
            row_axes=row_axes,
            column_axes=column_axes,
            normal_axes=normal_axes,
            surface_size=[0.002, 0.002],
            contact_sigma=0.001,
        )
        gym = SphereSDFGym(batch_size=1, radius=0.03)
        sensor = VTRefineTactileArray(
            gym=gym,
            gymtorch=FakeGymTorch,
            sim='fake_sim',
            shape_ids=torch.tensor([2], dtype=torch.int32),
            local_taxel_positions=layout.local_taxel_positions,
            normal_axes=layout.normal_axes,
            stiffness_n_per_m=1.0,
        )
        sensor.acquire_sdf_view_tensor()
        object_state = torch.zeros(1, 13)
        object_state[:, 6] = 1.0
        readings = sensor.sense(make_states(1, 4), object_state)

        self.assertEqual(tuple(readings.shape), (1, 4, 6, 6))
        self.assertEqual(gym.acquire_args, ('fake_sim', 1, 144))
        self.assertEqual(tuple(gym.last_queries.shape), (1, 1, 144, 3))
        self.assertGreater(readings.sum().item(), 0.0)

    def test_uniform_actor_scale_is_applied_in_physical_meters(self):
        layout = PiezoresistiveArray(
            rows=1,
            columns=1,
            surface_centers=[[0.0, 0.0, 0.02]],
            row_axes=[[1.0, 0.0, 0.0]],
            column_axes=[[0.0, 1.0, 0.0]],
            normal_axes=[[0.0, 0.0, 1.0]],
            surface_size=[0.001, 0.001],
            contact_sigma=0.001,
        )
        gym = SphereSDFGym(batch_size=1, radius=0.03)
        sensor = VTRefineTactileArray(
            gym=gym,
            gymtorch=FakeGymTorch,
            sim='fake_sim',
            shape_ids=torch.tensor([3], dtype=torch.int32),
            local_taxel_positions=layout.local_taxel_positions,
            normal_axes=layout.normal_axes,
            stiffness_n_per_m=100.0,
        )
        sensor.acquire_sdf_view_tensor()
        object_state = torch.zeros(1, 13)
        object_state[:, 6] = 1.0
        readings = sensor.sense(
            make_states(1, 1), object_state, object_scales=torch.tensor([2.0])
        )

        # A 30 mm cooked sphere scaled by 2 has a 60 mm physical radius.
        self.assertAlmostEqual(sensor.last_depth.item(), 0.04, places=6)
        self.assertAlmostEqual(readings.item(), 4.0, places=5)
        self.assertAlmostEqual(gym.last_queries[0, 0, 0, 2].item(), 0.01, places=6)

    def test_missing_tacsl_api_is_reported(self):
        layout = PiezoresistiveArray(
            rows=1,
            columns=1,
            surface_centers=[[0.0, 0.0, 0.0]],
            row_axes=[[1.0, 0.0, 0.0]],
            column_axes=[[0.0, 1.0, 0.0]],
            normal_axes=[[0.0, 0.0, 1.0]],
            surface_size=[0.001, 0.001],
            contact_sigma=0.001,
        )
        sensor = VTRefineTactileArray(
            gym=object(),
            gymtorch=FakeGymTorch,
            sim=None,
            shape_ids=torch.tensor([0]),
            local_taxel_positions=layout.local_taxel_positions,
            normal_axes=layout.normal_axes,
        )
        with self.assertRaisesRegex(RuntimeError, 'TacSL Isaac Gym'):
            sensor.acquire_sdf_view_tensor()


if __name__ == '__main__':
    unittest.main()
