import importlib.util
import math
from pathlib import Path
import unittest

import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'hora'
    / 'tasks'
    / 'tactile'
    / 'piezoresistive_array.py'
)
_SPEC = importlib.util.spec_from_file_location('piezoresistive_array', _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PiezoresistiveArray = _MODULE.PiezoresistiveArray
active_tactile_sensor_mask = _MODULE.active_tactile_sensor_mask


def make_sensor():
    return PiezoresistiveArray(
        rows=6,
        columns=6,
        surface_centers=[[0.0, 0.0, 0.0]],
        row_axes=[[1.0, 0.0, 0.0]],
        column_axes=[[0.0, 1.0, 0.0]],
        normal_axes=[[0.0, 0.0, 1.0]],
        surface_size=[0.024, 0.018],
        contact_sigma=0.004,
    )


def make_states(batch_size=1):
    states = torch.zeros(batch_size, 1, 13)
    states[..., 6] = 1.0
    return states


class PiezoresistiveArrayTest(unittest.TestCase):
    def test_shape_and_force_conservation_for_100_grams(self):
        sensor = make_sensor()
        readings = sensor.sense(
            make_states(),
            torch.tensor([[0.0, 0.0, 0.05]]),
            torch.tensor([[[0.0, 0.0, -0.98]]]),
        )

        self.assertEqual(tuple(readings.shape), (1, 1, 6, 6))
        self.assertTrue(torch.allclose(readings.sum(), torch.tensor(0.98), atol=1e-6))

    def test_tangential_and_outward_forces_do_not_activate_sensor(self):
        sensor = make_sensor()
        states = make_states(batch_size=2)
        readings = sensor.sense(
            states,
            torch.tensor([[0.0, 0.0, 0.05], [0.0, 0.0, 0.05]]),
            torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]]),
        )

        self.assertEqual(torch.count_nonzero(readings).item(), 0)

    def test_contact_center_moves_to_edge_taxels(self):
        sensor = make_sensor()
        readings = sensor.sense(
            make_states(),
            torch.tensor([[1.0, -1.0, 0.05]]),
            torch.tensor([[[0.0, 0.0, -2.0]]]),
        )[0, 0]

        peak_row, peak_column = torch.nonzero(readings == readings.max())[0]
        self.assertEqual(peak_row.item(), 5)
        self.assertEqual(peak_column.item(), 0)
        self.assertTrue(torch.allclose(readings.sum(), torch.tensor(2.0), atol=1e-6))

    def test_fingertip_rotation_rotates_sensor_normal(self):
        sensor = make_sensor()
        states = make_states()
        half_angle = math.pi / 4.0
        # A +90 degree rotation around Y maps local +Z to world +X.
        states[0, 0, 4] = math.sin(half_angle)
        states[0, 0, 6] = math.cos(half_angle)
        readings = sensor.sense(
            states,
            torch.tensor([[0.05, 0.0, 0.0]]),
            torch.tensor([[[-3.0, 0.0, 0.0]]]),
        )

        self.assertTrue(torch.allclose(readings.sum(), torch.tensor(3.0), atol=1e-5))

    def test_taxel_debug_geometry_follows_fingertip_pose(self):
        sensor = make_sensor()
        states = make_states()
        states[0, 0, :3] = torch.tensor([1.0, 2.0, 3.0])
        half_angle = math.pi / 4.0
        # A +90 degree rotation around Y maps local +Z to world +X.
        states[0, 0, 4] = math.sin(half_angle)
        states[0, 0, 6] = math.cos(half_angle)

        points, row_axes, column_axes, normal_axes = (
            sensor.taxel_geometry_world(states)
        )

        self.assertEqual(tuple(points.shape), (1, 1, 6, 6, 3))
        self.assertTrue(
            torch.allclose(
                points.mean(dim=(2, 3)),
                torch.tensor([[[1.0, 2.0, 3.0]]]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(row_axes, torch.tensor([[[0.0, 0.0, -1.0]]]), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(column_axes, torch.tensor([[[0.0, 1.0, 0.0]]]), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(normal_axes, torch.tensor([[[1.0, 0.0, 0.0]]]), atol=1e-6)
        )

    def test_active_sensor_mask_requires_an_active_taxel_per_sensor(self):
        readings = torch.zeros(3, 4, 2, 2)
        readings[0, 0, 0, 0] = 0.03
        readings[0, 2, 1, 1] = 0.04
        readings[1, 1, 0, 1] = 0.02
        readings[1, 3, 1, 0] = 0.01

        mask = active_tactile_sensor_mask(readings, activation_threshold=0.01)
        expected = torch.tensor(
            [[True, False, True, False], [False, True, False, False], [False] * 4]
        )
        self.assertTrue(torch.equal(mask, expected))


if __name__ == '__main__':
    unittest.main()
