import importlib.util
from pathlib import Path
import unittest

from omegaconf import OmegaConf
import torch
import trimesh


_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / 'hora' / 'tasks' / 'tactile' / 'piezoresistive_array.py'
_SPEC = importlib.util.spec_from_file_location('simplehand_tactile_layout', _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PiezoresistiveArray = _MODULE.PiezoresistiveArray


class SimpleHandTactileLayoutTest(unittest.TestCase):
    def test_four_6x6_grids_are_on_fingertip_pad_faces(self):
        cfg = OmegaConf.load(_ROOT / 'configs' / 'task' / 'SimpleHandHora.yaml')
        tactile = cfg.env.tactile
        self.assertEqual(tactile.cacheMinActiveFingers, 2)
        self.assertEqual(tactile.cacheTaxelActivationThresholdN, 0.0)
        layout = PiezoresistiveArray(
            rows=tactile.rows,
            columns=tactile.columns,
            surface_centers=[list(x) for x in tactile.surfaceCenters],
            row_axes=[list(x) for x in tactile.rowAxes],
            column_axes=[list(x) for x in tactile.columnAxes],
            normal_axes=[list(x) for x in tactile.normalAxes],
            surface_size=list(tactile.surfaceSize),
            contact_sigma=tactile.contactSigma,
        )

        self.assertEqual(tuple(layout.local_taxel_positions.shape), (4, 6, 6, 3))
        self.assertTrue(
            torch.allclose(
                layout.local_taxel_positions.mean(dim=(1, 2)),
                torch.tensor(tactile.surfaceCenters, dtype=torch.float32),
                atol=1e-7,
            )
        )
        row_pitch = torch.norm(
            layout.local_taxel_positions[:, 1:] - layout.local_taxel_positions[:, :-1],
            dim=-1,
        )
        column_pitch = torch.norm(
            layout.local_taxel_positions[:, :, 1:] - layout.local_taxel_positions[:, :, :-1],
            dim=-1,
        )
        self.assertTrue(torch.allclose(row_pitch, torch.full_like(row_pitch, 0.0025)))
        self.assertTrue(
            torch.allclose(column_pitch, torch.full_like(column_pitch, 0.0025))
        )
        mesh_paths = [
            _ROOT / 'assets/simplehand/meshes/collision/col_thumb_fingertip_link.STL',
            _ROOT / 'assets/simplehand/meshes/collision/col_fingertip_link.STL',
            _ROOT / 'assets/simplehand/meshes/collision/col_fingertip_link.STL',
            _ROOT / 'assets/simplehand/meshes/collision/col_fingertip_link.STL',
        ]
        for sensor_id, mesh_path in enumerate(mesh_paths):
            mesh = trimesh.load(mesh_path, force='mesh')
            points = layout.local_taxel_positions[sensor_id]
            top_z = torch.tensor(mesh.bounds[1, 2], dtype=points.dtype)
            self.assertTrue(torch.allclose(points[..., 2], top_z, atol=1e-7))
            for axis in (0, 1):
                self.assertGreater(points[..., axis].min().item(), mesh.bounds[0, axis])
                self.assertLess(points[..., axis].max().item(), mesh.bounds[1, axis])


if __name__ == '__main__':
    unittest.main()
