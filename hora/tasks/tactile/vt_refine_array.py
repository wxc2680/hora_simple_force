# Algorithm adapted from NVlabs/easysim-envs tactile_sensor.py (MIT),
# commit 5512c3389685b217fae046a32b0b95f2f3020c39 used by VT-Refine.

import os

import torch

from .piezoresistive_array import _quat_rotate, _quat_rotate_inverse


def _quat_mul(quaternion_a, quaternion_b):
    ax, ay, az, aw = quaternion_a.unbind(dim=-1)
    bx, by, bz, bw = quaternion_b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


class VTRefineTactileArray:
    """VT-Refine/TacSL SDF taxel backend adapted to a 6x6 pressure array.

    The implementation follows the official EasySim tactile sensor: query
    points are transformed into the indenter mesh frame, TacSL's SDF view is
    refreshed, and negative signed distance is converted to normal force with
    ``force = stiffness_n_per_m * depth``. The class is deliberately isolated
    from the task so the standard Isaac Gym contact-force backend can remain a
    runnable fallback when the TacSL binary is not installed.
    """

    backend_name = 'vt_refine_sdf'

    def __init__(
        self,
        gym,
        gymtorch,
        sim,
        shape_ids,
        local_taxel_positions,
        normal_axes,
        stiffness_n_per_m=1.0,
        normal_sign=1.0,
        force_threshold=0.0,
        max_force=float('inf'),
        noise_std=0.0,
        mesh_pos_local=None,
        mesh_orn_local=None,
        sdf_bounds_min=None,
        sdf_bounds_max=None,
        sdf_query_margin_m=0.0005,
    ):
        self.gym = gym
        self.gymtorch = gymtorch
        self.sim = sim
        self.device = local_taxel_positions.device
        self.local_taxel_positions = local_taxel_positions
        self.num_sensors = local_taxel_positions.shape[0]
        self.rows = local_taxel_positions.shape[1]
        self.columns = local_taxel_positions.shape[2]
        self.normal_axes = normal_axes
        self.num_queries = self.num_sensors * self.rows * self.columns
        self.shape_ids = (
            shape_ids.to(device=self.device, dtype=torch.int32)
            .view(-1, 1)
            .contiguous()
        )
        self.stiffness_n_per_m = float(stiffness_n_per_m)
        self.normal_sign = float(normal_sign)
        self.force_threshold = float(force_threshold)
        self.max_force = float(max_force)
        self.noise_std = float(noise_std)
        self.sdf_query_margin_m = float(sdf_query_margin_m)
        if self.sdf_query_margin_m < 0.0:
            raise ValueError('sdf_query_margin_m cannot be negative')
        if mesh_pos_local is None:
            mesh_pos_local = [0.0, 0.0, 0.0]
        if mesh_orn_local is None:
            mesh_orn_local = [0.0, 0.0, 0.0, 1.0]
        self.mesh_pos_local = torch.as_tensor(
            mesh_pos_local, dtype=torch.float32, device=self.device
        )
        self.mesh_orn_local = torch.as_tensor(
            mesh_orn_local, dtype=torch.float32, device=self.device
        )
        if self.mesh_pos_local.shape != (3,) or self.mesh_orn_local.shape != (4,):
            raise ValueError('indenter mesh local pose must be [3] and [4]')
        if (sdf_bounds_min is None) != (sdf_bounds_max is None):
            raise ValueError('sdf_bounds_min and sdf_bounds_max must be provided together')
        if sdf_bounds_min is None:
            self.sdf_bounds_min = None
            self.sdf_bounds_max = None
        else:
            self.sdf_bounds_min = torch.as_tensor(
                sdf_bounds_min, dtype=torch.float32, device=self.device
            )
            self.sdf_bounds_max = torch.as_tensor(
                sdf_bounds_max, dtype=torch.float32, device=self.device
            )
            if (
                self.sdf_bounds_min.shape != (3,)
                or self.sdf_bounds_max.shape != (3,)
                or torch.any(self.sdf_bounds_max <= self.sdf_bounds_min)
            ):
                raise ValueError('SDF bounds must be ordered vectors of shape [3]')
        if self.stiffness_n_per_m <= 0.0:
            raise ValueError('stiffness_n_per_m must be positive')
        self._sdf_view = None
        self.last_signed_distance = None
        self.last_depth = None
        self.last_normal_mesh = None
        self.last_normal_world = None

    def acquire_sdf_view_tensor(self):
        """Acquire TacSL's SDF view before ``gym.prepare_sim`` is called."""
        if not hasattr(self.gym, 'acquire_sdf_view_tensor'):
            raise RuntimeError(
                'TacSL Isaac Gym is required for the VT-Refine backend; '
                'the active gym binding has no acquire_sdf_view_tensor method'
            )
        sdf_view = self.gym.acquire_sdf_view_tensor(self.sim, 1, self.num_queries)
        self._sdf_view = self.gymtorch.wrap_tensor(sdf_view)
        return self._sdf_view

    def _require_sdf_view(self):
        if self._sdf_view is None:
            raise RuntimeError('acquire_sdf_view_tensor must run before sensing')

    def sense(self, fingertip_states, object_state, object_scales=None):
        """Return SDF-derived normal forces with shape [B, sensors, rows, cols]."""
        self._require_sdf_view()
        batch_size = fingertip_states.shape[0]
        if fingertip_states.shape[1:] != (self.num_sensors, 13):
            raise ValueError('fingertip_states has an unexpected shape')
        if object_state.shape != (batch_size, 13):
            raise ValueError('object_state must have shape [B, 13]')
        if self.shape_ids.shape[0] != batch_size:
            raise ValueError('shape_ids and state batch dimensions differ')
        if object_scales is None:
            object_scales = torch.ones(
                batch_size, dtype=torch.float32, device=self.device
            )
        else:
            object_scales = torch.as_tensor(
                object_scales, dtype=torch.float32, device=self.device
            ).view(-1)
        if object_scales.shape != (batch_size,) or torch.any(object_scales <= 0.0):
            raise ValueError('object_scales must contain one positive value per env')

        fingertip_pos = fingertip_states[..., :3]
        fingertip_orn = fingertip_states[..., 3:7]
        local_points = self.local_taxel_positions.reshape(self.num_sensors, -1, 3)
        local_points = local_points.unsqueeze(0).expand(batch_size, -1, -1, -1)

        fingertip_pos_expanded = fingertip_pos.unsqueeze(2).expand(
            -1, -1, self.rows * self.columns, -1
        )
        fingertip_orn_expanded = fingertip_orn.unsqueeze(2).expand(
            -1, -1, self.rows * self.columns, -1
        )
        points_world = _quat_rotate(fingertip_orn_expanded, local_points)
        points_world = points_world + fingertip_pos_expanded
        points_world = points_world.reshape(batch_size, self.num_queries, 3)

        object_pos = object_state[:, :3]
        object_orn = object_state[:, 3:7]
        scaled_mesh_pos_local = (
            self.mesh_pos_local.unsqueeze(0) * object_scales.unsqueeze(-1)
        )
        mesh_pos = object_pos + _quat_rotate(object_orn, scaled_mesh_pos_local)
        mesh_orn = _quat_mul(
            object_orn, self.mesh_orn_local.unsqueeze(0).expand(batch_size, -1)
        )
        mesh_pos_expanded = mesh_pos.unsqueeze(1).expand(-1, self.num_queries, -1)
        mesh_orn_expanded = mesh_orn.unsqueeze(1).expand(-1, self.num_queries, -1)
        points_mesh_physical = _quat_rotate_inverse(
            mesh_orn_expanded, points_world - mesh_pos_expanded
        )
        # The cooked SDF remains in the unscaled asset frame. Isaac Gym actor
        # scaling therefore requires inverse-scaling query positions and
        # scaling the returned signed distances back to physical meters.
        points_mesh = points_mesh_physical / object_scales[:, None, None]

        if self.sdf_bounds_min is None:
            valid_query = torch.ones(
                batch_size, self.num_queries, dtype=torch.bool, device=self.device
            )
            query_points_mesh = points_mesh
        else:
            lower = self.sdf_bounds_min + self.sdf_query_margin_m
            upper = self.sdf_bounds_max - self.sdf_query_margin_m
            valid_query = torch.logical_and(
                (points_mesh >= lower.view(1, 1, 3)).all(dim=-1),
                (points_mesh <= upper.view(1, 1, 3)).all(dim=-1),
            )
            # The TacSL kernel assumes every query lies inside the cooked SDF
            # volume. Invalid points are safely queried at the volume center,
            # then masked back to no-contact below.
            fallback_query = ((self.sdf_bounds_min + self.sdf_bounds_max) * 0.5).view(
                1, 1, 3
            )
            query_points_mesh = torch.where(
                valid_query.unsqueeze(-1), points_mesh, fallback_query
            )
        query_points_mesh = query_points_mesh.contiguous()

        if os.environ.get('VT_REFINE_DEBUG') == '1':
            print(
                '[VTRefine] shape_ids=', self.shape_ids.detach().cpu().tolist(),
                'shape_ids_stride=', self.shape_ids.stride(),
                'shape_ids_contiguous=', self.shape_ids.is_contiguous(),
                'query_stride=', query_points_mesh.stride(),
                'query_contiguous=', query_points_mesh.is_contiguous(),
                'object_pos=', object_pos.detach().cpu().tolist(),
                'object_orn=', object_orn.detach().cpu().tolist(),
                'object_scales=', object_scales.detach().cpu().tolist(),
                'query_mesh_min=', points_mesh.detach().amin(dim=(0, 1)).cpu().tolist(),
                'query_mesh_max=', points_mesh.detach().amax(dim=(0, 1)).cpu().tolist(),
                'valid_queries=', valid_query.detach().sum(dim=-1).cpu().tolist(),
            )
        # Match EasySim's synchronization before the custom PhysX SDF query.
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        # TacSL expects [num_envs, 1] shape ids and [num_envs, 1, Q, 3] queries.
        self.gym.refresh_sdf_view_tensor(
            self.sim,
            self.gymtorch.unwrap_tensor(self.shape_ids),
            self.gymtorch.unwrap_tensor(query_points_mesh.unsqueeze(1)),
        )
        sdf = self._sdf_view[:, 0, :, :]
        signed_distance = sdf[:, :, 3].reshape(
            batch_size, self.num_sensors, self.rows, self.columns
        )
        signed_distance = signed_distance * object_scales[:, None, None, None]
        normal_mesh = sdf[:, :, :3].reshape(
            batch_size, self.num_sensors, self.rows, self.columns, 3
        )
        valid_taxels = valid_query.reshape(
            batch_size, self.num_sensors, self.rows, self.columns
        )
        signed_distance = torch.where(
            valid_taxels, signed_distance, torch.ones_like(signed_distance)
        )
        normal_mesh = torch.where(
            valid_taxels.unsqueeze(-1), normal_mesh, torch.zeros_like(normal_mesh)
        )
        depth = (-signed_distance).clamp(min=0.0)
        normal_world = _quat_rotate(
            mesh_orn[:, None, None, None, :].expand(
                -1, self.num_sensors, self.rows, self.columns, -1
            ),
            normal_mesh,
        )
        sensor_normal_world = _quat_rotate(
            fingertip_orn[:, :, None, None, :].expand(
                -1, -1, self.rows, self.columns, -1
            ),
            self.normal_axes[None, :, None, None, :].expand(
                batch_size, -1, self.rows, self.columns, -1
            ),
        )
        normal_force = self.normal_sign * self.stiffness_n_per_m * depth * (
            normal_world * sensor_normal_world
        ).sum(dim=-1)
        readings = normal_force.clamp(min=0.0)
        if self.noise_std > 0.0:
            readings = readings + torch.randn_like(readings) * self.noise_std
        readings = readings.clamp(min=0.0, max=self.max_force)
        if self.force_threshold > 0.0:
            readings = torch.where(
                readings >= self.force_threshold, readings, torch.zeros_like(readings)
            )
        self.last_signed_distance = signed_distance
        self.last_depth = depth
        self.last_normal_mesh = normal_mesh
        self.last_normal_world = normal_world
        return readings
