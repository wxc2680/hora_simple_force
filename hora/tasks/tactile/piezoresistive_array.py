import torch


def _quat_rotate(quaternion, vector):
    """Rotate vectors by xyzw quaternions with matching leading dimensions."""
    quaternion_xyz = quaternion[..., :3]
    twice_cross = 2.0 * torch.cross(quaternion_xyz, vector, dim=-1)
    return vector + quaternion[..., 3:] * twice_cross + torch.cross(
        quaternion_xyz, twice_cross, dim=-1
    )


def _quat_rotate_inverse(quaternion, vector):
    inverse = torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)
    return _quat_rotate(inverse, vector)


def active_tactile_sensor_mask(readings, activation_threshold=0.0):
    """Return whether each sensor has at least one activated taxel.

    ``readings`` has shape ``[batch, sensors, rows, columns]`` and is expected
    to contain forces in newtons. A sensor is active when any of its taxels is
    strictly above ``activation_threshold``.
    """
    if readings.ndim != 4:
        raise ValueError('readings must have shape [B, sensors, rows, columns]')
    if activation_threshold < 0.0:
        raise ValueError('activation_threshold cannot be negative')
    activated_taxels = (readings > activation_threshold).reshape(
        readings.shape[0], readings.shape[1], -1
    )
    return torch.any(activated_taxels, dim=-1)


class PiezoresistiveArray:
    """Convert fingertip contact forces into force-conserving taxel readings.

    Isaac Gym Preview 4 exposes net contact force per rigid body but not a GPU
    tensor containing contact positions. The object center projected onto each
    fingertip pad is therefore used as the contact-center estimate. A Gaussian
    contact patch distributes the measured normal force across the taxels.

    This model never derives force from virtual penetration. Its input force is
    the force solved by PhysX, and all outputs are in newtons.
    """

    def __init__(
        self,
        rows,
        columns,
        surface_centers,
        row_axes,
        column_axes,
        normal_axes,
        surface_size,
        contact_sigma,
        force_threshold=0.0,
        max_force=float('inf'),
        noise_std=0.0,
        device='cpu',
    ):
        if rows <= 0 or columns <= 0:
            raise ValueError('rows and columns must be positive')
        self.rows = int(rows)
        self.columns = int(columns)
        self.device = torch.device(device)
        self.surface_centers = self._sensor_vectors(surface_centers, 'surface_centers')
        self.num_sensors = self.surface_centers.shape[0]
        self.row_axes = self._normalized_vectors(row_axes, 'row_axes')
        self.column_axes = self._normalized_vectors(column_axes, 'column_axes')
        self.normal_axes = self._normalized_vectors(normal_axes, 'normal_axes')

        size = torch.as_tensor(surface_size, dtype=torch.float32, device=self.device)
        if size.ndim == 1:
            size = size.unsqueeze(0).expand(self.num_sensors, -1)
        if size.shape != (self.num_sensors, 2) or torch.any(size <= 0.0):
            raise ValueError('surface_size must have shape [2] or [num_sensors, 2] and be positive')
        self.surface_size = size

        row_fraction = (torch.arange(self.rows, device=self.device) + 0.5) / self.rows - 0.5
        column_fraction = (torch.arange(self.columns, device=self.device) + 0.5) / self.columns - 0.5
        self.row_positions = size[:, 0:1] * row_fraction.unsqueeze(0)
        self.column_positions = size[:, 1:2] * column_fraction.unsqueeze(0)

        row_offset = self.row_positions[:, :, None, None] * self.row_axes[:, None, None, :]
        column_offset = self.column_positions[:, None, :, None] * self.column_axes[:, None, None, :]
        self.local_taxel_positions = self.surface_centers[:, None, None, :] + row_offset + column_offset

        sigma = torch.as_tensor(contact_sigma, dtype=torch.float32, device=self.device)
        if sigma.ndim == 0:
            sigma = sigma.expand(self.num_sensors)
        if sigma.shape != (self.num_sensors,) or torch.any(sigma <= 0.0):
            raise ValueError('contact_sigma must be a scalar or have one value per sensor')
        self.contact_sigma = sigma
        self.force_threshold = float(force_threshold)
        self.max_force = float(max_force)
        self.noise_std = float(noise_std)

    def taxel_geometry_world(self, fingertip_states):
        """Return taxel positions and pad axes transformed into world space."""
        if fingertip_states.ndim != 3 or fingertip_states.shape[1:] != (
            self.num_sensors,
            13,
        ):
            raise ValueError('fingertip_states must have shape [B, num_sensors, 13]')

        batch_size = fingertip_states.shape[0]
        fingertip_positions = fingertip_states[..., :3]
        fingertip_rotations = fingertip_states[..., 3:7]
        points_local = self.local_taxel_positions.unsqueeze(0).expand(
            batch_size, -1, -1, -1, -1
        )
        rotations = fingertip_rotations[:, :, None, None, :].expand(
            -1, -1, self.rows, self.columns, -1
        )
        points_world = _quat_rotate(rotations, points_local)
        points_world = points_world + fingertip_positions[:, :, None, None, :]

        def rotate_axes(local_axes):
            axes = local_axes.unsqueeze(0).expand(batch_size, -1, -1)
            return _quat_rotate(fingertip_rotations, axes)

        return (
            points_world,
            rotate_axes(self.row_axes),
            rotate_axes(self.column_axes),
            rotate_axes(self.normal_axes),
        )

    def _sensor_vectors(self, values, name):
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.ndim != 2 or tensor.shape[-1] != 3:
            raise ValueError(f'{name} must have shape [num_sensors, 3]')
        return tensor

    def _normalized_vectors(self, values, name):
        tensor = self._sensor_vectors(values, name)
        if tensor.shape[0] != self.num_sensors:
            raise ValueError(f'{name} must contain one vector per sensor')
        norm = torch.norm(tensor, dim=-1, keepdim=True)
        if torch.any(norm <= 1e-8):
            raise ValueError(f'{name} cannot contain zero vectors')
        return tensor / norm

    def sense(self, fingertip_states, object_positions, contact_forces):
        """Return taxel normal forces with shape [B, sensors, rows, columns]."""
        if fingertip_states.ndim != 3 or fingertip_states.shape[1:] != (self.num_sensors, 13):
            raise ValueError('fingertip_states must have shape [B, num_sensors, 13]')
        batch_size = fingertip_states.shape[0]
        if object_positions.shape != (batch_size, 3):
            raise ValueError('object_positions must have shape [B, 3]')
        if contact_forces.shape != (batch_size, self.num_sensors, 3):
            raise ValueError('contact_forces must have shape [B, num_sensors, 3]')

        fingertip_positions = fingertip_states[..., :3]
        fingertip_rotations = fingertip_states[..., 3:7]
        object_offset_world = object_positions[:, None, :] - fingertip_positions
        object_position_local = _quat_rotate_inverse(fingertip_rotations, object_offset_world)
        surface_offset = object_position_local - self.surface_centers[None, :, :]

        row_coordinate = torch.sum(surface_offset * self.row_axes[None, :, :], dim=-1)
        column_coordinate = torch.sum(surface_offset * self.column_axes[None, :, :], dim=-1)
        row_coordinate = torch.maximum(
            torch.minimum(row_coordinate, self.row_positions[:, -1][None, :]),
            self.row_positions[:, 0][None, :],
        )
        column_coordinate = torch.maximum(
            torch.minimum(column_coordinate, self.column_positions[:, -1][None, :]),
            self.column_positions[:, 0][None, :],
        )

        row_distance = row_coordinate[:, :, None, None] - self.row_positions[None, :, :, None]
        column_distance = (
            column_coordinate[:, :, None, None] - self.column_positions[None, :, None, :]
        )
        squared_distance = row_distance.square() + column_distance.square()
        sigma_squared = self.contact_sigma[None, :, None, None].square()
        weights = torch.exp(-0.5 * squared_distance / sigma_squared)
        weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

        local_normals = self.normal_axes[None, :, :].expand(batch_size, -1, -1)
        world_normals = _quat_rotate(fingertip_rotations, local_normals)
        normal_force = torch.clamp(
            -torch.sum(contact_forces * world_normals, dim=-1), min=0.0
        )
        readings = weights * normal_force[:, :, None, None]

        if self.noise_std > 0.0:
            readings = readings + torch.randn_like(readings) * self.noise_std
        readings = readings.clamp(min=0.0, max=self.max_force)
        if self.force_threshold > 0.0:
            readings = torch.where(
                readings >= self.force_threshold, readings, torch.zeros_like(readings)
            )
        return readings
