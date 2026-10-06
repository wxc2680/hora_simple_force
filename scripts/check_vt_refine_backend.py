#!/usr/bin/env python3

import sys

import isaacgym
from isaacgym import gymapi


def main():
    gym = gymapi.acquire_gym()
    required_methods = (
        'acquire_sdf_view_tensor',
        'refresh_sdf_view_tensor',
    )
    missing_methods = [name for name in required_methods if not hasattr(gym, name)]
    shape_properties = gymapi.RigidShapeProperties()
    has_global_index = hasattr(shape_properties, 'global_index')

    print(f'isaacgym package: {isaacgym.__file__}')
    print(f'gymapi module: {gymapi.__file__}')
    for name in required_methods:
        print(f'{name}: {hasattr(gym, name)}')
    print(f'RigidShapeProperties.global_index: {has_global_index}')

    if missing_methods or not has_global_index:
        print(
            'The active binding is standard Isaac Gym, not the TacSL build. '
            'Put TacSL IsaacGym_Preview_TacSL_Package/isaacgym/python first '
            'on PYTHONPATH.',
            file=sys.stderr,
        )
        return 1

    print('VT_REFINE_BACKEND_OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
