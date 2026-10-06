# SimpleHand 6x6 fingertip force arrays

The SimpleHand HORA task exposes four simulated piezoresistive arrays, one on
each fingertip. Each array has 6 rows and 6 columns on a 15 mm by 15 mm pad
with a 2.5 mm center-to-center pitch, so the raw output has shape
`[num_envs, 4, 6, 6]` and contains force in newtons.

## Backends

### VT-Refine SDF backend

The `vt_refine` backend ports the official EasySim sensor algorithm used by
VT-Refine. It:

1. Creates four 6x6 taxel grids on the configured fingertip pad surfaces.
2. Sets the fingertip elastomer collision-shape compliance before simulation
   preparation.
3. Acquires one TacSL SDF view for all 144 taxels before `gym.prepare_sim()`.
4. Creates an isolated fixed SDF proxy of the object mesh, so the TacSL query
   target is not the dynamic shape currently in hand contact. The proxy is
   placed outside the environment and never contributes to contact dynamics.
5. Transforms every taxel into the manipulated object's collision-shape frame.
6. Compensates for uniform actor scale when converting query coordinates and
   signed distance back to meters.
7. Calls `refresh_sdf_view_tensor` and reads signed distance and surface normal.
8. Computes `depth = max(0, -signed_distance)` and
   `normal_force = K_N * depth`, projected onto the local pad normal.

The default `sdfNormalSign=-1` is used because the object outward SDF normal
opposes the outward normal of a fingertip pad. The initial
`stiffnessNPerM=1000` calibration maps 1 mm equivalent film compression to
approximately 1 N; it must be fitted against the real sensor.

Only contact on the configured flat `+Z` membrane activates these arrays.
Contact elsewhere on the fingertip collision mesh correctly produces zero
taxel force even when the fingertip's aggregate PhysX contact force is nonzero.

This path uses the TacSL-specific Isaac Gym APIs
`acquire_sdf_view_tensor` and `refresh_sdf_view_tensor`. These are custom
Isaac Gym APIs, not Isaac Sim APIs and not part of standard Preview 4.

### Contact-force compatibility backend

The `contact_force` backend is retained for standard Isaac Gym and for
calibrated piezoresistive sensors that should use solved force rather than
virtual indentation. It projects the PhysX fingertip net contact force onto the
pad normal and distributes it around the estimated center of pressure. With
noise, thresholding, and saturation disabled:

```text
sum(taxel_force) = fingertip_normal_contact_force
```

Standard Isaac Gym does not expose GPU contact positions, so this backend uses
the object-center projection as a spatial estimate. It is not labeled as
VT-Refine. With `backend: auto`, the task selects `vt_refine_sdf` only when
both custom SDF APIs are present; otherwise it reports `contact_force`.

## Task outputs

The policy observation has 240 values:

- 96 values: three frames of 16 normalized joint positions and 16 targets.
- 144 values: the current four 6x6 arrays, flattened in the configured link
  order and divided by `forceNormalizationN`, then clamped to `[0, 1]`.

The unnormalized readings are available separately as:

```python
raw_newtons = obs_dict['tactile_forces']  # [num_envs, 4, 6, 6]
raw_grams = raw_newtons * 1000.0 / 9.8
sdf_depth_m = obs_dict['tactile_depth']   # zero for contact_force
backend = obs_dict['tactile_backend']     # vt_refine_sdf or contact_force
```

The policy receives the same tactile values through `obs_dict['obs']`: entries
`0:96` are the three-frame proprioceptive history and entries `96:240` are the
four flattened, normalized 6x6 arrays. HORA stage 1 and stage 2 therefore both
consume the full 240-value policy observation. Stage 2 keeps its separate
`proprio_hist` input at `[num_envs, 30, 32]`, because that tensor is only used
by its proprioceptive adaptation network.

Existing 96-input checkpoints cannot be loaded into the new 240-input policy
without adapting or retraining the first network layer.

## Installing the TacSL Isaac Gym backend

TacSL's official Isaac Gym instructions require Python 3.8 and the dedicated
binary package, available from the link in
`/opt/tacsl/IsaacGymEnvs-tacsl/isaacgymenvs/tacsl_sensors/install/tacsl_setup.md`.
Install that package separately from standard `/opt/isaacgym`, then put its
`isaacgym/python` directory first on `PYTHONPATH`.

The repository wrapper applies both required paths and defaults to the full
task smoke test:

```bash
./scripts/run_with_tacsl.sh
```

Set `TACSL_ISAACGYM_PYTHON` only if the package is installed elsewhere.

Verify that the active binding is the TacSL build:

```bash
SDK=/opt/tacsl/IsaacGym_Preview_TacSL_Package/isaacgym/python
BIND=$SDK/isaacgym/_bindings/linux-x86_64
LD_LIBRARY_PATH=$BIND:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
PYTHONPATH=$SDK \
python scripts/check_vt_refine_backend.py
```

Then run the controlled four-array SDF smoke test. It creates a fixed ball and
checks one 144-query SDF view, finite signed distances and normals, positive
penetration depth, and a `[1, 4, 6, 6]` force tensor:

```bash
LD_LIBRARY_PATH=$BIND:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
PYTHONPATH=$SDK \
python scripts/smoke_vt_refine_sdf.py
```

Run the complete task-level check, including a controlled 1 mm compression
that should produce about 1 N on one taxel:

```bash
LD_LIBRARY_PATH=$BIND:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
PYTHONPATH=$SDK \
python scripts/smoke_simplehand_vt_refine.py --num-envs 4 --steps 8
```

Run the task in strict VT-Refine mode so a missing or wrong binary cannot
silently select the compatibility backend:

```bash
LD_LIBRARY_PATH=$BIND:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
PYTHONPATH=$SDK \
  python train.py task=SimpleHandHora headless=True pipeline=gpu \
  sim_device=cuda:0 rl_device=cuda:0 \
  task.env.tactile.backend=vt_refine \
  task.env.tactile.allowContactForceFallback=False
```

## Configuration and calibration

All settings are under `env.tactile` in
`configs/task/SimpleHandHora.yaml`:

- `backend` selects `auto`, strict `vt_refine`, or `contact_force`.
- `allowContactForceFallback` controls whether strict VT-Refine may fall back.
- `surfaceCenters`, `rowAxes`, `columnAxes`, and `normalAxes` define each
  membrane in fingertip-local coordinates.
- `surfaceSize` defines the active membrane size in meters.
- `stiffnessNPerM`, `sdfNormalSign`, and `elastomerCompliance` configure the
  VT-Refine penalty/SDF model. `stiffnessNPerM=1000` is only an initial
  calibration, not a universal material constant.
- `indenterMeshPosition` and `indenterMeshOrientation` describe the collision
  mesh frame relative to the object link.
- `contactSigma` controls only the compatibility backend's contact-patch width.
- `forceThresholdN`, `maxForceN`, and `noiseStdN` reproduce sensor threshold,
  saturation, and noise.
- `forceNormalizationN` only scales policy input. It does not change
  `obs_dict['tactile_forces']`.

Calibrate `stiffnessNPerM`, `forceThresholdN`, `maxForceN`, and `noiseStdN`
from repeated measurements with known weights. A 100 gram normal load should
produce a total of about `0.98 N` across the activated taxels. The task-level
smoke uses a controlled 1 mm compression and reports the resulting force so
the scale can be checked before training.

Run the pure tensor tests with:

```bash
python -m unittest tests.test_piezoresistive_array tests.test_vt_refine_array
```

## Visualizing taxel locations

Start the dedicated one-environment viewer from a graphical session:

```bash
./scripts/run_with_tacsl.sh python scripts/visualize_simplehand_taxels.py
```

Each taxel is drawn as a small cross with an outward normal. Cyan means zero
force, and the marks transition to red as they reach `forceNormalizationN`.
The overlay dimensions can be adjusted with `debugPointSizeM`,
`debugNormalLengthM`, and `debugSurfaceOffsetM` under `env.tactile`.

## Cache collection filter

`SimpleHandGrasp` preserves a grasp state only when all of these conditions
hold throughout the collection episode: all fingertips are near the object, at
least two fingertips have rigid-body contact with it, at least two tactile
arrays have one or more active taxels, and the object remains above the reset
height. The tactile condition is controlled by `cacheMinActiveFingers` and
`cacheTaxelActivationThresholdN` under `env.tactile`.

`scripts/gen_grasp.sh` deliberately uses the CPU `contact_force` backend,
because its rigid-contact filter requires CPU PhysX contact pairs. It applies
the tactile-array activation condition to that backend's taxel readings.
