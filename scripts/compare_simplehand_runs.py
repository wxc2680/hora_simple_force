#!/usr/bin/env python3
"""Compare paired baseline/tactile SimpleHand TensorBoard runs.

Usage examples:
  python scripts/compare_simplehand_runs.py --pair 091301cube 091302cube
  python scripts/compare_simplehand_runs.py --all-recent
"""

import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path('/opt')
BASELINE_ROOT = ROOT / 'hora_simplehand'
TACTILE_ROOT = ROOT / 'hora_simplehand_force'
OUTPUT_DIR = None
SMOOTHING_WINDOW = 25


def load_scalars(run_dir):
    event_files = sorted(run_dir.glob('events.out.tfevents.*'))
    if len(event_files) != 1:
        raise RuntimeError(f'expected one event file in {run_dir}, found {event_files}')
    accumulator = EventAccumulator(str(event_files[0]), size_guidance={'scalars': 0})
    accumulator.Reload()
    return {
        tag: np.asarray([(item.step, item.value) for item in accumulator.Scalars(tag)])
        for tag in accumulator.Tags()['scalars']
    }


def smooth(values, window=SMOOTHING_WINDOW):
    if len(values) < window:
        return values
    kernel = np.full(window, 1.0 / window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def plot_tags(data, tags, title, ylabel, output_name, y_zero_line=False):
    figure, axes = plt.subplots(len(tags), 1, figsize=(10, 3.0 * len(tags)), sharex=True)
    axes = np.atleast_1d(axes)
    colors = {'Proprioception only': '#C44E52', 'Proprioception + tactile': '#4C72B0'}
    for axis, tag in zip(axes, tags):
        for label, scalars in data.items():
            values = scalars[tag]
            steps_million = values[:, 0] / 1e6
            axis.plot(
                steps_million,
                values[:, 1],
                color=colors[label],
                alpha=0.18,
                linewidth=0.8,
            )
            axis.plot(
                steps_million,
                smooth(values[:, 1]),
                color=colors[label],
                label=label,
                linewidth=2.0,
            )
        if y_zero_line:
            axis.axhline(0.0, color='#555555', linewidth=0.8)
        axis.set_title(tag)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(loc='best')
    axes[-1].set_xlabel('Agent steps (millions)')
    figure.suptitle(title, y=1.01)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / output_name, dpi=180, bbox_inches='tight')
    plt.close(figure)


def final_window_stats(series, fraction=0.1):
    values = series[:, 1]
    count = max(1, int(np.ceil(len(values) * fraction)))
    tail = values[-count:]
    return {
        'last': float(values[-1]),
        'tail_mean': float(np.mean(tail)),
        'tail_std': float(np.std(tail)),
        'tail_min': float(np.min(tail)),
        'tail_max': float(np.max(tail)),
        'max': float(np.max(values)),
        'final_step_million': float(series[-1, 0] / 1e6),
        'points': int(len(values)),
    }


def write_summary(data, baseline_id, tactile_id):
    tags = [
        'episode_rewards/step',
        'episode_lengths/step',
        'rotation_reward',
        'object_linvel_penalty',
        'pose_diff_penalty',
        'work_done',
        'torques',
        'performance/EnvStepFPS',
        'performance/RLTrainFPS',
    ]
    rows = []
    for label, scalars in data.items():
        for tag in tags:
            row = {'run': label, 'metric': tag}
            row.update(final_window_stats(scalars[tag]))
            rows.append(row)

    fieldnames = list(rows[0])
    with (OUTPUT_DIR / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (OUTPUT_DIR / 'summary.md').open('w') as handle:
        handle.write('# SimpleHand Run Comparison\n\n')
        handle.write(f'Runs: baseline `{baseline_id}` versus tactile `{tactile_id}`. '\
                     'Tail statistics use the final 10% of logged points.\n\n')
        handle.write('| Metric | Proprioception only tail mean | Proprioception + tactile tail mean | Difference (tactile - baseline) |\n')
        handle.write('| --- | ---: | ---: | ---: |\n')
        for tag in tags:
            baseline = final_window_stats(data['Proprioception only'][tag])['tail_mean']
            tactile = final_window_stats(data['Proprioception + tactile'][tag])['tail_mean']
            handle.write(f'| {tag} | {baseline:.4f} | {tactile:.4f} | {tactile - baseline:+.4f} |\n')


def compare_pair(baseline_id, tactile_id):
    global OUTPUT_DIR
    runs = {
        'Proprioception only': BASELINE_ROOT / 'outputs' / 'Hand2Hora' / baseline_id / 'stage1_tb',
        'Proprioception + tactile': TACTILE_ROOT / 'outputs' / 'Hand2Hora' / tactile_id / 'stage1_tb',
    }
    OUTPUT_DIR = TACTILE_ROOT / 'outputs' / f'comparison_{baseline_id}_vs_{tactile_id}'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {label: load_scalars(path) for label, path in runs.items()}
    required = {
        'episode_rewards/step',
        'episode_lengths/step',
        'rotation_reward',
        'object_linvel_penalty',
        'pose_diff_penalty',
        'work_done',
        'torques',
        'performance/EnvStepFPS',
        'performance/RLTrainFPS',
    }
    missing = {
        label: sorted(required - set(scalars)) for label, scalars in data.items()
    }
    if any(missing.values()):
        raise RuntimeError(f'missing TensorBoard tags: {missing}')

    plot_tags(
        data,
        ['episode_rewards/step'],
        'Training Episode Return',
        'Return',
        'episode_reward.png',
    )
    plot_tags(
        data,
        ['episode_lengths/step'],
        'Training Episode Length',
        'Steps per episode',
        'episode_length.png',
    )
    plot_tags(
        data,
        ['rotation_reward', 'object_linvel_penalty', 'pose_diff_penalty', 'work_done', 'torques'],
        'Reward Diagnostics',
        'Logged mean',
        'reward_diagnostics.png',
        y_zero_line=True,
    )
    plot_tags(
        data,
        ['performance/EnvStepFPS', 'performance/RLTrainFPS'],
        'Training Throughput',
        'Frames per second',
        'throughput.png',
    )
    write_summary(data, baseline_id, tactile_id)
    print(f'Wrote comparison to {OUTPUT_DIR}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', nargs=2, metavar=('BASELINE', 'TACTILE'))
    parser.add_argument('--all-recent', action='store_true')
    args = parser.parse_args()
    pairs = [('081301', '081302')]
    if args.pair:
        pairs = [tuple(args.pair)]
    elif args.all_recent:
        pairs = [('091301cube', '091302cube'),
                 ('091401cylinder', '091402cylinder'),
                 ('091401cube_seed43', '091402cube_seed43')]
    for baseline_id, tactile_id in pairs:
        compare_pair(baseline_id, tactile_id)


if __name__ == '__main__':
    main()
