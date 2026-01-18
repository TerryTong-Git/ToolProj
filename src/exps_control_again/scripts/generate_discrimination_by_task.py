#!/usr/bin/env python3
"""
Generate source discrimination by task bar plot.

Usage:
    uv run python src/exps_control_again/scripts/generate_discrimination_by_task.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data from JSON (using older, more balanced dataset)
results_path = Path('src/exps_control_again/results/source_discrimination_20260117_145524.json')
with results_path.open() as f:
    data = json.load(f)

by_kind = data['results']['by_kind']
overall_accuracy = data['results']['accuracy'] * 100  # For the average line

# Extract and sort by accuracy
tasks = []
for kind, kdata in by_kind.items():
    tasks.append({
        'name': kind,
        'accuracy': kdata['accuracy'] * 100,
        'ci_low': kdata['accuracy_ci'][0] * 100,
        'ci_high': kdata['accuracy_ci'][1] * 100,
        'n': kdata['total'],
    })

tasks.sort(key=lambda x: x['accuracy'])

# Format task names with newlines for readability
def format_name(name):
    # Special formatting for specific task names
    name_map = {
        'bubble_sort': 'bubble\nsort',
        'find_maximum_subarray_kadane': 'kadane',
        'insertion_sort': 'insertion\nsort',
        'minimum': 'minimum',
        'knap': 'knapsack',
        'rod': 'rod\ncutting',
        'heapsort': 'heapsort',
        'lcs_length': 'lcs',
        'strongly_connected_components': 'scc',
        'quicksort': 'quicksort',
        'dfs': 'dfs',
        'mst_prim': 'prim',
        'bfs': 'bfs',
        'floyd_warshall': 'floyd\nwarshall',
        'segments_intersect': 'segments\nintersect',
        'graham_scan': 'graham\nscan',
        'dijkstra': 'dijkstra',
        'mst_kruskal': 'kruskal',
        'matrix_chain_order': 'matrix\nchain',
        'task_scheduling': 'task\nscheduling',
        'optimal_bst': 'optimal\nbst',
    }
    return name_map.get(name, name.replace('_', '\n'))

# Prepare data
names = [format_name(t['name']) for t in tasks]
accuracies = [t['accuracy'] for t in tasks]
ci_low = [t['ci_low'] for t in tasks]
ci_high = [t['ci_high'] for t in tasks]
n_samples = [t['n'] for t in tasks]

# Calculate error bars
errors_low = [acc - low for acc, low in zip(accuracies, ci_low)]
errors_high = [high - acc for acc, high in zip(accuracies, ci_high)]

# Create figure (wider for better label spacing)
fig, ax = plt.subplots(figsize=(18, 5.5))

x = np.arange(len(names))
width = 0.7

# Color bars based on whether CI contains 50%
colors = ['#7fbf7f' if low <= 50 <= high else '#e07070'
          for low, high in zip(ci_low, ci_high)]

bars = ax.bar(x, accuracies, width, yerr=[errors_low, errors_high],
              color=colors, edgecolor='black', linewidth=1,
              capsize=3, error_kw={'linewidth': 1.5})

# Add value labels above bars (with sample size)
for i, (bar, acc, n) in enumerate(zip(bars, accuracies, n_samples)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors_high[i] + 1.5,
            f'{acc:.0f}%\n(n={n})', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add reference lines
ax.axhline(y=50, color='blue', linestyle='--', linewidth=2, label='Chance (50%)')
ax.axhline(y=overall_accuracy, color='orange', linestyle='-', linewidth=2,
           label=f'Overall ({overall_accuracy:.1f}%)')

# Customize plot
ax.set_ylabel('Judge Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Task Type', fontsize=14, fontweight='bold')
ax.set_title('Source Discrimination by Task\n(Green = CI contains 50%, Red = Distinguishable)',
             fontsize=16, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11, ha='center')

# Set y-axis limits (compressed to save space but show all data + labels)
ax.set_ylim(20, 75)

ax.legend(loc='upper left', fontsize=12)
ax.tick_params(axis='y', labelsize=12)

# Add grid for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()

# Save
output_dir = Path('src/exps_control_again/results')
fig.savefig(output_dir / 'discrimination_by_task.pdf', bbox_inches='tight', dpi=300)
fig.savefig(output_dir / 'discrimination_by_task.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Saved to {output_dir}/discrimination_by_task.pdf/png")
