"""Generate a figure showing prompts used in different arms."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.reasoning_benchmark.artifact_paths import LEGACY_BENCHMARK_FIGURES_DIR


def create_arm_prompts_figure():
    """Create a 1x4 horizontal figure showing the evaluation arm prompts."""

    # Colors for each arm (even lighter shades)
    header_colors = {
        'nl': '#90CAF9',      # Very Light Blue
        'sim': '#A5D6A7',     # Very Light Green
        'controlsim': '#CE93D8',  # Very Light Purple
        'code': '#FFCC80',    # Very Light Orange
    }

    arms = [
        {
            'name': 'Arm 1: Natural Language (NL)',
            'color_key': 'nl',
            'prompt': '''"Solve the following algorithmic problem:
{question}
YOU ARE NEVER ALLOWED TO USE CODE.
FOLLOW THE FORMAT CAREFULLY."''',
            'example': '''Q: "Compute: 123 + 456"
A: {"simulation": "Adding 123 + 456:
      units 3+6=9, tens 2+5=7,
      hundreds 1+4=5. Result: 579",
    "Answer": "579"}'''
        },
        {
            'name': 'Arm 2: Code Similarity (Sim)',
            'color_key': 'sim',
            'prompt': '''"Solve the following algorithmic problem:
{question}
FOLLOW THE FORMAT CAREFULLY."

Response includes:
  • code: Python solution() function
  • simulation: Natural language trace''',
            'example': '''Q: "Compute: 123 + 456"
A: {"code":
def solution():
    return 123 + 456
    "simulation": "Adds 123 + 456",
    "Answer": "579"}'''
        },
        {
            'name': 'Arm 2.5: Controlled Simulation (ControlSim)',
            'color_key': 'controlsim',
            'prompt': '''"Simulate execution of the provided code:
{code}
ALL NECESSARY INFORMATION IS IN THE CODE.
FOLLOW THE FORMAT CAREFULLY."

Note: {code} is solution() from Arm 2''',
            'example': '''Q: "Simulate:
def solution():
    return 123 + 456"
A: {"simulation": "The function computes
      123 + 456 and returns 579",
    "Answer": "579"}'''
        },
        {
            'name': 'Arm 3: Code Execution (Code)',
            'color_key': 'code',
            'prompt': '''Uses code from Arm 2

Process:
  1. Extract solution() from Arm 2
  2. Execute in sandboxed Python (5s)
  3. Compare output to ground truth''',
            'example': '''Code from Arm 2:
def solution():
    return 123 + 456

>>> solution()
579'''
        },
    ]

    # Create figure with 1x4 layout (compressed vertically)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    fig.patch.set_facecolor('white')
    fig.suptitle("Evaluation Arm Prompts", fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.03)

    for ax, arm in zip(axes, arms):
        header_color = header_colors[arm['color_key']]

        # White background for main content
        ax.set_facecolor('white')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Border
        for spine in ax.spines.values():
            spine.set_color(header_color)
            spine.set_linewidth(2)

        ax.set_xticks([])
        ax.set_yticks([])

        # Colored header bar
        header_rect = mpatches.FancyBboxPatch(
            (0, 0.93), 1.0, 0.07,
            boxstyle="square,pad=0",
            facecolor=header_color,
            edgecolor='none',
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(header_rect)

        # Title in header (dark text for light backgrounds)
        ax.text(0.5, 0.965, arm['name'], fontsize=11, fontweight='bold',
                color='#333333', va='center', ha='center', transform=ax.transAxes)

        # Prompt box - 50% of space
        prompt_box = mpatches.FancyBboxPatch(
            (0.02, 0.48), 0.96, 0.43,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor='#f8f8f8',
            edgecolor='#cccccc',
            linewidth=1,
            transform=ax.transAxes,
        )
        ax.add_patch(prompt_box)

        # Prompt section
        ax.text(0.04, 0.89, "Prompt:", fontsize=12, fontweight='bold',
                color='#333333', va='top', transform=ax.transAxes)
        ax.text(0.04, 0.81, arm['prompt'], fontsize=11, family='monospace',
                va='top', transform=ax.transAxes, linespacing=1.05)

        # Example box - 45% of space
        example_box = mpatches.FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.44,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor='#f8f8f8',
            edgecolor='#cccccc',
            linewidth=1,
            transform=ax.transAxes,
        )
        ax.add_patch(example_box)

        # Example section
        ax.text(0.04, 0.44, "Example:", fontsize=12, fontweight='bold',
                color='#333333', va='top', transform=ax.transAxes)
        # Render example with bold Q: and A: labels
        example_lines = arm['example'].split('\n')
        y_pos = 0.36
        for line in example_lines:
            if line.startswith('Q:'):
                ax.text(0.04, y_pos, "Q:", fontsize=11, family='monospace',
                        fontweight='bold', va='top', transform=ax.transAxes)
                ax.text(0.08, y_pos, line[2:], fontsize=11, family='monospace',
                        va='top', transform=ax.transAxes)
            elif line.startswith('A:'):
                ax.text(0.04, y_pos, "A:", fontsize=11, family='monospace',
                        fontweight='bold', va='top', transform=ax.transAxes)
                ax.text(0.08, y_pos, line[2:], fontsize=11, family='monospace',
                        va='top', transform=ax.transAxes)
            else:
                ax.text(0.04, y_pos, line, fontsize=11, family='monospace',
                        va='top', transform=ax.transAxes)
            y_pos -= 0.05

    # Save to the correct location
    output_path = LEGACY_BENCHMARK_FIGURES_DIR / "arm_prompts.png"
    plt.savefig(output_path, dpi=150,
                facecolor='white', edgecolor='none', bbox_inches='tight', pad_inches=0.02)
    print(f"[arm_prompts_figure] Saved {output_path}")
    plt.close()


def main() -> int:
    create_arm_prompts_figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
