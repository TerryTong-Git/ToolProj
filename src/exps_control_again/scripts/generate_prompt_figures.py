#!/usr/bin/env python3
"""
Generate publication-quality figures for translator and judge prompts.

Usage:
    uv run python src/exps_control_again/scripts/generate_prompt_figures.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def create_translator_figure():
    """Create a figure showing the translator prompt structure."""

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Main box
    box = mpatches.FancyBboxPatch(
        (0.15, 0.15), 9.7, 9.7,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#f8f8f8',
        edgecolor='#333333',
        linewidth=1.5
    )
    ax.add_patch(box)

    # Title
    ax.text(5, 9.6, 'Translator Prompt', fontsize=16, fontweight='bold',
            ha='center', va='top', fontfamily='serif')

    # Dashed line under title
    ax.axhline(y=9.0, xmin=0.03, xmax=0.97, color='#666', linewidth=0.8, linestyle='--')

    # System instruction
    system_text = """You are given code that solves an algorithmic
problem. Reason through the problem step-by-step
using natural language and arrive at the answer.
Do NOT describe or translate the code mechanically."""

    ax.text(0.5, 8.7, system_text, fontsize=16, va='top', fontfamily='monospace',
            linespacing=1.25)

    # Dashed line before guidelines
    ax.axhline(y=6.6, xmin=0.03, xmax=0.97, color='#666', linewidth=0.8, linestyle='--')

    # Guidelines section
    ax.text(0.5, 6.3, 'GUIDELINES', fontsize=16, fontweight='bold', va='top',
            fontfamily='serif')

    guidelines = """• Think like a human (exploratory reasoning)
• Be conversational ("Let me check...", "I notice...")
• Skip obvious steps, focus on insights (WHY)
• Use natural structure (paragraphs over lists)"""

    ax.text(0.5, 5.8, guidelines, fontsize=16, va='top', fontfamily='monospace',
            linespacing=1.25)

    # Dashed line before examples
    ax.axhline(y=4.2, xmin=0.03, xmax=0.97, color='#666', linewidth=0.8, linestyle='--')

    # Examples section
    ax.text(0.5, 3.9, '10 IN-CONTEXT EXAMPLES', fontsize=16,
            fontweight='bold', va='top', fontfamily='serif')

    example_text = """Example: Topological Sort
  Input:  Adjacency matrix A = [[0,1,0,...],...]
  Output: "Looking at the matrix, node 3 has
          in-degree 0... The answer is 3."
(+ 9 more: KMP, Bridges, LCS, Bellman-Ford, ...)"""

    ax.text(0.5, 3.4, example_text, fontsize=11, va='top', fontfamily='monospace',
            linespacing=1.25, color='#444')

    # Dashed line before input
    ax.axhline(y=1.7, xmin=0.03, xmax=0.97, color='#666', linewidth=0.8, linestyle='--')

    # Input section
    ax.text(0.5, 1.4, 'TEST INPUT', fontsize=16, fontweight='bold', va='top',
            fontfamily='serif')

    input_text = """def solution():
    # [Code to translate]"""

    ax.text(0.5, 0.9, input_text, fontsize=16, va='top', fontfamily='monospace',
            color='#0066cc', linespacing=1.25)

    plt.tight_layout()
    return fig


def create_judge_figure():
    """Create a figure showing the judge/classifier prompt."""

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Main box - fill the space
    box = mpatches.FancyBboxPatch(
        (0.15, 0.15), 9.7, 9.7,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#f8f8f8',
        edgecolor='#333333',
        linewidth=1.5
    )
    ax.add_patch(box)

    # Title
    ax.text(5, 9.6, 'Discriminator Prompt', fontsize=16, fontweight='bold',
            ha='center', va='top', fontfamily='serif')

    # Dashed line under title
    ax.axhline(y=9.0, xmin=0.03, xmax=0.97, color='#666', linewidth=0.8, linestyle='--')

    # Task description
    task_text = """You are analyzing an explanation of how to
solve an algorithmic problem.

TASK: Determine whether this explanation was
written by someone solving the problem naturally
using language ("Native NL") or by someone
translating/simulating code execution into
natural language ("Translated")."""

    ax.text(0.5, 8.7, task_text, fontsize=16, va='top', fontfamily='monospace',
            linespacing=1.25)

    # Dashed line before input section
    ax.axhline(y=5.4, xmin=0.03, xmax=0.97, color='#666', linewidth=0.8, linestyle='--')

    # Input section
    ax.text(0.5, 5.1, 'INPUT FORMAT', fontsize=16, fontweight='bold', va='top',
            fontfamily='serif')

    input_section = """PROBLEM:
  {Algorithmic problem description}

EXPLANATION:
  {Reasoning trace to classify}"""

    ax.text(0.5, 4.6, input_section, fontsize=16, va='top', fontfamily='monospace',
            color='#0066cc', linespacing=1.25)

    # Dashed line before output section
    ax.axhline(y=2.4, xmin=0.03, xmax=0.97, color='#666', linewidth=0.8, linestyle='--')

    # Output format
    ax.text(0.5, 2.1, 'OUTPUT FORMAT', fontsize=16, fontweight='bold',
            va='top', fontfamily='serif')

    output_format = """PREDICTION: [NATIVE or TRANSLATED]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
REASONING:  [1-2 sentence justification]"""

    ax.text(0.5, 1.6, output_format, fontsize=16, va='top', fontfamily='monospace',
            linespacing=1.25)

    plt.tight_layout()
    return fig


def create_combined_figure():
    """Create a combined two-panel figure for the paper."""

    fig = plt.figure(figsize=(14, 5.6))

    # Panel (a): Translator - slightly wider
    ax1 = fig.add_axes([0.01, 0.02, 0.52, 0.94])  # [left, bottom, width, height]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    # Box - tighter fit
    box1 = mpatches.FancyBboxPatch(
        (0.1, 0.5), 9.8, 9.4,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor='#f8f8f8',
        edgecolor='#333333',
        linewidth=1.5
    )
    ax1.add_patch(box1)

    # Title
    ax1.text(5, 9.7, '(a) Translator Prompt', fontsize=20, fontweight='bold',
             ha='center', va='top', fontfamily='serif')

    # Dashed line under title
    ax1.axhline(y=9.0, xmin=0.02, xmax=0.98, color='#666', linewidth=0.8, linestyle='--')

    # System instruction
    system_text = """You are given code that solves an algorithmic
problem. Reason through the problem step-by-step
using natural language and arrive at the answer.
Do NOT translate the code mechanically."""

    ax1.text(0.3, 8.7, system_text, fontsize=14, va='top', fontfamily='monospace',
             linespacing=1.2)

    # Dashed line before guidelines
    ax1.axhline(y=6.6, xmin=0.02, xmax=0.98, color='#666', linewidth=0.8, linestyle='--')

    # Guidelines section
    ax1.text(0.3, 6.3, 'GUIDELINES', fontsize=14, fontweight='bold', va='top',
             fontfamily='serif')

    guidelines = """• Think like a human (exploratory reasoning)
• Be conversational ("Let me check...", "I notice")
• Skip obvious steps, focus on insights (WHY)
• Use natural structure (paragraphs over lists)"""

    ax1.text(0.3, 5.8, guidelines, fontsize=14, va='top', fontfamily='monospace',
             linespacing=1.2)

    # Dashed line before examples
    ax1.axhline(y=3.9, xmin=0.02, xmax=0.98, color='#666', linewidth=0.8, linestyle='--')

    # Examples section
    ax1.text(0.3, 3.6, '10 IN-CONTEXT EXAMPLES', fontsize=14,
             fontweight='bold', va='top', fontfamily='serif')

    example_text = """Example: Topological Sort
  Input:  Adjacency matrix A = [[0,1,0,...],...]
  Output: "Node 3 has in-degree 0... Answer is 3."
(+ 9 more: KMP, Bridges, LCS, Bellman-Ford, ...)"""

    ax1.text(0.3, 3.1, example_text, fontsize=12, va='top', fontfamily='monospace',
             linespacing=1.2, color='#444')

    # Dashed line before input
    ax1.axhline(y=1.6, xmin=0.02, xmax=0.98, color='#666', linewidth=0.8, linestyle='--')

    # Input section
    ax1.text(0.3, 1.3, 'TEST INPUT', fontsize=14, fontweight='bold', va='top',
             fontfamily='serif')

    input_text = """def solution():  # [Code to translate]"""

    ax1.text(0.3, 0.8, input_text, fontsize=14, va='top', fontfamily='monospace',
             color='#0066cc', linespacing=1.2)

    # Panel (b): Discriminator
    ax2 = fig.add_axes([0.54, 0.02, 0.45, 0.96])  # [left, bottom, width, height]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Box - tighter fit
    box2 = mpatches.FancyBboxPatch(
        (0.1, 0.5), 9.8, 9.4,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor='#f8f8f8',
        edgecolor='#333333',
        linewidth=1.5
    )
    ax2.add_patch(box2)

    # Title
    ax2.text(5, 9.7, '(b) Discriminator Prompt', fontsize=20, fontweight='bold',
             ha='center', va='top', fontfamily='serif')

    # Dashed line under title
    ax2.axhline(y=9.0, xmin=0.02, xmax=0.98, color='#666', linewidth=0.8, linestyle='--')

    # Task description
    task_text = """You are analyzing an explanation of how to
solve an algorithmic problem.

TASK: Determine whether this was written by
someone solving naturally ("Native NL") or
translating/simulating code ("Translated")."""

    ax2.text(0.3, 8.7, task_text, fontsize=14, va='top', fontfamily='monospace',
             linespacing=1.2)

    # Dashed line before input section
    ax2.axhline(y=5.8, xmin=0.02, xmax=0.98, color='#666', linewidth=0.8, linestyle='--')

    # Input section
    ax2.text(0.3, 5.5, 'INPUT FORMAT', fontsize=14, fontweight='bold', va='top',
             fontfamily='serif')

    input_section = """PROBLEM:
  {Algorithmic problem description}
EXPLANATION:
  {Reasoning trace to classify}"""

    ax2.text(0.3, 5.0, input_section, fontsize=14, va='top', fontfamily='monospace',
             color='#0066cc', linespacing=1.2)

    # Dashed line before output section
    ax2.axhline(y=2.9, xmin=0.02, xmax=0.98, color='#666', linewidth=0.8, linestyle='--')

    # Output format
    ax2.text(0.3, 2.6, 'OUTPUT FORMAT', fontsize=14, fontweight='bold',
             va='top', fontfamily='serif')

    output_format = """PREDICTION: [NATIVE or TRANSLATED]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
REASONING:  [1-2 sentence justification]"""

    ax2.text(0.3, 2.1, output_format, fontsize=14, va='top', fontfamily='monospace',
             linespacing=1.2)

    return fig


def main():
    output_dir = Path('src/exps_control_again/results')
    output_dir.mkdir(exist_ok=True)

    # Individual figures
    print("Generating translator prompt figure...")
    fig_translator = create_translator_figure()
    fig_translator.savefig(output_dir / 'fig_translator_prompt.pdf',
                           bbox_inches='tight', dpi=300)
    fig_translator.savefig(output_dir / 'fig_translator_prompt.png',
                           bbox_inches='tight', dpi=300)
    plt.close(fig_translator)

    print("Generating discriminator prompt figure...")
    fig_judge = create_judge_figure()
    fig_judge.savefig(output_dir / 'fig_discriminator_prompt.pdf',
                      bbox_inches='tight', dpi=300)
    fig_judge.savefig(output_dir / 'fig_discriminator_prompt.png',
                      bbox_inches='tight', dpi=300)
    plt.close(fig_judge)

    # Combined figure
    print("Generating combined figure...")
    fig_combined = create_combined_figure()
    fig_combined.savefig(output_dir / 'fig_prompts_combined.pdf',
                         bbox_inches='tight', dpi=300)
    fig_combined.savefig(output_dir / 'fig_prompts_combined.png',
                         bbox_inches='tight', dpi=300)
    plt.close(fig_combined)

    print(f"\nFigures saved to {output_dir}")


if __name__ == '__main__':
    main()
