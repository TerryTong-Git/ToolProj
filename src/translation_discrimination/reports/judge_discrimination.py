#!/usr/bin/env python3
"""
Bar plot of judge discrimination accuracy + control accuracy for 3 judges.

Usage:
    uv run python -m src.translation_discrimination.reports.judge_discrimination
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Opus 4 and Grok 4.1 Fast: named files
# Gemini 2.5 Pro: original dated file (janky naming)
JUDGE_FILES = {
    "Claude Opus 4": "opus4_judge_gpt4o_translator_n1000.json",
    "Grok 4.1 Fast": "grok41fast_judge_gpt4o_translator_n1000.json",
    "Gemini 2.5 Pro": "source_discrimination_20260117_145524.json",
}


def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, centre - margin), min(1, centre + margin)


def main():
    names = []
    disc_accs = []
    disc_ci_los = []
    disc_ci_his = []
    ctrl_accs = []
    ctrl_ci_los = []
    ctrl_ci_his = []

    for name, fname in JUDGE_FILES.items():
        with open(RESULTS_DIR / fname) as f:
            data = json.load(f)
        r = data["results"]
        ctrls = data["controls"]

        names.append(name)

        # Discrimination
        disc_accs.append(r["accuracy"] * 100)
        disc_ci_los.append(r["accuracy_ci_low"] * 100)
        disc_ci_his.append(r["accuracy_ci_high"] * 100)

        # Control
        n_ctrl = len(ctrls)
        k_ctrl = sum(c["correct"] for c in ctrls)
        ctrl_acc = k_ctrl / n_ctrl * 100
        lo, hi = wilson_ci(k_ctrl, n_ctrl)
        ctrl_accs.append(ctrl_acc)
        ctrl_ci_los.append(lo * 100)
        ctrl_ci_his.append(hi * 100)

    disc_accs = np.array(disc_accs)
    ctrl_accs = np.array(ctrl_accs)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    x = np.arange(len(names))
    w = 0.32

    # Discrimination bars
    d_err_lo = disc_accs - np.array(disc_ci_los)
    d_err_hi = np.array(disc_ci_his) - disc_accs
    bars_d = ax.bar(
        x - w / 2,
        disc_accs,
        w,
        yerr=[d_err_lo, d_err_hi],
        color="#5b9bd5",
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        error_kw={"linewidth": 1.3},
        label="Discrimination\n(Native vs Translated)",
        zorder=3,
    )

    # Control bars
    c_err_lo = ctrl_accs - np.array(ctrl_ci_los)
    c_err_hi = np.array(ctrl_ci_his) - ctrl_accs
    bars_c = ax.bar(
        x + w / 2,
        ctrl_accs,
        w,
        yerr=[c_err_lo, c_err_hi],
        color="#ed7d31",
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        error_kw={"linewidth": 1.3},
        label="Control\n(Code vs NL)",
        zorder=3,
    )

    # Chance line
    ax.axhline(50, color="red", linestyle="--", linewidth=1.5, zorder=2, label="Chance (50%)")

    # Value labels
    for i in range(len(names)):
        ax.text(
            bars_d[i].get_x() + bars_d[i].get_width() / 2,
            bars_d[i].get_height() + d_err_hi[i] + 0.8,
            f"{disc_accs[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#2a5a8a",
        )
        ax.text(
            bars_c[i].get_x() + bars_c[i].get_width() / 2,
            bars_c[i].get_height() + c_err_hi[i] + 0.8,
            f"{ctrl_accs[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#b35900",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_xlabel("Judge Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Judge Discrimination: Native NL vs GPT-4o Translated Traces\n" "~1000 pairs per judge · Wilson 95% CIs",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(35, 100)
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    out = RESULTS_DIR / "judge_discrimination_barplot"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out.with_suffix('.png')} / .pdf")


if __name__ == "__main__":
    main()
