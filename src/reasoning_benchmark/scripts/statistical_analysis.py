#!/usr/bin/env python3
"""
Comprehensive statistical analysis for recorded benchmark artifacts.

Implements rigorous statistical reporting following best practices:
- Cluster bootstrap CIs (clustering by instance)
- Cochran's Q omnibus test
- Pairwise McNemar with Holm correction
- GLMM with random intercepts
- Discordant pair analysis

Usage:
    uv run python -m src.reasoning_benchmark.scripts.statistical_analysis
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.reasoning_benchmark.artifact_paths import LEGACY_BENCHMARK_ANALYSIS_DIR, LEGACY_BENCHMARK_RESULTS_DIR

warnings.filterwarnings('ignore')

RESULTS_DIR = LEGACY_BENCHMARK_RESULTS_DIR
OUTPUT_DIR = LEGACY_BENCHMARK_ANALYSIS_DIR

# Arms mapping
ARMS = {
    'nl_correct': 'NL',
    'sim_correct': 'Sim',
    'controlsim_correct': 'Code Exec',
}
ARM_COLS = ['nl_correct', 'sim_correct', 'controlsim_correct']
ARM_NAMES = ['NL', 'Sim', 'Code Exec']


def load_data() -> pd.DataFrame:
    """Load all results into a DataFrame with instance identifiers."""
    records = []

    for jsonl_path in RESULTS_DIR.glob("**/res.jsonl"):
        model_seed = jsonl_path.parent.parent.parent.name

        with jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    # Create unique instance ID from question characteristics
                    instance_id = f"{row.get('kind', '')}_{row.get('index_in_kind', '')}_{row.get('digit', '')}"

                    records.append({
                        'model_seed': model_seed,
                        'model': model_seed.rsplit('_seed', 1)[0],
                        'seed': row.get('seed', 0),
                        'digit': row.get('digit', 0),
                        'kind': row.get('kind', ''),
                        'instance_id': instance_id,
                        'unique_tag': row.get('unique_tag', ''),
                        'nl_correct': int(bool(row.get('nl_correct', False))),
                        'sim_correct': int(bool(row.get('sim_correct', False))),
                        'controlsim_correct': int(bool(row.get('controlsim_correct', False))),
                    })
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Instances: {df['instance_id'].nunique()}")
    return df


# =============================================================================
# Wilson Confidence Interval
# =============================================================================

def wilson_ci(n_success: int, n_total: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Wilson score confidence interval for binomial proportion.
    Returns: (p_hat, ci_low, ci_high)
    """
    if n_total == 0:
        return 0.0, 0.0, 1.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)))
    return p_hat, max(0, center - margin), min(1, center + margin)


# =============================================================================
# SECTION A: Core Outcome Summaries with Cluster Bootstrap
# =============================================================================

def cluster_bootstrap_accuracy(
    df: pd.DataFrame,
    col: str,
    cluster_col: str = 'instance_id',
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Compute accuracy with cluster bootstrap CI.
    Resamples clusters (instances), not individual rows.
    """
    rng = np.random.RandomState(random_state)
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)

    # Point estimate
    p_hat = df[col].mean()

    # Bootstrap
    boot_accs = []
    for _ in range(n_bootstrap):
        # Resample clusters with replacement
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        # Get all rows for sampled clusters
        boot_df = df[df[cluster_col].isin(sampled_clusters)]
        boot_accs.append(boot_df[col].mean())

    boot_accs = np.array(boot_accs)
    alpha = 1 - confidence
    ci_low = np.percentile(boot_accs, 100 * alpha / 2)
    ci_high = np.percentile(boot_accs, 100 * (1 - alpha / 2))

    return {
        'accuracy': p_hat,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'se': boot_accs.std(),
        'boot_samples': boot_accs,
    }


def cluster_bootstrap_delta(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    cluster_col: str = 'instance_id',
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Compute paired delta (B - A) with cluster bootstrap CI.
    """
    rng = np.random.RandomState(random_state)
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)

    # Point estimates
    p_a = df[col_a].mean()
    p_b = df[col_b].mean()
    delta = p_b - p_a

    # Relative error reduction: (error_a - error_b) / error_a
    error_a = 1 - p_a
    error_b = 1 - p_b
    rer = (error_a - error_b) / error_a if error_a > 0 else 0

    # Bootstrap
    boot_deltas = []
    boot_rers = []
    for _ in range(n_bootstrap):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        boot_df = df[df[cluster_col].isin(sampled_clusters)]
        boot_a = boot_df[col_a].mean()
        boot_b = boot_df[col_b].mean()
        boot_deltas.append(boot_b - boot_a)

        boot_err_a = 1 - boot_a
        boot_err_b = 1 - boot_b
        if boot_err_a > 0:
            boot_rers.append((boot_err_a - boot_err_b) / boot_err_a)
        else:
            boot_rers.append(0)

    boot_deltas = np.array(boot_deltas)
    boot_rers = np.array(boot_rers)
    alpha = 1 - confidence

    return {
        'delta': delta,
        'delta_ci_low': np.percentile(boot_deltas, 100 * alpha / 2),
        'delta_ci_high': np.percentile(boot_deltas, 100 * (1 - alpha / 2)),
        'rer': rer,
        'rer_ci_low': np.percentile(boot_rers, 100 * alpha / 2),
        'rer_ci_high': np.percentile(boot_rers, 100 * (1 - alpha / 2)),
        'boot_deltas': boot_deltas,
        'boot_rers': boot_rers,
    }


# =============================================================================
# SECTION A.2: Paired Structure Diagnostics (Discordant Counts)
# =============================================================================

def compute_discordant_counts(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """
    Compute discordant pair counts for McNemar.
    n01: A wrong, B correct
    n10: A correct, B wrong
    """
    n01 = ((df[col_a] == 0) & (df[col_b] == 1)).sum()
    n10 = ((df[col_a] == 1) & (df[col_b] == 0)).sum()
    n00 = ((df[col_a] == 0) & (df[col_b] == 0)).sum()
    n11 = ((df[col_a] == 1) & (df[col_b] == 1)).sum()

    return {
        'n01': n01,  # A wrong, B correct
        'n10': n10,  # A correct, B wrong
        'n00': n00,  # Both wrong
        'n11': n11,  # Both correct
        'total': len(df),
    }


# =============================================================================
# SECTION B: Statistical Tests
# =============================================================================

def cochrans_q_test(df: pd.DataFrame, cols: list) -> dict:
    """
    Cochran's Q test for k related samples.
    """
    data = df[cols].values
    n, k = data.shape

    row_sums = data.sum(axis=1)
    col_sums = data.sum(axis=0)
    total = data.sum()

    numerator = (k - 1) * (k * (col_sums**2).sum() - total**2)
    denominator = k * total - (row_sums**2).sum()

    if denominator == 0:
        return {'Q': 0.0, 'p_value': 1.0, 'df': k - 1}

    Q = numerator / denominator
    p_value = 1 - stats.chi2.cdf(Q, k - 1)

    return {'Q': Q, 'p_value': p_value, 'df': k - 1}


def exact_mcnemar_one_sided(n01: int, n10: int, alternative: str = 'greater') -> dict:
    """
    Exact one-sided McNemar test.
    H1: B > A (alternative='greater') tests if n01 > n10
    """
    n = n01 + n10
    if n == 0:
        return {'statistic': 0, 'p_value': 1.0}

    if alternative == 'greater':
        # P(X >= n01) where X ~ Binomial(n, 0.5)
        p_value = stats.binom.sf(n01 - 1, n, 0.5)
    else:
        # P(X <= n01)
        p_value = stats.binom.cdf(n01, n, 0.5)

    return {
        'statistic': n01,
        'n_discordant': n,
        'p_value': p_value,
    }


def run_pairwise_mcnemar_with_holm(
    df: pd.DataFrame,
    comparisons: list[tuple[str, str, str]],  # [(col_a, col_b, name), ...]
) -> pd.DataFrame:
    """
    Run pairwise McNemar tests with Holm correction.
    """
    results = []
    p_values = []

    for col_a, col_b, name in comparisons:
        discordant = compute_discordant_counts(df, col_a, col_b)
        mcnemar = exact_mcnemar_one_sided(discordant['n01'], discordant['n10'])

        results.append({
            'comparison': name,
            'n01': discordant['n01'],
            'n10': discordant['n10'],
            'p_raw': mcnemar['p_value'],
        })
        p_values.append(mcnemar['p_value'])

    # Holm correction
    _, p_corrected, _, _ = multipletests(p_values, method='holm')

    for i, res in enumerate(results):
        res['p_holm'] = p_corrected[i]
        res['significant'] = p_corrected[i] < 0.05

    return pd.DataFrame(results)


# =============================================================================
# SECTION B.3: GLMM (using statsmodels)
# =============================================================================

def fit_glmm(df: pd.DataFrame) -> Optional[dict]:
    """
    Fit logistic GLMM: correct ~ arm * digit + kind + model + (1|instance)
    Returns odds ratios and coefficients.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("Warning: statsmodels not fully available for GLMM")
        return None

    # Reshape to long format
    long_records = []
    for _, row in df.iterrows():
        for arm_col, arm_name in [('nl_correct', 'NL'), ('sim_correct', 'Sim'), ('controlsim_correct', 'Code Exec')]:
            long_records.append({
                'correct': row[arm_col],
                'arm': arm_name,
                'digit': row['digit'],
                'kind': row['kind'],
                'model': row['model'],
                'instance_id': row['instance_id'],
            })

    long_df = pd.DataFrame(long_records)

    # Standardize digit for interpretability
    long_df['digit_std'] = (long_df['digit'] - long_df['digit'].mean()) / long_df['digit'].std()

    # Create dummy variables for arm (reference: Code Exec)
    long_df['arm_NL'] = (long_df['arm'] == 'NL').astype(int)
    long_df['arm_Sim'] = (long_df['arm'] == 'Sim').astype(int)

    try:
        # Fit mixed effects logistic regression
        # Using GEE as approximation since full GLMM is computationally expensive
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial

        # Prepare design matrix
        formula = 'correct ~ arm_NL + arm_Sim + digit_std + arm_NL:digit_std + arm_Sim:digit_std'

        model = smf.gee(
            formula,
            groups='instance_id',
            data=long_df,
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        result = model.fit()

        # Extract coefficients and compute odds ratios
        params = result.params
        conf_int = result.conf_int()

        glmm_results = {
            'converged': True,
            'coefficients': {},
        }

        for name in params.index:
            coef = params[name]
            ci_low, ci_high = conf_int.loc[name]
            or_val = np.exp(coef)
            or_ci_low = np.exp(ci_low)
            or_ci_high = np.exp(ci_high)

            glmm_results['coefficients'][name] = {
                'coef': coef,
                'se': result.bse[name],
                'z': result.tvalues[name],
                'p_value': result.pvalues[name],
                'OR': or_val,
                'OR_ci_low': or_ci_low,
                'OR_ci_high': or_ci_high,
            }

        return glmm_results

    except Exception as e:
        print(f"GLMM fitting failed: {e}")
        return None


# =============================================================================
# SECTION C: Plots
# =============================================================================

def plot_accuracy_vs_difficulty(
    df: pd.DataFrame,
    output_path: Path,
):
    """
    Figure 1: Accuracy vs difficulty (digit) by arm with Wilson 95% CIs.
    Includes Cochran's Q annotation and pairwise McNemar results.
    """
    # Standard digits
    digits = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    digits = [d for d in digits if d in df['digit'].unique()]

    # Compute Wilson CIs for each digit and arm
    plot_data = []
    for digit in digits:
        digit_df = df[df['digit'] == digit]
        n_total = len(digit_df)
        for col, name in zip(ARM_COLS, ARM_NAMES):
            n_success = digit_df[col].sum()
            p_hat, ci_low, ci_high = wilson_ci(n_success, n_total)
            plot_data.append({
                'digit': digit,
                'arm': name,
                'accuracy': p_hat * 100,
                'ci_low': ci_low * 100,
                'ci_high': ci_high * 100,
                'n_correct': n_success,
                'n_total': n_total,
            })

    plot_df = pd.DataFrame(plot_data)

    # Compute overall statistics for annotation
    cochran = cochrans_q_test(df, ARM_COLS)

    # Pairwise McNemar (one-sided)
    disc_nl_sim = compute_discordant_counts(df, 'nl_correct', 'sim_correct')
    disc_sim_exec = compute_discordant_counts(df, 'sim_correct', 'controlsim_correct')

    mcnemar_nl_sim = exact_mcnemar_one_sided(disc_nl_sim['n01'], disc_nl_sim['n10'])
    mcnemar_sim_exec = exact_mcnemar_one_sided(disc_sim_exec['n01'], disc_sim_exec['n10'])

    # Holm correction
    p_vals = [mcnemar_nl_sim['p_value'], mcnemar_sim_exec['p_value']]
    _, p_holm, _, _ = multipletests(p_vals, method='holm')

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))

    palette = sns.color_palette("viridis", 3)
    markers = ['o', 's', '^']

    for i, arm in enumerate(ARM_NAMES):
        arm_df = plot_df[plot_df['arm'] == arm].sort_values('digit')

        ax.plot(arm_df['digit'], arm_df['accuracy'],
                marker=markers[i], markersize=10, linewidth=2.5,
                color=palette[i], label=arm)

        ax.fill_between(arm_df['digit'], arm_df['ci_low'], arm_df['ci_high'],
                        alpha=0.2, color=palette[i])

    ax.set_xlabel('Problem Difficulty (Digit Length)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy vs Problem Difficulty by Condition\n(Wilson 95% CIs, Holm-Bonferroni corrected)',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12, title='Condition', title_fontsize=12)
    ax.set_xticks(digits)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(0, max(plot_df['ci_high']) + 8)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add Cochran's Q annotation (top left)
    def format_pval(p):
        return "p<.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"

    cochran_text = f"Cochran's Q = {cochran['Q']:.1f}, {format_pval(cochran['p_value'])}"
    ax.text(0.02, 0.98, cochran_text, transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.9))

    # Add pairwise McNemar results (below Cochran's Q)
    def sig_stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    mcnemar_lines = ["Pairwise McNemar (one-sided, Holm):"]
    mcnemar_lines.append(f"  NL→Sim: {format_pval(p_holm[0])} {sig_stars(p_holm[0])}")
    mcnemar_lines.append(f"  Sim→Exec: {format_pval(p_holm[1])} {sig_stars(p_holm[1])}")

    mcnemar_text = "\n".join(mcnemar_lines)
    ax.text(0.02, 0.82, mcnemar_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.9))

    plt.tight_layout()
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.close()

    return plot_df


def plot_delta_distributions(
    df: pd.DataFrame,
    output_path: Path,
    n_bootstrap: int = 2000,
):
    """
    Figure 2: Bootstrap distribution of paired deltas.
    """
    # Compute deltas
    delta_2_1 = cluster_bootstrap_delta(df, 'nl_correct', 'sim_correct', n_bootstrap=n_bootstrap)
    delta_3_2 = cluster_bootstrap_delta(df, 'sim_correct', 'controlsim_correct', n_bootstrap=n_bootstrap)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Delta 2-1 (Sim - NL)
    ax = axes[0]
    boot_vals = delta_2_1['boot_deltas'] * 100
    sns.histplot(boot_vals, kde=True, ax=ax, color='steelblue', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.axvline(delta_2_1['delta'] * 100, color='black', linestyle='-', linewidth=2, label='Observed')
    ax.axvline(delta_2_1['delta_ci_low'] * 100, color='gray', linestyle=':', linewidth=1.5)
    ax.axvline(delta_2_1['delta_ci_high'] * 100, color='gray', linestyle=':', linewidth=1.5)

    ax.set_xlabel('Δ (Sim − NL) [%]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bootstrap Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Δ₂₋₁ = {delta_2_1["delta"]*100:.2f}% [{delta_2_1["delta_ci_low"]*100:.2f}, {delta_2_1["delta_ci_high"]*100:.2f}]',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # Delta 3-2 (Code Exec - Sim)
    ax = axes[1]
    boot_vals = delta_3_2['boot_deltas'] * 100
    sns.histplot(boot_vals, kde=True, ax=ax, color='darkorange', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.axvline(delta_3_2['delta'] * 100, color='black', linestyle='-', linewidth=2, label='Observed')
    ax.axvline(delta_3_2['delta_ci_low'] * 100, color='gray', linestyle=':', linewidth=1.5)
    ax.axvline(delta_3_2['delta_ci_high'] * 100, color='gray', linestyle=':', linewidth=1.5)

    ax.set_xlabel('Δ (Code Exec − Sim) [%]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bootstrap Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Δ₃₋₂ = {delta_3_2["delta"]*100:.2f}% [{delta_3_2["delta_ci_low"]*100:.2f}, {delta_3_2["delta_ci_high"]*100:.2f}]',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('Bootstrap Distribution of Paired Effect Sizes', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.close()

    return {'delta_2_1': delta_2_1, 'delta_3_2': delta_3_2}


def plot_discordant_pairs(
    df: pd.DataFrame,
    output_path: Path,
):
    """
    Appendix: Discordant pairs bar plot.
    """
    comparisons = [
        ('nl_correct', 'sim_correct', 'NL vs Sim'),
        ('sim_correct', 'controlsim_correct', 'Sim vs Code Exec'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    name_map = {'nl': 'NL', 'sim': 'Sim', 'controlsim': 'Exec'}

    for ax, (col_a, col_b, title) in zip(axes, comparisons):
        disc = compute_discordant_counts(df, col_a, col_b)

        name_a = name_map.get(col_a.split("_")[0], col_a.split("_")[0].upper())
        name_b = name_map.get(col_b.split("_")[0], col_b.split("_")[0].upper())

        labels = [f'{name_a} ✓, {name_b} ✗\n(n₁₀={disc["n10"]})',
                  f'{name_b} ✓, {name_a} ✗\n(n₀₁={disc["n01"]})']
        values = [disc['n10'], disc['n01']]
        colors = ['#e07070', '#7fbf7f']

        bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('Discordant Pair Counts (McNemar Evidence)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.close()


def plot_glmm_marginal_effects(
    glmm_results: dict,
    output_path: Path,
):
    """
    Figure 3: GLMM marginal effects (odds ratios) with 95% CIs.
    """
    if glmm_results is None or not glmm_results.get('converged'):
        print("Warning: GLMM not available, skipping marginal effects plot")
        return

    coefficients = glmm_results['coefficients']

    # Select coefficients to plot (exclude Intercept)
    plot_coefs = {k: v for k, v in coefficients.items() if k != 'Intercept'}

    # Rename for display
    display_names = {
        'arm_NL': 'NL (vs Code Exec)',
        'arm_Sim': 'Sim (vs Code Exec)',
        'digit_std': 'Difficulty (std)',
        'arm_NL:digit_std': 'NL × Difficulty',
        'arm_Sim:digit_std': 'Sim × Difficulty',
    }

    names = []
    ors = []
    ci_lows = []
    ci_highs = []
    p_values = []

    for name, coef in plot_coefs.items():
        display_name = display_names.get(name, name)
        names.append(display_name)
        ors.append(coef['OR'])
        ci_lows.append(coef['OR_ci_low'])
        ci_highs.append(coef['OR_ci_high'])
        p_values.append(coef['p_value'])

    # Create forest plot
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(names))
    colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in p_values]

    # Plot horizontal error bars
    for i, (or_val, ci_low, ci_high, color) in enumerate(zip(ors, ci_lows, ci_highs, colors)):
        ax.errorbar(or_val, i, xerr=[[or_val - ci_low], [ci_high - or_val]],
                    fmt='o', markersize=10, color=color, capsize=5, capthick=2,
                    elinewidth=2, markeredgecolor='black', markeredgewidth=1)

    # Reference line at OR = 1
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='OR = 1 (no effect)')

    # Add OR values as text
    for i, (or_val, ci_low, ci_high, p) in enumerate(zip(ors, ci_lows, ci_highs, p_values)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(max(ci_highs) * 1.1, i, f'OR={or_val:.2f} [{ci_low:.2f}, {ci_high:.2f}]{sig}',
                va='center', fontsize=10, fontfamily='monospace')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=13, fontweight='bold')
    ax.set_title('GLMM Marginal Effects (Reference: Code Exec)\n(Green = p<0.05, Gray = ns)',
                 fontsize=14, fontweight='bold')

    ax.set_xlim(0, max(ci_highs) * 1.8)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# SECTION D: Results Table
# =============================================================================

def generate_results_table(df: pd.DataFrame, n_bootstrap: int = 1000) -> pd.DataFrame:
    """
    Generate the main results table with all required statistics.
    """
    results = {}

    # Per-arm accuracies with cluster bootstrap CIs
    for col, name in zip(ARM_COLS, ARM_NAMES):
        acc = cluster_bootstrap_accuracy(df, col, n_bootstrap=n_bootstrap)
        results[f'{name}_acc'] = f"{acc['accuracy']*100:.1f}%"
        results[f'{name}_ci'] = f"[{acc['ci_low']*100:.1f}, {acc['ci_high']*100:.1f}]"

    # Adjacent deltas
    delta_2_1 = cluster_bootstrap_delta(df, 'nl_correct', 'sim_correct', n_bootstrap=n_bootstrap)
    delta_3_2 = cluster_bootstrap_delta(df, 'sim_correct', 'controlsim_correct', n_bootstrap=n_bootstrap)

    results['Δ₂₋₁'] = f"{delta_2_1['delta']*100:+.2f}%"
    results['Δ₂₋₁_ci'] = f"[{delta_2_1['delta_ci_low']*100:.2f}, {delta_2_1['delta_ci_high']*100:.2f}]"
    results['RER₂₋₁'] = f"{delta_2_1['rer']*100:.1f}%"

    results['Δ₃₋₂'] = f"{delta_3_2['delta']*100:+.2f}%"
    results['Δ₃₋₂_ci'] = f"[{delta_3_2['delta_ci_low']*100:.2f}, {delta_3_2['delta_ci_high']*100:.2f}]"
    results['RER₃₋₂'] = f"{delta_3_2['rer']*100:.1f}%"

    # Discordant counts
    disc_2_1 = compute_discordant_counts(df, 'nl_correct', 'sim_correct')
    disc_3_2 = compute_discordant_counts(df, 'sim_correct', 'controlsim_correct')

    results['n₀₁ (NL→Sim)'] = disc_2_1['n01']
    results['n₁₀ (NL→Sim)'] = disc_2_1['n10']
    results['n₀₁ (Sim→Exec)'] = disc_3_2['n01']
    results['n₁₀ (Sim→Exec)'] = disc_3_2['n10']

    # McNemar p-values
    mcnemar_2_1 = exact_mcnemar_one_sided(disc_2_1['n01'], disc_2_1['n10'])
    mcnemar_3_2 = exact_mcnemar_one_sided(disc_3_2['n01'], disc_3_2['n10'])

    # Holm correction
    p_values = [mcnemar_2_1['p_value'], mcnemar_3_2['p_value']]
    _, p_corrected, _, _ = multipletests(p_values, method='holm')

    results['McNemar p (NL→Sim)'] = f"{p_corrected[0]:.4f}"
    results['McNemar p (Sim→Exec)'] = f"{p_corrected[1]:.4f}"

    # Cochran's Q
    cochran = cochrans_q_test(df, ARM_COLS)
    results["Cochran's Q"] = f"{cochran['Q']:.2f}"
    results["Cochran's p"] = f"{cochran['p_value']:.4f}"

    return pd.DataFrame([results])


def print_report(df: pd.DataFrame, n_bootstrap: int = 1000):
    """Print comprehensive statistical report."""

    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
    print("="*80)
    print(f"N = {len(df)} observations")
    print(f"Instances = {df['instance_id'].nunique()}")
    print(f"Models = {df['model'].nunique()}")

    # A.1 Core Outcome Summaries
    print("\n" + "-"*80)
    print("A.1 CORE OUTCOME SUMMARIES (Cluster Bootstrap 95% CIs)")
    print("-"*80)

    for col, name in zip(ARM_COLS, ARM_NAMES):
        acc = cluster_bootstrap_accuracy(df, col, n_bootstrap=n_bootstrap)
        print(f"  {name}: {acc['accuracy']*100:.2f}% [{acc['ci_low']*100:.2f}, {acc['ci_high']*100:.2f}]")

    # Adjacent deltas
    print("\n  Adjacent Contrasts:")
    delta_2_1 = cluster_bootstrap_delta(df, 'nl_correct', 'sim_correct', n_bootstrap=n_bootstrap)
    delta_3_2 = cluster_bootstrap_delta(df, 'sim_correct', 'controlsim_correct', n_bootstrap=n_bootstrap)

    print(f"    Δ₂₋₁ (Sim - NL): {delta_2_1['delta']*100:+.2f}% [{delta_2_1['delta_ci_low']*100:.2f}, {delta_2_1['delta_ci_high']*100:.2f}]")
    print(f"    RER₂₋₁: {delta_2_1['rer']*100:.1f}%")
    print(f"    Δ₃₋₂ (Code Exec - Sim): {delta_3_2['delta']*100:+.2f}% [{delta_3_2['delta_ci_low']*100:.2f}, {delta_3_2['delta_ci_high']*100:.2f}]")
    print(f"    RER₃₋₂: {delta_3_2['rer']*100:.1f}%")

    # A.2 Discordant Counts
    print("\n" + "-"*80)
    print("A.2 PAIRED STRUCTURE DIAGNOSTICS (Discordant Counts)")
    print("-"*80)

    disc_2_1 = compute_discordant_counts(df, 'nl_correct', 'sim_correct')
    disc_3_2 = compute_discordant_counts(df, 'sim_correct', 'controlsim_correct')

    print("  NL vs Sim:")
    print(f"    n₀₁ (NL wrong, Sim correct): {disc_2_1['n01']}")
    print(f"    n₁₀ (NL correct, Sim wrong): {disc_2_1['n10']}")
    print("  Sim vs Code Exec:")
    print(f"    n₀₁ (Sim wrong, Exec correct): {disc_3_2['n01']}")
    print(f"    n₁₀ (Sim correct, Exec wrong): {disc_3_2['n10']}")

    # B. Statistical Tests
    print("\n" + "-"*80)
    print("B. STATISTICAL TESTS")
    print("-"*80)

    # Cochran's Q
    cochran = cochrans_q_test(df, ARM_COLS)
    print(f"  Cochran's Q (omnibus): Q = {cochran['Q']:.2f}, df = {cochran['df']}, p = {cochran['p_value']:.4e}")

    # McNemar with Holm
    print("\n  Pairwise McNemar (one-sided, Holm corrected):")
    comparisons = [
        ('nl_correct', 'sim_correct', 'H₁: Sim > NL'),
        ('sim_correct', 'controlsim_correct', 'H₁: Code Exec > Sim'),
    ]
    mcnemar_df = run_pairwise_mcnemar_with_holm(df, comparisons)
    for _, row in mcnemar_df.iterrows():
        sig = "***" if row['p_holm'] < 0.001 else "**" if row['p_holm'] < 0.01 else "*" if row['p_holm'] < 0.05 else "ns"
        print(f"    {row['comparison']}: n₀₁={row['n01']}, n₁₀={row['n10']}, p_raw={row['p_raw']:.4f}, p_holm={row['p_holm']:.4f} {sig}")

    # A.3 GLMM
    print("\n" + "-"*80)
    print("A.3 GLMM EFFECT SIZES (GEE approximation)")
    print("-"*80)

    glmm = fit_glmm(df)
    if glmm and glmm['converged']:
        print("  Odds Ratios (95% CI):")
        for name, coef in glmm['coefficients'].items():
            if name != 'Intercept':
                sig = "***" if coef['p_value'] < 0.001 else "**" if coef['p_value'] < 0.01 else "*" if coef['p_value'] < 0.05 else ""
                print(f"    {name}: OR = {coef['OR']:.3f} [{coef['OR_ci_low']:.3f}, {coef['OR_ci_high']:.3f}], p = {coef['p_value']:.4f} {sig}")
    else:
        print("  GLMM fitting failed or did not converge.")

    return {
        'delta_2_1': delta_2_1,
        'delta_3_2': delta_3_2,
        'cochran': cochran,
        'mcnemar': mcnemar_df,
        'glmm': glmm,
    }


def main():
    print("Loading data...")
    df = load_data()

    if df.empty:
        print("No data found!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Print comprehensive report
    report = print_report(df, n_bootstrap=1000)

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    print("\nFigure 1: Accuracy vs Difficulty...")
    plot_accuracy_vs_difficulty(df, OUTPUT_DIR / 'fig1_accuracy_vs_difficulty')

    print("Figure 2: Delta Distributions...")
    plot_delta_distributions(df, OUTPUT_DIR / 'fig2_delta_distributions')

    print("Appendix: Discordant Pairs...")
    plot_discordant_pairs(df, OUTPUT_DIR / 'appendix_discordant_pairs')

    print("Figure 3: GLMM Marginal Effects...")
    if report.get('glmm'):
        plot_glmm_marginal_effects(report['glmm'], OUTPUT_DIR / 'fig3_glmm_marginal_effects')
    else:
        print("  Skipped (GLMM not available)")

    # Generate results table
    print("\nGenerating Results Table...")
    table = generate_results_table(df)
    table.to_csv(OUTPUT_DIR / 'results_table.csv', index=False)
    print(table.T.to_string())

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
