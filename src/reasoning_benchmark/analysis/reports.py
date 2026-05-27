import itertools
import warnings
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src.reasoning_benchmark.records import create_big_df

# =============================================================================
# Statistical Functions: McNemar (one-sided) and Cochran's Q
# =============================================================================

def mcnemar_one_sided(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """
    One-sided exact McNemar test.
    H1: col_b > col_a (tests if col_b has more correct than col_a)
    Returns: dict with n01, n10, p_value
    """
    n01 = ((df[col_a] == 0) & (df[col_b] == 1)).sum()  # A wrong, B correct
    n10 = ((df[col_a] == 1) & (df[col_b] == 0)).sum()  # A correct, B wrong
    n = n01 + n10

    if n == 0:
        return {'n01': n01, 'n10': n10, 'p_value': 1.0}

    # One-sided: P(X >= n01) where X ~ Binomial(n, 0.5)
    p_value = stats.binom.sf(n01 - 1, n, 0.5)

    return {'n01': n01, 'n10': n10, 'p_value': p_value}


def cochrans_q_test(df: pd.DataFrame, cols: list) -> dict:
    """
    Cochran's Q test for k related samples (binary outcomes).
    Returns: dict with Q statistic, p_value, df
    """
    data = df[cols].values.astype(int)
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


def create_instance_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unique instance_id from kind + digit + index if not present.
    Instance = unique problem, not model/seed variation.
    """
    df = df.copy()
    if 'instance_id' not in df.columns:
        # Create instance_id from kind, digit, and row index within kind+digit
        df['instance_id'] = df.groupby(['kind', 'digit']).cumcount().astype(str)
        df['instance_id'] = df['kind'] + '_' + df['digit'].astype(str) + '_' + df['instance_id']
    return df


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    col: str,
    cluster_col: str = 'instance_id',
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Compute cluster bootstrap confidence interval.
    Resamples instances (unique problems), not individual rows.
    Task kinds are fixed design categories, instances are the sampling units.
    Returns: (ci_lower, ci_upper)
    """
    # Ensure instance_id exists
    if cluster_col == 'instance_id' and 'instance_id' not in df.columns:
        df = create_instance_id(df)

    rng = np.random.RandomState(random_state)

    # Pre-compute cluster means for efficiency
    cluster_means = df.groupby(cluster_col)[col].mean()
    clusters = cluster_means.index.values
    means_array = cluster_means.values
    n_clusters = len(clusters)

    if n_clusters == 0:
        return (np.nan, np.nan)

    # Vectorized bootstrap: sample indices and compute means
    boot_indices = rng.choice(n_clusters, size=(n_bootstrap, n_clusters), replace=True)
    boot_means = means_array[boot_indices].mean(axis=1)

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    return (lower, upper)


def cluster_bootstrap_delta(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    cluster_col: str = 'instance_id',
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Compute paired delta (B - A) with cluster bootstrap CI.
    Bootstraps on instance level (unique problems).
    Returns: dict with delta, CI bounds, and bootstrap samples.
    """
    # Ensure instance_id exists
    if cluster_col == 'instance_id' and 'instance_id' not in df.columns:
        df = create_instance_id(df)

    rng = np.random.RandomState(random_state)

    # Pre-compute cluster means for both columns
    cluster_means_a = df.groupby(cluster_col)[col_a].mean()
    cluster_means_b = df.groupby(cluster_col)[col_b].mean()

    # Align clusters
    clusters = cluster_means_a.index.values
    means_a = cluster_means_a.values
    means_b = cluster_means_b.loc[clusters].values
    n_clusters = len(clusters)

    # Point estimates
    p_a = means_a.mean()
    p_b = means_b.mean()
    delta = p_b - p_a

    if n_clusters == 0:
        return {'delta': delta, 'delta_ci_low': np.nan, 'delta_ci_high': np.nan, 'boot_deltas': np.array([])}

    # Vectorized bootstrap
    boot_indices = rng.choice(n_clusters, size=(n_bootstrap, n_clusters), replace=True)
    boot_means_a = means_a[boot_indices].mean(axis=1)
    boot_means_b = means_b[boot_indices].mean(axis=1)
    boot_deltas = boot_means_b - boot_means_a

    alpha = 1 - confidence
    return {
        'delta': delta,
        'delta_ci_low': np.percentile(boot_deltas, 100 * alpha / 2),
        'delta_ci_high': np.percentile(boot_deltas, 100 * (1 - alpha / 2)),
        'boot_deltas': boot_deltas,
    }


# =============================================================================
# GLMM Functions: Mixed Effects Logistic Regression
# =============================================================================

# Task family mapping
TASK_FAMILY_MAP = {
    # Arithmetic
    "add": "arithmetic", "sub": "arithmetic", "mul": "arithmetic",
    # Dynamic Programming
    "lcs": "dp", "rod": "dp", "knap": "dp", "lcs_length": "dp",
    "matrix_chain_order": "dp", "optimal_bst": "dp",
    # ILP
    "ilp_assign": "ilp", "ilp_prod": "ilp", "ilp_partition": "ilp",
    # Graph - Search
    "bfs": "graph_search", "dfs": "graph_search", "topological_sort": "graph_search",
    # Graph - Shortest Path
    "dijkstra": "shortest_path", "bellman_ford": "shortest_path",
    "floyd_warshall": "shortest_path", "dag_shortest_paths": "shortest_path",
    # Graph - MST
    "mst_kruskal": "mst", "mst_prim": "mst",
    # Graph - Connectivity
    "articulation_points": "connectivity", "bridges": "connectivity",
    "strongly_connected_components": "connectivity",
    # Sorting
    "bubble_sort": "sorting", "insertion_sort": "sorting",
    "quicksort": "sorting", "heapsort": "sorting",
    # Selection/Search
    "binary_search": "selection", "quickselect": "selection",
    "minimum": "selection", "find_maximum_subarray_kadane": "selection",
    # String
    "kmp_matcher": "string", "naive_string_matcher": "string",
    # Geometry
    "graham_scan": "geometry", "jarvis_march": "geometry",
    "segments_intersect": "geometry",
    # Greedy
    "activity_selector": "greedy", "task_scheduling": "greedy",
    # NP-Hard
    "edp": "np_hard", "gcp": "np_hard", "ksp": "np_hard",
    "spp": "np_hard", "tsp": "np_hard",
}


def prepare_glmm_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for GLMM: long format with arm, τ (digit), task_family, model, instance.
    """
    cols = ["nl_correct", "sim_correct", "code_correct"]
    arm_names = {"nl_correct": "NL", "sim_correct": "Sim", "code_correct": "Code"}

    # Create instance_id if not present
    if 'instance_id' not in df.columns:
        df = create_instance_id(df)

    # Add task_family
    df = df.copy()
    df['task_family'] = df['kind'].map(TASK_FAMILY_MAP).fillna('other')

    # Melt to long format
    id_vars = ['instance_id', 'digit', 'kind', 'task_family', 'model']
    long_df = pd.melt(df, id_vars=id_vars, value_vars=cols, var_name='arm_col', value_name='correct')
    long_df['arm'] = long_df['arm_col'].map(arm_names)
    long_df['tau'] = long_df['digit']  # τ = digit (difficulty)

    # Standardize τ for numerical stability
    long_df['tau_std'] = (long_df['tau'] - long_df['tau'].mean()) / long_df['tau'].std()

    # Create short model names
    long_df['model_short'] = long_df['model'].apply(lambda x: x.split('/')[-1] if '/' in x else x)

    return long_df


def fit_glmm(long_df: pd.DataFrame) -> dict:
    """
    Fit GLMM: correct ~ arm + tau + task_family + model + arm:tau + (1|instance_id)

    Fixed effects: arm, τ, task_family, model
    Interaction: arm × τ
    Random effects: approximated via cluster-robust SEs
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    long_df = long_df.copy()

    # CRITICAL: Convert correct to integer (0/1) to avoid encoding issues
    long_df['correct'] = long_df['correct'].astype(int)

    # Set reference categories
    long_df['arm'] = pd.Categorical(long_df['arm'], categories=['NL', 'Sim', 'Code'], ordered=False)
    long_df['task_family'] = pd.Categorical(long_df['task_family'])
    long_df['model_short'] = pd.Categorical(long_df['model_short'])

    # Formula: fixed effects + arm×tau interaction
    formula = "correct ~ C(arm, Treatment('NL')) + tau_std + C(task_family) + C(model_short) + C(arm, Treatment('NL')):tau_std"

    print("[fit_glmm] Fitting logistic regression with cluster-robust SEs...")
    print(f"[fit_glmm] N observations: {len(long_df)}")
    print(f"[fit_glmm] N instances (clusters): {long_df['instance_id'].nunique()}")
    print(f"[fit_glmm] Arms: {long_df['arm'].unique().tolist()}")
    print(f"[fit_glmm] Task families: {long_df['task_family'].nunique()}")
    print(f"[fit_glmm] Models: {long_df['model_short'].nunique()}")
    print(f"[fit_glmm] Overall accuracy: {long_df['correct'].mean():.3f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # GLM with cluster-robust standard errors (approximates random intercept)
        result = smf.glm(
            formula,
            data=long_df,
            family=sm.families.Binomial()
        ).fit(cov_type='cluster', cov_kwds={'groups': long_df['instance_id']})
        print("[fit_glmm] GLM with cluster-robust SEs fitted successfully")

    return {
        'model': result,
        'data': long_df,
        'tau_mean': long_df['digit'].mean(),
        'tau_std': long_df['digit'].std(),
    }


def plot_glmm_predicted_prob(glmm_result: dict, output_path: str = "figures/glmm_predicted_prob.png") -> None:
    """
    Main plot: Predicted probability vs τ (digit), by arm.
    Uses marginal predictions with model-based CIs.
    Log-scaled τ axis.
    """
    from matplotlib import rcParams

    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["font.family"] = "Arial"

    model = glmm_result['model']
    long_df = glmm_result['data']
    tau_mean = glmm_result['tau_mean']
    tau_std_val = glmm_result['tau_std']

    # Extended τ range with log scaling (2 to 64)
    tau_values = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 56, 64])
    max_obs_tau = 20
    interp_mask = tau_values <= max_obs_tau
    extrap_mask = tau_values >= max_obs_tau  # overlap at boundary for continuity

    arms = ['NL', 'Sim', 'Code']

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'NL': 'steelblue', 'Sim': 'forestgreen', 'Code': 'darkorange'}

    # Get model parameters for manual prediction
    params = model.params
    intercept = params.get('Intercept', 0)
    arm_sim = params.get("C(arm, Treatment('NL'))[T.Sim]", 0)
    arm_code = params.get("C(arm, Treatment('NL'))[T.Code]", 0)
    tau_coef = params.get("tau_std", 0)
    arm_sim_tau = params.get("C(arm, Treatment('NL'))[T.Sim]:tau_std", 0)
    arm_code_tau = params.get("C(arm, Treatment('NL'))[T.Code]:tau_std", 0)

    # Find model's actual reference categories (absorbed into intercept, coef=0)
    all_families = set(long_df['task_family'].unique())
    modeled_families = {n.split('[T.')[1].rstrip(']')
                        for n in params.index if 'task_family' in n}
    ref_family = sorted(all_families - modeled_families)[0]

    family_coef = 0  # reference category: implicit coefficient = 0
    model_coef = 0

    # Get covariance matrix for CIs
    cov = model.cov_params()

    def inv_logit(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    # Extrapolation region shading
    ax.axvspan(max_obs_tau, 80, color='#f0f0f0', zorder=0)
    ax.axvline(x=max_obs_tau, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(max_obs_tau * 1.08, 0.50, 'extrapolation →', fontsize=10,
            color='gray', ha='left', va='center', style='italic', rotation=90)

    for arm in arms:
        probs = []
        ci_lower = []
        ci_upper = []

        for tau in tau_values:
            tau_std = (tau - tau_mean) / tau_std_val

            if arm == 'NL':
                eta = intercept + tau_coef * tau_std + family_coef + model_coef
                var_eta = (cov.loc['Intercept', 'Intercept'] +
                           tau_std**2 * cov.loc['tau_std', 'tau_std'] +
                           2 * tau_std * cov.loc['Intercept', 'tau_std'])
            elif arm == 'Sim':
                eta = intercept + arm_sim + tau_coef * tau_std + arm_sim_tau * tau_std + family_coef + model_coef
                var_eta = (cov.loc['Intercept', 'Intercept'] +
                           cov.loc["C(arm, Treatment('NL'))[T.Sim]", "C(arm, Treatment('NL'))[T.Sim]"] +
                           (tau_std**2) * (cov.loc['tau_std', 'tau_std'] +
                                           cov.loc["C(arm, Treatment('NL'))[T.Sim]:tau_std", "C(arm, Treatment('NL'))[T.Sim]:tau_std"]))
            else:  # Code
                eta = intercept + arm_code + tau_coef * tau_std + arm_code_tau * tau_std + family_coef + model_coef
                var_eta = (cov.loc['Intercept', 'Intercept'] +
                           cov.loc["C(arm, Treatment('NL'))[T.Code]", "C(arm, Treatment('NL'))[T.Code]"] +
                           (tau_std**2) * (cov.loc['tau_std', 'tau_std'] +
                                           cov.loc["C(arm, Treatment('NL'))[T.Code]:tau_std", "C(arm, Treatment('NL'))[T.Code]:tau_std"]))

            se_eta = np.sqrt(max(0, var_eta))
            prob = inv_logit(eta)
            probs.append(prob)
            ci_lower.append(inv_logit(eta - 1.96 * se_eta))
            ci_upper.append(inv_logit(eta + 1.96 * se_eta))

        probs = np.array(probs)
        ci_lower = np.array(ci_lower)
        ci_upper = np.array(ci_upper)

        # Interpolation: solid lines
        ax.plot(tau_values[interp_mask], probs[interp_mask], color=colors[arm],
                linewidth=2.5, label=arm, zorder=3)
        ax.fill_between(tau_values[interp_mask], ci_lower[interp_mask],
                        ci_upper[interp_mask], color=colors[arm], alpha=0.2)

        # Extrapolation: dashed lines, lighter CI
        ax.plot(tau_values[extrap_mask], probs[extrap_mask], color=colors[arm],
                linewidth=2.5, linestyle='--', zorder=3)
        ax.fill_between(tau_values[extrap_mask], ci_lower[extrap_mask],
                        ci_upper[extrap_mask], color=colors[arm], alpha=0.08)

        # Observed empirical means at reference task family (calibration)
        obs_df = long_df[(long_df['arm'] == arm) &
                         (long_df['task_family'] == ref_family)]
        obs = obs_df.groupby('digit')['correct'].mean()
        ax.scatter(obs.index, obs.values, color=colors[arm],
                   marker='x', s=60, zorder=5, linewidths=2)

    # GOF and odds ratios
    or_sim = np.exp(arm_sim)
    or_code = np.exp(arm_code)
    or_tau = np.exp(tau_coef)
    or_sim_tau = np.exp(arm_sim_tau)
    or_code_tau = np.exp(arm_code_tau)
    pseudo_r2 = 1 - (model.llf / model.llnull)

    coef_text = (
        f"Logistic GLMM — Odds Ratios:\n"
        f"  Sim vs NL: {or_sim:.2f}\n"
        f"  Code vs NL: {or_code:.2f}\n"
        f"  τ (per SD): {or_tau:.2f}\n"
        f"  Sim × τ: {or_sim_tau:.2f}\n"
        f"  Code × τ: {or_code_tau:.2f}\n"
        f"  McFadden R²: {pseudo_r2:.3f}"
    )
    ax.text(0.02, 0.02, coef_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # Observed marker annotation
    ax.text(0.98, 0.02, f'× observed ({ref_family})', fontsize=9, color='gray',
            transform=ax.transAxes, ha='right', va='bottom')

    # Log scale for τ
    ax.set_xscale('log', base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(['2', '4', '8', '16', '32', '64'])

    ax.set_xlabel('τ (Digit Length / Difficulty, log₂ scale)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted P(Correct)', fontsize=14, fontweight='bold')
    ax.set_title('Logistic GLMM: Marginal Predictions vs Difficulty\n'
                 'logit P(correct) = arm + τ + arm×τ + task_family + model  (cluster-robust SEs)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Arm', fontsize=12, title_fontsize=13, loc='upper right')
    ax.set_ylim([0, 1])
    ax.set_xlim([1.5, 80])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', dpi=300)
    print(f"[plot_glmm_predicted_prob] Saved {output_path}")
    plt.close()


def plot_glmm_odds_ratio_forest(glmm_result: dict, output_path: str = "figures/glmm_odds_ratio_forest.png") -> None:
    """
    Appendix plot: Odds ratio forest plot for fixed effects.
    Filters out effects with extreme coefficients (perfect/near-perfect separation).
    """
    from matplotlib import rcParams

    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["font.family"] = "Arial"

    model = glmm_result['model']

    # Extract coefficients and CIs
    params = model.params
    conf_int = model.conf_int()

    # Filter to main effects of interest (exclude intercept and extreme coefficients)
    effects_of_interest = []
    labels = []

    for name in params.index:
        if 'Intercept' in name:
            continue

        # Skip extreme coefficients (near-perfect separation, |coef| > 5)
        coef = params[name]
        if abs(coef) > 5:
            print(f"[plot_glmm_odds_ratio_forest] Skipping extreme coefficient: {name} = {coef:.2f}")
            continue

        # Clean up names
        label = name
        label = label.replace("C(arm, Treatment('NL'))[T.", "Arm: ").replace("]", "")
        label = label.replace("C(task_family)[T.", "Family: ").replace("]", "")
        label = label.replace("C(model_short)[T.", "Model: ").replace("]", "")
        label = label.replace("tau_std", "τ (std)")
        label = label.replace(":tau_std", " × τ")

        effects_of_interest.append(name)
        labels.append(label)

    if not effects_of_interest:
        print("[plot_glmm_odds_ratio_forest] No valid effects to plot")
        return

    # Compute odds ratios and CIs
    coefs = params[effects_of_interest].values
    ci_low = conf_int.loc[effects_of_interest, 0].values
    ci_high = conf_int.loc[effects_of_interest, 1].values

    odds_ratios = np.exp(coefs)
    or_ci_low = np.exp(np.clip(ci_low, -10, 10))  # Clip to avoid overflow
    or_ci_high = np.exp(np.clip(ci_high, -10, 10))

    # Sort by odds ratio magnitude
    sort_idx = np.argsort(odds_ratios)
    odds_ratios = odds_ratios[sort_idx]
    or_ci_low = or_ci_low[sort_idx]
    or_ci_high = or_ci_high[sort_idx]
    labels = [labels[i] for i in sort_idx]

    # Create forest plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4)))

    y_pos = np.arange(len(labels))

    # Color by type
    colors = []
    for label in labels:
        if 'Arm:' in label:
            colors.append('darkorange')
        elif '× τ' in label:
            colors.append('red')
        elif 'τ' in label:
            colors.append('purple')
        elif 'Family:' in label:
            colors.append('forestgreen')
        elif 'Model:' in label:
            colors.append('steelblue')
        else:
            colors.append('gray')

    # Plot points and CIs
    for i, (y, or_val, ci_l, ci_h, color) in enumerate(zip(y_pos, odds_ratios, or_ci_low, or_ci_high, colors)):
        ax.errorbar(or_val, y, xerr=[[or_val - ci_l], [ci_h - or_val]],
                    fmt='o', color=color, markersize=8, capsize=4, capthick=2, elinewidth=2)

    # Reference line at OR=1
    ax.axvline(1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='OR = 1 (no effect)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=14, fontweight='bold')
    ax.set_title('GLMM Fixed Effects: Odds Ratio Forest Plot\n(Excluding effects with perfect separation)', fontsize=15, fontweight='bold')

    # Log scale for x-axis
    ax.set_xscale('log')
    ax.set_xlim([0.05, 50])

    # Add legend for colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=10, label='Arm effect'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Arm × τ interaction'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='τ (difficulty)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='forestgreen', markersize=10, label='Task family'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=10, label='Model'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', dpi=300)
    print(f"[plot_glmm_odds_ratio_forest] Saved {output_path}")
    plt.close()


def run_glmm_analysis(df: pd.DataFrame) -> dict:
    """
    Run full GLMM analysis and generate plots.
    """
    print("\n=== GLMM Analysis ===")

    # Prepare data
    long_df = prepare_glmm_data(df)
    print(f"[run_glmm_analysis] Prepared {len(long_df)} observations in long format")

    # Fit model
    glmm_result = fit_glmm(long_df)

    # Print summary
    print("\n[run_glmm_analysis] Model Summary:")
    print(glmm_result['model'].summary())

    # Generate plots
    plot_glmm_predicted_prob(glmm_result)

    return glmm_result


goodset = [
    "clrs30",
    "add",
    "sub",
    "mul",
    "lcs",
    "rod",
    "knap",
    "ilp_assign",
    "ilp_prod",
    "ilp_partition",
    "spp",
    "tsp",
    "tsp_d",
    "msp",
    "ksp",
    "gcp",
    "gcp_d",
    "bsp",
    "edp",
]


def plot_main_fig(df: pd.DataFrame) -> None:
    # train_lengths_dict = {}
    # for alg, train_length in _DEFAULT_VAL_ALGOS_AND_LENGTHS.items():
    #     train_lengths_dict[alg] = np.array(train_length)
    sns.reset_defaults()
    # import pdb; pdb.set_trace()
    df1 = df
    df2 = df1[df1["kind"].isin(["add", "mul", "lcs", "rod", "knap", "ilp_assign", "ilp_prod", "ilp_partition"])]
    name_map = {
        "nl_correct": "Arm 1 \n (NL)",
        "sim_correct": "Arm 2 \n (Code Sim)",
        "controlsim_correct": "Arm 2.5 \n (Controlled Code Sim)",
        "code_correct": "Arm 3 \n (Code Exec)",
    }
    dfnew = df2.rename(columns=name_map)

    cols = list(name_map.values())
    mdf = pd.melt(dfnew, value_vars=cols, id_vars=["kind", "digit"])
    # mdf1 = mdf.groupby(["variable", "digit", "kind"]).mean().reset_index()
    g = sns.FacetGrid(mdf, col="kind", col_wrap=4, hue="variable", hue_order=cols, sharex=False)
    g.map(sns.lineplot, "digit", "value")
    g.set_titles("{col_name}")

    for ax in g.axes:
        alg = ax.title.get_text()
        ax.set_title(alg.replace("_", " "))
        # train_lengths = train_lengths_dict[alg]
        train_lengths = [2, 4, 8, 10, 12, 14, 16, 18, 20]
        ax.scatter(train_lengths, np.ones(len(train_lengths)) + 0.05, color="red", s=1.0)
        ax.set_xlim(None, 20)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend().set_visible(False)  # Hide individual legends

    g.set_xlabels("test length")

    # Add legend at the bottom
    handles, labels = g.axes[0].get_legend_handles_labels()
    g.fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=9,
        frameon=True,
    )
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("figures/main.png", bbox_inches="tight")


def plot_main_combined(df: pd.DataFrame, glmm_result: dict = None) -> None:
    """
    Combined figure with:
    - Left: 8 task panels (without controlsim, without sub)
    - Right: GLMM marginal predictions + Average
    Two rows layout.
    """
    from matplotlib import rcParams
    from matplotlib.gridspec import GridSpec

    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["font.family"] = "Arial"

    # Filter to 8 tasks (no sub)
    tasks = ["add", "mul", "lcs", "rod", "knap", "ilp_assign", "ilp_prod", "ilp_partition"]
    df_tasks = df[df["kind"].isin(tasks)].copy()

    # Arms to plot (no controlsim) - use red, blue, orange
    arms = ["nl_correct", "sim_correct", "code_correct"]
    arm_labels = {"nl_correct": "NL", "sim_correct": "Sim", "code_correct": "Code Exec"}
    arm_colors = {"nl_correct": "tab:blue", "sim_correct": "tab:orange", "code_correct": "tab:red"}

    # Create figure with custom grid: 2 rows × 5 cols
    # 8 tasks (4 per row) + GLMM + Average on right
    # More horizontal aspect ratio
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1], wspace=0.3, hspace=0.4)

    # Plot 8 task panels (2×4 grid on left)
    for idx, task in enumerate(tasks):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])

        task_df = df_tasks[df_tasks["kind"] == task]

        for arm in arms:
            # Group by digit and compute mean + bootstrap CI
            digits = sorted(task_df["digit"].unique())
            means = []
            ci_low = []
            ci_high = []

            for digit in digits:
                vals = task_df[task_df["digit"] == digit][arm].values
                mean = vals.mean()
                means.append(mean)
                # Bootstrap CI
                if len(vals) > 1:
                    boot_means = [np.random.choice(vals, size=len(vals), replace=True).mean()
                                  for _ in range(1000)]
                    ci_low.append(np.percentile(boot_means, 2.5))
                    ci_high.append(np.percentile(boot_means, 97.5))
                else:
                    ci_low.append(mean)
                    ci_high.append(mean)

            ax.plot(digits, means, marker='o', markersize=4, linewidth=1.5,
                    color=arm_colors[arm], label=arm_labels[arm])
            ax.fill_between(digits, ci_low, ci_high, color=arm_colors[arm], alpha=0.2)

        ax.set_title(task.replace("_", " "), fontsize=11, fontweight='bold')
        ax.set_xlim(1, 21)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)

        if row == 1:
            ax.set_xlabel("τ", fontsize=10)
        if col == 0:
            ax.set_ylabel("Accuracy", fontsize=10)

        # Store handles for shared legend (from first panel only)
        if idx == 0:
            task_handles, task_labels = ax.get_legend_handles_labels()

    # Right top: GLMM marginal predictions
    ax_glmm = fig.add_subplot(gs[0, 4])

    if glmm_result is not None:
        model = glmm_result['model']
        tau_mean = glmm_result['tau_mean']
        tau_std_val = glmm_result['tau_std']
        long_df = glmm_result['data']

        tau_values = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 56, 64])
        max_obs_tau = 20
        interp_mask = tau_values <= max_obs_tau
        extrap_mask = tau_values >= max_obs_tau  # overlap at boundary for line continuity

        glmm_arms = ['NL', 'Sim', 'Code']
        glmm_colors = {'NL': 'tab:blue', 'Sim': 'tab:orange', 'Code': 'tab:red'}

        params = model.params
        intercept = params.get('Intercept', 0)
        arm_sim = params.get("C(arm, Treatment('NL'))[T.Sim]", 0)
        arm_code = params.get("C(arm, Treatment('NL'))[T.Code]", 0)
        tau_coef = params.get("tau_std", 0)
        arm_sim_tau = params.get("C(arm, Treatment('NL'))[T.Sim]:tau_std", 0)
        arm_code_tau = params.get("C(arm, Treatment('NL'))[T.Code]:tau_std", 0)

        # Find model's actual reference categories (coef absorbed into intercept)
        all_families = set(long_df['task_family'].unique())
        modeled_families = {n.split('[T.')[1].rstrip(']')
                            for n in params.index if 'task_family' in n}
        ref_family = sorted(all_families - modeled_families)[0]

        family_coef = 0  # reference category: implicit coef = 0
        model_coef = 0

        cov = model.cov_params()

        def inv_logit(x):
            return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

        # Extrapolation region: gray background + vertical demarcation
        ax_glmm.axvspan(max_obs_tau, 80, color='#f0f0f0', zorder=0)
        ax_glmm.axvline(x=max_obs_tau, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

        for arm in glmm_arms:
            probs = []
            ci_lower = []
            ci_upper = []

            for tau in tau_values:
                tau_std = (tau - tau_mean) / tau_std_val

                if arm == 'NL':
                    eta = intercept + tau_coef * tau_std + family_coef + model_coef
                    var_eta = (cov.loc['Intercept', 'Intercept'] +
                               tau_std**2 * cov.loc['tau_std', 'tau_std'] +
                               2 * tau_std * cov.loc['Intercept', 'tau_std'])
                elif arm == 'Sim':
                    eta = intercept + arm_sim + tau_coef * tau_std + arm_sim_tau * tau_std + family_coef + model_coef
                    var_eta = (cov.loc['Intercept', 'Intercept'] +
                               cov.loc["C(arm, Treatment('NL'))[T.Sim]", "C(arm, Treatment('NL'))[T.Sim]"] +
                               (tau_std**2) * (cov.loc['tau_std', 'tau_std'] +
                                               cov.loc["C(arm, Treatment('NL'))[T.Sim]:tau_std", "C(arm, Treatment('NL'))[T.Sim]:tau_std"]))
                else:  # Code
                    eta = intercept + arm_code + tau_coef * tau_std + arm_code_tau * tau_std + family_coef + model_coef
                    var_eta = (cov.loc['Intercept', 'Intercept'] +
                               cov.loc["C(arm, Treatment('NL'))[T.Code]", "C(arm, Treatment('NL'))[T.Code]"] +
                               (tau_std**2) * (cov.loc['tau_std', 'tau_std'] +
                                               cov.loc["C(arm, Treatment('NL'))[T.Code]:tau_std", "C(arm, Treatment('NL'))[T.Code]:tau_std"]))

                se_eta = np.sqrt(max(0, var_eta))
                prob = inv_logit(eta)
                probs.append(prob)
                ci_lower.append(inv_logit(eta - 1.96 * se_eta))
                ci_upper.append(inv_logit(eta + 1.96 * se_eta))

            probs = np.array(probs)
            ci_lower = np.array(ci_lower)
            ci_upper = np.array(ci_upper)

            # Interpolation region: solid lines, full CI shading
            ax_glmm.plot(tau_values[interp_mask], probs[interp_mask],
                        linewidth=2, color=glmm_colors[arm], label=arm, zorder=3)
            ax_glmm.fill_between(tau_values[interp_mask], ci_lower[interp_mask],
                                ci_upper[interp_mask], color=glmm_colors[arm], alpha=0.2)

            # Extrapolation region: dashed lines, lighter CI shading
            ax_glmm.plot(tau_values[extrap_mask], probs[extrap_mask],
                        linewidth=2, linestyle='--', color=glmm_colors[arm], zorder=3)
            ax_glmm.fill_between(tau_values[extrap_mask], ci_lower[extrap_mask],
                                ci_upper[extrap_mask], color=glmm_colors[arm], alpha=0.08)

            # Observed empirical means at reference task family (calibration)
            obs_df = long_df[(long_df['arm'] == arm) &
                             (long_df['task_family'] == ref_family)]
            obs = obs_df.groupby('digit')['correct'].mean()
            ax_glmm.scatter(obs.index, obs.values, color=glmm_colors[arm],
                           marker='x', s=25, zorder=5, linewidths=1.5)

        # GOF and odds ratios annotation
        or_code = np.exp(arm_code)
        or_code_tau = np.exp(arm_code_tau)
        pseudo_r2 = 1 - (model.llf / model.llnull)
        ax_glmm.text(0.02, 0.02,
                     f"OR Code/NL: {or_code:.1f}\nCode×τ: {or_code_tau:.2f}\nMcF R²={pseudo_r2:.3f}",
                     transform=ax_glmm.transAxes, fontsize=7, verticalalignment='bottom',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))


    ax_glmm.set_title("Logistic GLMM", fontsize=11, fontweight='bold')
    ax_glmm.set_xlabel("difficulty (τ)", fontsize=10)
    ax_glmm.set_ylabel("P(Correct)", fontsize=10)
    ax_glmm.set_xscale('log', base=2)
    ax_glmm.set_xticks([2, 4, 8, 16, 32, 64])
    ax_glmm.set_xticklabels(['2', '4', '8', '16', '32', '64'])
    ax_glmm.set_xlim(1.5, 80)
    ax_glmm.set_ylim(0, 1.05)
    ax_glmm.tick_params(labelsize=9)
    ax_glmm.grid(True, alpha=0.3)

    # Right bottom: Average across all 8 tasks with instance bootstrap CI
    ax_avg = fig.add_subplot(gs[1, 4])

    # Create instance ID for clustering
    df_tasks['instance_id'] = df_tasks['kind'] + '_' + df_tasks['digit'].astype(str) + '_' + df_tasks['unique_tag'].astype(str)

    digits = sorted(df_tasks["digit"].unique())

    for arm in arms:
        means = []
        ci_low = []
        ci_high = []

        for digit in digits:
            digit_df = df_tasks[df_tasks["digit"] == digit]

            # Instance-level bootstrap
            instances = digit_df['instance_id'].unique()
            n_boot = 1000
            boot_means = []

            for _ in range(n_boot):
                boot_instances = np.random.choice(instances, size=len(instances), replace=True)
                boot_df = digit_df[digit_df['instance_id'].isin(boot_instances)]
                boot_means.append(boot_df[arm].mean())

            means.append(digit_df[arm].mean())
            ci_low.append(np.percentile(boot_means, 2.5))
            ci_high.append(np.percentile(boot_means, 97.5))

        ax_avg.plot(digits, means, marker='o', markersize=5, linewidth=2,
                    color=arm_colors[arm], label=arm_labels[arm])
        ax_avg.fill_between(digits, ci_low, ci_high, color=arm_colors[arm], alpha=0.2)

    ax_avg.set_title("Average (8 tasks)", fontsize=11, fontweight='bold')
    ax_avg.set_xlabel("τ", fontsize=10)
    ax_avg.set_ylabel("Accuracy", fontsize=10)
    ax_avg.set_xlim(1, 21)
    ax_avg.set_ylim(0, 1.05)
    ax_avg.tick_params(labelsize=9)
    ax_avg.grid(True, alpha=0.3)

    # Add shared legend at bottom
    fig.legend(task_handles, task_labels, loc='lower center', ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(bottom=0.15)
    plt.savefig("figures/main_combined.png", bbox_inches="tight", dpi=300)
    plt.savefig("figures/main_combined.pdf", bbox_inches="tight", dpi=300)
    print("[plot_main_combined] Saved figures/main_combined.png and .pdf")
    plt.close()


def plot_v_graph(df: pd.DataFrame) -> None:
    from matplotlib import rcParams

    rcParams["figure.dpi"] = 500
    rcParams["savefig.dpi"] = 500
    rcParams["font.family"] = "Arial"
    rcParams["axes.labelsize"] = 18
    rcParams["axes.titlesize"] = 18
    rcParams["legend.fontsize"] = 18
    rcParams["figure.titlesize"] = 18
    rcParams["markers.fillstyle"] = "none"
    fig, ax = plt.subplots(figsize=(6, 6))
    cols = ["nl_correct", "sim_correct", "controlsim_correct", "code_correct"]
    df1 = df[df["kind"].isin(["gsm8k"])]
    melted_df = pd.melt(df1, value_vars=cols, id_vars=["model", "kind"])
    # Classify models by provider prefix to split markers between open/closed
    closed_providers = {"anthropic", "openai", "xai"}
    open_providers = {"meta-llama", "mistral", "mistralai", "qwen", "deepseek", "microsoft", "allenai", "zhipuai"}

    def _is_open_model(model_name: str) -> bool:
        prefix = str(model_name).split("/")[0].lower()
        if prefix in closed_providers:
            return False
        if prefix in open_providers:
            return True
        return False

    melted_df = melted_df.copy()
    melted_df["model_type"] = melted_df["model"].apply(lambda m: "open" if _is_open_model(m) else "closed")
    open_models_df = melted_df[melted_df["model_type"] == "open"]
    closed_models_df = melted_df[melted_df["model_type"] == "closed"]

    hue_order = sorted(melted_df["model"].unique())
    palette_base = sorted(sns.color_palette("tab20", n_colors=20), key=lambda x: x[0] - x[2])

    closed_candidates = [m for m in hue_order if not _is_open_model(m)]
    open_candidates = [m for m in hue_order if _is_open_model(m)]

    closed_palette_map = {model: palette_base[i % len(palette_base)] for i, model in enumerate(closed_candidates)}
    # Offset open model colors slightly to reduce closeness with closed-model colors.
    open_palette_rotated = palette_base[2:] + palette_base[:2]
    open_palette_map = {model: open_palette_rotated[i % len(open_palette_rotated)] for i, model in enumerate(open_candidates)}

    palette_map = {**closed_palette_map, **open_palette_map}

    # Keep original coloring; draw separately with different markers but shared palette.
    if not closed_models_df.empty:
        sns.pointplot(
            data=closed_models_df,
            ax=ax,
            x="variable",
            y="value",
            hue="model",
            hue_order=hue_order,
            linestyle="",
            alpha=0.8,
            marker="x",  # closed models -> x
            markersize=12,
            linewidth=1.6,
            palette=palette_map,
            errorbar=None,
            legend=False,
        )
    if not open_models_df.empty:
        sns.pointplot(
            data=open_models_df,
            ax=ax,
            x="variable",
            y="value",
            hue="model",
            hue_order=hue_order,
            linestyle="",
            alpha=0.8,
            marker="o",  # open models -> o
            markersize=12,
            linewidth=1.6,
            palette=palette_map,
            errorbar=None,
            legend=False,
        )
    sns.lineplot(
        data=melted_df, ax=ax, x="variable", y="value", color="black", marker="o", markersize=10, fillstyle="full", label="All models", errorbar=None
    )
    # Build legend: first three closed models (x), then three open models (o), then All models.
    closed_priority = ["openai/gpt-4o-mini", "anthropic/claude-haiku-4.5", "google/gemini-2.5-flash"]
    open_priority = ["mistralai/ministral-14b-2512", "meta-llama/llama-3.1-405b-instruct", "qwen/qwen-2.5-coder-32b-instruct"]

    def _pick_models(priority: list[str], candidates: list[str], k: int) -> list[str]:
        picked: list[str] = []
        for m in priority:
            if m in candidates and m not in picked:
                picked.append(m)
            if len(picked) >= k:
                return picked
        for m in candidates:
            if m not in picked:
                picked.append(m)
            if len(picked) >= k:
                break
        return picked

    closed_candidates = [m for m in hue_order if not _is_open_model(m)]
    open_candidates = [m for m in hue_order if _is_open_model(m)]
    closed_order = _pick_models(closed_priority, closed_candidates, 3)
    open_order = _pick_models(open_priority, open_candidates, 3)

    from matplotlib.lines import Line2D

    handles_custom: list[Line2D] = []
    labels_custom: list[str] = []
    for m in closed_order:
        handles_custom.append(Line2D([0], [0], marker="x", color=palette_map[m], linestyle="", markersize=12, markeredgewidth=1.6))
        labels_custom.append(m)
    for m in open_order:
        handles_custom.append(Line2D([0], [0], marker="o", color=palette_map[m], linestyle="", markersize=12, markeredgewidth=1.6))
        labels_custom.append(m)

    # Append the "All models" entry from the lineplot
    line_handles, line_labels = plt.gca().get_legend_handles_labels()
    for h, label in zip(line_handles, line_labels):
        if label == "All models":
            handles_custom.append(h)
            labels_custom.append(label)
            break

    ax.legend(
        handles_custom,
        labels_custom,
        title="Model (o=open, x=closed)",
        markerscale=1.3,
        fontsize="large",
        title_fontsize="x-large",
    )
    ax.set_ylim([0, 1])
    plt.xlabel("Arm")
    ax.set_xticklabels(["NL", "Sim", "ControlSim", "Code"])
    plt.savefig("figures/line.png", bbox_inches="tight")


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    return (lower, upper)


def calc_parse_error_rate(series: pd.Series) -> float:
    """Calculate parse error rate handling both bool and string columns."""
    if len(series) == 0:
        return 0.0
    if series.dtype == bool:
        return series.sum() / len(series) * 100
    else:
        # String column - "ok" or empty means no error
        return ((series != "ok") & (series.notna()) & (series != "")).sum() / len(series) * 100


def filter_models_by_parse_error(df: pd.DataFrame, threshold: float = 50.0) -> tuple[pd.DataFrame, List[str]]:
    """
    Filter out models with >threshold% parse error on NL, Sim, or ControlSim arms.
    Code parse error is excluded since it tracks execution failures, not JSON parsing.

    Returns:
        Filtered DataFrame and list of excluded models.
    """
    parse_cols = ["nl_parse_err", "sim_parse_err", "controlsim_parse_err"]
    excluded_models = []

    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        for col in parse_cols:
            if col in model_df.columns:
                err_rate = calc_parse_error_rate(model_df[col])
                if err_rate > threshold:
                    excluded_models.append(model)
                    print(f"[filter] Excluding {model}: {col} error rate = {err_rate:.1f}%")
                    break

    filtered_df = df[~df["model"].isin(excluded_models)]
    return filtered_df, excluded_models


def plot_v_graph_closed(df: pd.DataFrame) -> None:
    """
    Line figure for closed models only, with bootstrap CIs and p-value annotations.
    Uses CLRS30, NPHard, and fine-grained tasks (excludes gsm8k).
    """
    from matplotlib import rcParams
    from matplotlib.lines import Line2D

    rcParams["figure.dpi"] = 500
    rcParams["savefig.dpi"] = 500
    rcParams["font.family"] = "Arial"
    rcParams["axes.labelsize"] = 10
    rcParams["axes.titlesize"] = 10
    rcParams["legend.fontsize"] = 6
    rcParams["figure.titlesize"] = 10
    rcParams["markers.fillstyle"] = "none"

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    cols = ["nl_correct", "sim_correct", "controlsim_correct", "code_correct"]
    arm_labels = ["NL", "Sim", "ControlSim", "Code"]

    # Define task sets
    CLRS_KINDS = {
        "activity_selector",
        "articulation_points",
        "bellman_ford",
        "bfs",
        "binary_search",
        "bridges",
        "bubble_sort",
        "dag_shortest_paths",
        "dfs",
        "dijkstra",
        "find_maximum_subarray_kadane",
        "floyd_warshall",
        "graham_scan",
        "heapsort",
        "insertion_sort",
        "jarvis_march",
        "kmp_matcher",
        "lcs_length",
        "matrix_chain_order",
        "minimum",
        "mst_kruskal",
        "mst_prim",
        "naive_string_matcher",
        "optimal_bst",
        "quickselect",
        "quicksort",
        "segments_intersect",
        "strongly_connected_components",
        "task_scheduling",
        "topological_sort",
    }
    NPHARD_KINDS = {"edp", "gcp", "ksp", "spp", "tsp"}
    FG_KINDS = {"add", "sub", "mul", "lcs", "rod", "knap", "ilp_assign", "ilp_prod", "ilp_partition"}

    # Combine all task sets (exclude gsm8k)
    target_kinds = CLRS_KINDS | NPHARD_KINDS | FG_KINDS
    df1 = df[df["kind"].isin(target_kinds)]

    print(f"[plot_v_graph_closed] Filtering to {len(target_kinds)} task kinds")
    print(f"[plot_v_graph_closed] Kinds in data: {sorted(df1['kind'].unique())}")

    # Define closed providers
    closed_providers = {"anthropic", "openai", "google", "xai"}

    def _is_closed_model(model_name: str) -> bool:
        prefix = str(model_name).split("/")[0].lower()
        return prefix in closed_providers

    # Filter to closed models only
    df_closed = df1[df1["model"].apply(_is_closed_model)]

    if df_closed.empty:
        print("[plot_v_graph_closed] No closed models found in data.")
        return

    melted_df = pd.melt(df_closed, value_vars=cols, id_vars=["model", "kind"])
    melted_df = melted_df.copy()

    # Get unique models and set up palette
    unique_models = sorted(melted_df["model"].unique())
    palette_base = sns.color_palette("tab10", n_colors=len(unique_models))
    palette_map = {model: palette_base[i] for i, model in enumerate(unique_models)}

    # Compute aggregated stats per arm with bootstrap CIs
    arm_stats = []
    for col in cols:
        values = df_closed[col].dropna().values
        mean_val = np.mean(values) if len(values) > 0 else np.nan
        ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=1000, ci=0.95)
        arm_stats.append(
            {
                "arm": col,
                "mean": mean_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
    stats_df = pd.DataFrame(arm_stats)

    # Plot individual model points
    for model in unique_models:
        model_data = melted_df[melted_df["model"] == model]
        model_means = model_data.groupby("variable")["value"].mean().reindex(cols)
        x_positions = list(range(len(cols)))
        ax.scatter(
            x_positions,
            model_means.values,
            marker="x",
            s=120,
            color=palette_map[model],
            alpha=0.8,
            linewidths=2,
            label=model,
        )

    # Plot aggregated line with bootstrap CI error bars
    x_positions = list(range(len(cols)))
    means = stats_df["mean"].values
    ci_lower = stats_df["ci_lower"].values
    ci_upper = stats_df["ci_upper"].values
    yerr_lower = means - ci_lower
    yerr_upper = ci_upper - means

    ax.errorbar(
        x_positions,
        means,
        yerr=[yerr_lower, yerr_upper],
        fmt="o-",
        color="black",
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        label="All closed models (mean)",
    )

    # Compute and display p-values between adjacent arms only (cleaner visualization)
    # Adjacent pairs: NL-Sim, Sim-ControlSim, ControlSim-Code
    # Use model×kind combinations for proper sample size (not just n=4 models)
    adjacent_pairs = [(cols[i], cols[i + 1]) for i in range(len(cols) - 1)]
    mdf_grouped = melted_df.groupby(["variable", "model", "kind"])["value"].mean().reset_index()
    mdf_pivot = mdf_grouped.pivot(index=["model", "kind"], columns="variable", values="value")

    p_values = []
    for pair in adjacent_pairs:
        col1, col2 = pair
        if col1 in mdf_pivot.columns and col2 in mdf_pivot.columns:
            x = mdf_pivot[col1].dropna()
            y = mdf_pivot[col2].dropna()
            common_idx = x.index.intersection(y.index)
            if len(common_idx) >= 2:
                try:
                    stat, p_val = wilcoxon(x.loc[common_idx], y.loc[common_idx])
                except Exception:
                    p_val = 1.0
            else:
                p_val = 1.0
        else:
            p_val = 1.0
        p_values.append(p_val)

    # Draw p-value annotations for adjacent pairs
    max_val = max(ci_upper) if len(ci_upper) > 0 else 1.0
    offset = 0.05
    col_to_x = {col: i for i, col in enumerate(cols)}

    for i, (pair, p_val) in enumerate(zip(adjacent_pairs, p_values)):
        col1, col2 = pair
        x1 = col_to_x[col1]
        x2 = col_to_x[col2]
        y = max_val + offset
        h = 0.02
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="steelblue")

        # Format p-value
        if p_val < 0.001:
            p_text = "p<0.001"
        elif p_val < 0.01:
            p_text = f"p={p_val:.3f}"
        else:
            p_text = f"p={p_val:.2f}"
        ax.text((x1 + x2) * 0.5, y + h + 0.01, p_text, ha="center", va="bottom", fontsize=9)
        offset += 0.08

    # Set up axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(arm_labels)
    ax.set_xlabel("Arm")
    ax.set_ylabel("Accuracy")
    # Fix y-axis at [0, 1] for accuracy, extend slightly above for annotations
    y_max = max_val + offset + 0.08
    ax.set_ylim([0, y_max])
    ax.set_title("Closed Models: Accuracy Across Arms\n(CLRS + NPHard + Fine-grained)")

    # Build legend with abbreviated names
    handles_custom: list[Line2D] = []
    labels_custom: list[str] = []
    for model in unique_models:
        handles_custom.append(Line2D([0], [0], marker="x", color=palette_map[model], linestyle="", markersize=7, markeredgewidth=1.5))
        short_name = model.split("/")[-1] if "/" in model else model
        labels_custom.append(short_name)
    # Add aggregated line entry
    handles_custom.append(Line2D([0], [0], marker="o", color="black", linestyle="-", markersize=7, linewidth=1.5))
    labels_custom.append("Mean ± 95% CI")

    ax.legend(
        handles_custom,
        labels_custom,
        title="Closed",
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    plt.tight_layout()
    plt.savefig("figures/line_closed.png", bbox_inches="tight", pad_inches=0.02)
    print("[plot_v_graph_closed] Saved figures/line_closed.png")


def plot_v_graph_open(df: pd.DataFrame) -> None:
    """
    Line figure for open models only, with bootstrap CIs and p-value annotations.
    Uses CLRS30, NPHard, and fine-grained tasks (excludes gsm8k).
    """
    from matplotlib import rcParams
    from matplotlib.lines import Line2D

    rcParams["figure.dpi"] = 500
    rcParams["savefig.dpi"] = 500
    rcParams["font.family"] = "Arial"
    rcParams["axes.labelsize"] = 10
    rcParams["axes.titlesize"] = 10
    rcParams["legend.fontsize"] = 6
    rcParams["figure.titlesize"] = 10
    rcParams["markers.fillstyle"] = "none"

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    cols = ["nl_correct", "sim_correct", "controlsim_correct", "code_correct"]
    arm_labels = ["NL", "Sim", "ControlSim", "Code"]

    # Define task sets (same as closed)
    CLRS_KINDS = {
        "activity_selector", "articulation_points", "bellman_ford", "bfs",
        "binary_search", "bridges", "bubble_sort", "dag_shortest_paths",
        "dfs", "dijkstra", "find_maximum_subarray_kadane", "floyd_warshall",
        "graham_scan", "heapsort", "insertion_sort", "jarvis_march",
        "kmp_matcher", "lcs_length", "matrix_chain_order", "minimum",
        "mst_kruskal", "mst_prim", "naive_string_matcher", "optimal_bst",
        "quickselect", "quicksort", "segments_intersect",
        "strongly_connected_components", "task_scheduling", "topological_sort",
    }
    NPHARD_KINDS = {"edp", "gcp", "ksp", "spp", "tsp"}
    FG_KINDS = {"add", "sub", "mul", "lcs", "rod", "knap", "ilp_assign", "ilp_prod", "ilp_partition"}

    target_kinds = CLRS_KINDS | NPHARD_KINDS | FG_KINDS
    df1 = df[df["kind"].isin(target_kinds)]

    # Define open providers
    closed_providers = {"anthropic", "openai", "google", "xai"}

    def _is_open_model(model_name: str) -> bool:
        prefix = str(model_name).split("/")[0].lower()
        return prefix not in closed_providers

    # Filter to open models only
    df_open = df1[df1["model"].apply(_is_open_model)]

    if df_open.empty:
        print("[plot_v_graph_open] No open models found in data.")
        return

    print(f"[plot_v_graph_open] Found {len(df_open['model'].unique())} open models")

    melted_df = pd.melt(df_open, value_vars=cols, id_vars=["model", "kind"])
    melted_df = melted_df.copy()

    unique_models = sorted(melted_df["model"].unique())
    palette_base = sns.color_palette("tab10", n_colors=len(unique_models))
    palette_map = {model: palette_base[i] for i, model in enumerate(unique_models)}

    # Compute aggregated stats per arm with bootstrap CIs
    arm_stats = []
    for col in cols:
        values = df_open[col].dropna().values
        mean_val = np.mean(values) if len(values) > 0 else np.nan
        ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=1000, ci=0.95)
        arm_stats.append({"arm": col, "mean": mean_val, "ci_lower": ci_lower, "ci_upper": ci_upper})
    stats_df = pd.DataFrame(arm_stats)

    # Plot individual model points
    for model in unique_models:
        model_data = melted_df[melted_df["model"] == model]
        model_means = model_data.groupby("variable")["value"].mean().reindex(cols)
        x_positions = list(range(len(cols)))
        ax.scatter(
            x_positions, model_means.values, marker="o", s=120,
            color=palette_map[model], alpha=0.8, linewidths=2, label=model,
        )

    # Plot aggregated line with bootstrap CI error bars
    x_positions = list(range(len(cols)))
    means = stats_df["mean"].values
    ci_lower = stats_df["ci_lower"].values
    ci_upper = stats_df["ci_upper"].values
    yerr_lower = means - ci_lower
    yerr_upper = ci_upper - means

    ax.errorbar(
        x_positions, means, yerr=[yerr_lower, yerr_upper],
        fmt="s-", color="black", markersize=10, linewidth=2,
        capsize=5, capthick=2, label="All open models (mean)",
    )

    # Compute p-values for adjacent pairs
    adjacent_pairs = [(cols[i], cols[i + 1]) for i in range(len(cols) - 1)]
    mdf_grouped = melted_df.groupby(["variable", "model", "kind"])["value"].mean().reset_index()
    mdf_pivot = mdf_grouped.pivot(index=["model", "kind"], columns="variable", values="value")

    p_values = []
    for pair in adjacent_pairs:
        col1, col2 = pair
        if col1 in mdf_pivot.columns and col2 in mdf_pivot.columns:
            x = mdf_pivot[col1].dropna()
            y = mdf_pivot[col2].dropna()
            common_idx = x.index.intersection(y.index)
            if len(common_idx) >= 2:
                try:
                    stat, p_val = wilcoxon(x.loc[common_idx], y.loc[common_idx])
                except Exception:
                    p_val = 1.0
            else:
                p_val = 1.0
        else:
            p_val = 1.0
        p_values.append(p_val)

    # Draw p-value annotations
    max_val = max(ci_upper) if len(ci_upper) > 0 else 1.0
    offset = 0.05
    col_to_x = {col: i for i, col in enumerate(cols)}

    for i, (pair, p_val) in enumerate(zip(adjacent_pairs, p_values)):
        col1, col2 = pair
        x1, x2 = col_to_x[col1], col_to_x[col2]
        y = max_val + offset
        h = 0.02
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="steelblue")
        p_text = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}" if p_val < 0.01 else f"p={p_val:.2f}"
        ax.text((x1 + x2) * 0.5, y + h + 0.01, p_text, ha="center", va="bottom", fontsize=9)
        offset += 0.08

    ax.set_xticks(x_positions)
    ax.set_xticklabels(arm_labels)
    ax.set_xlabel("Arm")
    ax.set_ylabel("Accuracy")
    y_max = max_val + offset + 0.08
    ax.set_ylim([0, y_max])
    ax.set_title("Open Models: Accuracy Across Arms\n(CLRS + NPHard + Fine-grained)")

    # Build legend with abbreviated names
    handles_custom: list[Line2D] = []
    labels_custom: list[str] = []
    for model in unique_models:
        handles_custom.append(Line2D([0], [0], marker="o", color=palette_map[model], linestyle="", markersize=7, markeredgewidth=1.5))
        short_name = model.split("/")[-1] if "/" in model else model
        labels_custom.append(short_name)
    handles_custom.append(Line2D([0], [0], marker="s", color="black", linestyle="-", markersize=7, linewidth=1.5))
    labels_custom.append("Mean ± 95% CI")

    ax.legend(
        handles_custom,
        labels_custom,
        title="Open",
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    plt.tight_layout()
    plt.savefig("figures/line_open.png", bbox_inches="tight", pad_inches=0.02)
    print("[plot_v_graph_open] Saved figures/line_open.png")


def plot_v_graph_all(df: pd.DataFrame, selected_models: list[str] | None = None) -> None:
    """
    Line figure for ALL models (open + closed), with cluster bootstrap CIs,
    McNemar pairwise tests, and Cochran's Q omnibus test.
    Uses CLRS30, NPHard, and fine-grained tasks (excludes gsm8k).
    Open models use 'o' marker, closed models use 'x' marker.

    NOTE: controlsim branch is excluded - only NL, Sim, Code are shown.

    Args:
        df: DataFrame with results
        selected_models: If provided, only include these models. Otherwise use all.
    """
    from matplotlib import rcParams
    from matplotlib.lines import Line2D

    # Compressed figure with bigger text
    rcParams["figure.dpi"] = 500
    rcParams["savefig.dpi"] = 500
    rcParams["font.family"] = "Arial"
    rcParams["axes.labelsize"] = 16
    rcParams["axes.titlesize"] = 17
    rcParams["legend.fontsize"] = 12
    rcParams["figure.titlesize"] = 17
    rcParams["markers.fillstyle"] = "none"

    # Compressed figure size
    fig, ax = plt.subplots(figsize=(7, 5))

    # Three branches only - NO controlsim
    cols = ["nl_correct", "sim_correct", "code_correct"]
    arm_labels = ["NL", "Sim", "Code Exec"]

    # Define task sets
    CLRS_KINDS = {
        "activity_selector", "articulation_points", "bellman_ford", "bfs",
        "binary_search", "bridges", "bubble_sort", "dag_shortest_paths",
        "dfs", "dijkstra", "find_maximum_subarray_kadane", "floyd_warshall",
        "graham_scan", "heapsort", "insertion_sort", "jarvis_march",
        "kmp_matcher", "lcs_length", "matrix_chain_order", "minimum",
        "mst_kruskal", "mst_prim", "naive_string_matcher", "optimal_bst",
        "quickselect", "quicksort", "segments_intersect",
        "strongly_connected_components", "task_scheduling", "topological_sort",
    }
    NPHARD_KINDS = {"edp", "gcp", "ksp", "spp", "tsp"}
    FG_KINDS = {"add", "sub", "mul", "lcs", "rod", "knap", "ilp_assign", "ilp_prod", "ilp_partition"}

    target_kinds = CLRS_KINDS | NPHARD_KINDS | FG_KINDS
    df1 = df[df["kind"].isin(target_kinds)].copy()

    # Filter to selected models if provided
    if selected_models is not None:
        df1 = df1[df1["model"].isin(selected_models)]
        print(f"[plot_v_graph_all] Filtered to {len(selected_models)} selected models: {selected_models}")

    if df1.empty:
        print("[plot_v_graph_all] No data after filtering.")
        return

    # Create instance_id for cluster bootstrap (unique problems, not model/seed variations)
    df1 = create_instance_id(df1)
    print(f"[plot_v_graph_all] Created {df1['instance_id'].nunique()} unique instances for clustering")

    # Classify models
    closed_providers = {"anthropic", "openai", "google", "xai"}

    def _is_closed_model(model_name: str) -> bool:
        prefix = str(model_name).split("/")[0].lower()
        return prefix in closed_providers

    melted_df = pd.melt(df1, value_vars=cols, id_vars=["model", "kind"])
    melted_df = melted_df.copy()
    melted_df["model_type"] = melted_df["model"].apply(lambda m: "closed" if _is_closed_model(m) else "open")

    unique_models = sorted(melted_df["model"].unique())
    closed_models = [m for m in unique_models if _is_closed_model(m)]
    open_models = [m for m in unique_models if not _is_closed_model(m)]

    print(f"[plot_v_graph_all] Closed models: {closed_models}")
    print(f"[plot_v_graph_all] Open models: {open_models}")

    # Create distinct color palettes for open vs closed
    closed_palette = sns.color_palette("Blues", n_colors=max(len(closed_models), 3))
    open_palette = sns.color_palette("Oranges", n_colors=max(len(open_models), 3))

    palette_map = {}
    for i, m in enumerate(closed_models):
        palette_map[m] = closed_palette[min(i, len(closed_palette) - 1)]
    for i, m in enumerate(open_models):
        palette_map[m] = open_palette[min(i, len(open_palette) - 1)]

    # Compute aggregated stats per arm with CLUSTER bootstrap CIs (by instance_id)
    arm_stats = []
    for col in cols:
        values = df1[col].dropna().values
        mean_val = np.mean(values) if len(values) > 0 else np.nan
        ci_lower, ci_upper = cluster_bootstrap_ci(df1, col, cluster_col='instance_id', n_bootstrap=1000, ci=0.95)
        arm_stats.append({"arm": col, "mean": mean_val, "ci_lower": ci_lower, "ci_upper": ci_upper})
    stats_df = pd.DataFrame(arm_stats)

    print("[plot_v_graph_all] Arm stats with cluster bootstrap CIs (by instance):")
    for _, row in stats_df.iterrows():
        print(f"  {row['arm']}: {row['mean']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")

    # Plot individual model points with per-model CIs
    # Add jitter for visibility - stagger models horizontally
    x_positions = list(range(len(cols)))
    all_model_values = {i: [] for i in range(len(cols))}

    # Create jitter offsets for each model
    n_closed = len(closed_models)
    n_open = len(open_models)
    jitter_width = 0.35  # Total width of jitter spread

    # Closed models: spread from -jitter_width/2 to center
    closed_offsets = np.linspace(-jitter_width/2, -0.02, n_closed) if n_closed > 1 else [(-jitter_width/4) if n_closed == 1 else 0]
    # Open models: spread from center to +jitter_width/2
    open_offsets = np.linspace(0.02, jitter_width/2, n_open) if n_open > 1 else [(jitter_width/4) if n_open == 1 else 0]

    model_offsets = {}
    for i, m in enumerate(closed_models):
        model_offsets[m] = closed_offsets[i] if i < len(closed_offsets) else 0
    for i, m in enumerate(open_models):
        model_offsets[m] = open_offsets[i] if i < len(open_offsets) else 0

    for model in unique_models:
        model_data = melted_df[melted_df["model"] == model]
        model_df = df1[df1["model"] == model]
        model_means = model_data.groupby("variable")["value"].mean().reindex(cols)
        marker = "x" if _is_closed_model(model) else "o"
        offset = model_offsets.get(model, 0)

        # Compute per-model cluster bootstrap CIs (by instance_id)
        model_cis = []
        for col in cols:
            if len(model_df) > 1:
                ci_lo, ci_hi = cluster_bootstrap_ci(model_df, col, cluster_col='instance_id', n_bootstrap=500, ci=0.95)
            else:
                ci_lo, ci_hi = model_means[col], model_means[col]
            model_cis.append((ci_lo, ci_hi))

        model_yerr_lower = [max(0, model_means.values[i] - model_cis[i][0]) for i in range(len(cols))]
        model_yerr_upper = [max(0, model_cis[i][1] - model_means.values[i]) for i in range(len(cols))]

        # Apply jitter offset to x positions
        x_jittered = [x + offset for x in x_positions]

        ax.errorbar(
            x_jittered, model_means.values, yerr=[model_yerr_lower, model_yerr_upper],
            fmt=marker, markersize=12, color=palette_map[model], alpha=0.8,
            linewidth=0, elinewidth=2, capsize=4, capthick=1.5, label=model,
        )

        for i, val in enumerate(model_means.values):
            all_model_values[i].append(val)
            all_model_values[i].append(model_cis[i][1])

    # Compute separate means and CIs for closed and open models
    df_closed = df1[df1["model"].isin(closed_models)]
    df_open = df1[df1["model"].isin(open_models)]

    # Closed models stats
    closed_stats = []
    for col in cols:
        values = df_closed[col].dropna().values
        mean_val = np.mean(values) if len(values) > 0 else np.nan
        ci_lo, ci_hi = cluster_bootstrap_ci(df_closed, col, cluster_col='instance_id', n_bootstrap=1000, ci=0.95)
        closed_stats.append({"arm": col, "mean": mean_val, "ci_lower": ci_lo, "ci_upper": ci_hi})
    closed_stats_df = pd.DataFrame(closed_stats)

    # Open models stats
    open_stats = []
    for col in cols:
        values = df_open[col].dropna().values
        mean_val = np.mean(values) if len(values) > 0 else np.nan
        ci_lo, ci_hi = cluster_bootstrap_ci(df_open, col, cluster_col='instance_id', n_bootstrap=1000, ci=0.95)
        open_stats.append({"arm": col, "mean": mean_val, "ci_lower": ci_lo, "ci_upper": ci_hi})
    open_stats_df = pd.DataFrame(open_stats)

    print("[plot_v_graph_all] Closed models mean ± 95% CI:")
    for _, row in closed_stats_df.iterrows():
        print(f"  {row['arm']}: {row['mean']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
    print("[plot_v_graph_all] Open models mean ± 95% CI:")
    for _, row in open_stats_df.iterrows():
        print(f"  {row['arm']}: {row['mean']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")

    # Plot closed models mean line (slightly left offset)
    closed_means = closed_stats_df["mean"].values
    closed_ci_lower = closed_stats_df["ci_lower"].values
    closed_ci_upper = closed_stats_df["ci_upper"].values
    closed_yerr_lower = np.maximum(0, closed_means - closed_ci_lower)
    closed_yerr_upper = np.maximum(0, closed_ci_upper - closed_means)

    ax.errorbar(
        [x - 0.12 for x in x_positions], closed_means, yerr=[closed_yerr_lower, closed_yerr_upper],
        fmt="s-", color="steelblue", markersize=9, linewidth=2.5,
        capsize=5, capthick=2, label="Closed mean",
    )

    # Plot open models mean line (slightly right offset)
    open_means = open_stats_df["mean"].values
    open_ci_lower = open_stats_df["ci_lower"].values
    open_ci_upper = open_stats_df["ci_upper"].values
    open_yerr_lower = np.maximum(0, open_means - open_ci_lower)
    open_yerr_upper = np.maximum(0, open_ci_upper - open_means)

    ax.errorbar(
        [x + 0.12 for x in x_positions], open_means, yerr=[open_yerr_lower, open_yerr_upper],
        fmt="^-", color="darkorange", markersize=9, linewidth=2.5,
        capsize=5, capthick=2, label="Open mean",
    )

    # Use overall stats for annotations
    means = stats_df["mean"].values
    ci_lower = stats_df["ci_lower"].values
    ci_upper = stats_df["ci_upper"].values
    yerr_upper = np.maximum(0, ci_upper - means)

    # Cochran's Q test for omnibus comparison
    cochran_result = cochrans_q_test(df1, cols)
    print(f"[plot_v_graph_all] Cochran's Q = {cochran_result['Q']:.2f}, p = {cochran_result['p_value']:.4e}")

    # Pairwise McNemar tests (one-sided) with Holm correction
    adjacent_pairs = [(cols[i], cols[i + 1]) for i in range(len(cols) - 1)]
    mcnemar_results = []
    p_values_raw = []

    for col_a, col_b in adjacent_pairs:
        result = mcnemar_one_sided(df1, col_a, col_b)
        mcnemar_results.append(result)
        p_values_raw.append(result['p_value'])

    # Holm correction
    _, p_values_holm, _, _ = multipletests(p_values_raw, method='holm')

    print("[plot_v_graph_all] Pairwise McNemar (one-sided, Holm corrected):")
    for i, (pair, result, p_holm) in enumerate(zip(adjacent_pairs, mcnemar_results, p_values_holm)):
        print(f"  {pair[0]} → {pair[1]}: n01={result['n01']}, n10={result['n10']}, p_raw={result['p_value']:.4f}, p_holm={p_holm:.4f}")

    # Determine y positions to avoid overlaps
    max_data_y = max(max(all_model_values[i]) for i in range(len(cols)) if all_model_values[i])

    # Draw p-value annotations - position above the max data
    col_to_x = {col: i for i, col in enumerate(cols)}
    p_y_start = max_data_y + 0.06
    p_y_step = 0.055

    for i, (pair, p_holm) in enumerate(zip(adjacent_pairs, p_values_holm)):
        col1, col2 = pair
        x1, x2 = col_to_x[col1], col_to_x[col2]
        y = p_y_start + i * p_y_step
        h = 0.018
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="steelblue")

        sig = "***" if p_holm < 0.001 else "**" if p_holm < 0.01 else "*" if p_holm < 0.05 else "ns"
        p_text = f"p<.001 {sig}" if p_holm < 0.001 else f"p={p_holm:.2f} {sig}"
        ax.text((x1 + x2) * 0.5, y + h + 0.012, p_text, ha="center", va="bottom", fontsize=13, fontweight="bold")

    # Add mean value annotations - position below p-value brackets but above data
    for i, (x, mean_val) in enumerate(zip(x_positions, means)):
        # Position annotation at a fixed height above the mean line
        y_pos = means[i] + yerr_upper[i] + 0.025
        ax.annotate(
            f"{mean_val:.1%}",
            xy=(x, y_pos),
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="black",
        )

    # Add Cochran's Q annotation box
    q_text = f"Cochran's Q = {cochran_result['Q']:.1f}"
    q_p = "p<.001" if cochran_result['p_value'] < 0.001 else f"p={cochran_result['p_value']:.3f}"
    ax.text(0.02, 0.98, f"{q_text}, {q_p}", transform=ax.transAxes,
            fontsize=13, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.9))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(arm_labels, fontsize=15, fontweight='bold')
    ax.set_xlabel("Arm", fontsize=16, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=13)

    # Dynamic y-limit to avoid overlaps
    y_max = p_y_start + len(adjacent_pairs) * p_y_step + 0.08
    ax.set_ylim([0.08, max(0.65, y_max)])
    ax.set_title("Accuracy Across Arms\n(Cluster Bootstrap 95% CIs, McNemar one-sided)", fontsize=16, fontweight="bold")

    # Build legend - show ALL models
    handles_custom: list[Line2D] = []
    labels_custom: list[str] = []

    # All closed models
    for model in closed_models:
        handles_custom.append(Line2D([0], [0], marker="x", color=palette_map[model], linestyle="", markersize=8, markeredgewidth=2))
        short_name = model.split("/")[-1] if "/" in model else model
        labels_custom.append(short_name)

    # All open models
    for model in open_models:
        handles_custom.append(Line2D([0], [0], marker="o", color=palette_map[model], linestyle="", markersize=8, markeredgewidth=2))
        short_name = model.split("/")[-1] if "/" in model else model
        labels_custom.append(short_name)

    # Add closed and open mean lines
    handles_custom.append(Line2D([0], [0], marker="s", color="steelblue", linestyle="-", markersize=8, linewidth=2))
    labels_custom.append("Closed ± 95% CI")
    handles_custom.append(Line2D([0], [0], marker="^", color="darkorange", linestyle="-", markersize=8, linewidth=2))
    labels_custom.append("Open ± 95% CI")

    ax.legend(
        handles_custom,
        labels_custom,
        title="Models (x=closed, o=open)",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(handles_custom), 5),
        fontsize=10,
        title_fontsize=11,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.0,
        framealpha=0.95,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.24)
    plt.savefig("figures/line_all.png", bbox_inches="tight", pad_inches=0.05)
    print("[plot_v_graph_all] Saved figures/line_all.png")


def plot_combined_accuracy_delta(df: pd.DataFrame, selected_models: list[str] | None = None) -> None:
    """
    Combined figure: Accuracy across routes (left) + Delta distributions (right).
    Bootstraps on instance level (unique problems).
    """
    from matplotlib import rcParams
    from matplotlib.lines import Line2D

    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["font.family"] = "Arial"
    rcParams["axes.labelsize"] = 11
    rcParams["axes.titlesize"] = 11
    rcParams["legend.fontsize"] = 8
    rcParams["figure.titlesize"] = 12
    rcParams["markers.fillstyle"] = "none"

    # Create figure with GridSpec for flexible layout (compressed for paper)
    fig = plt.figure(figsize=(14, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.8, 0.8], wspace=0.35)

    ax_acc = fig.add_subplot(gs[0])
    ax_delta1 = fig.add_subplot(gs[1])
    ax_delta2 = fig.add_subplot(gs[2])

    # Three branches only - NO controlsim
    cols = ["nl_correct", "sim_correct", "code_correct"]
    arm_labels = ["NL", "Sim", "Code Exec"]

    # Define task sets
    CLRS_KINDS = {
        "activity_selector", "articulation_points", "bellman_ford", "bfs",
        "binary_search", "bridges", "bubble_sort", "dag_shortest_paths",
        "dfs", "dijkstra", "find_maximum_subarray_kadane", "floyd_warshall",
        "graham_scan", "heapsort", "insertion_sort", "jarvis_march",
        "kmp_matcher", "lcs_length", "matrix_chain_order", "minimum",
        "mst_kruskal", "mst_prim", "naive_string_matcher", "optimal_bst",
        "quickselect", "quicksort", "segments_intersect",
        "strongly_connected_components", "task_scheduling", "topological_sort",
    }
    NPHARD_KINDS = {"edp", "gcp", "ksp", "spp", "tsp"}
    FG_KINDS = {"add", "sub", "mul", "lcs", "rod", "knap", "ilp_assign", "ilp_prod", "ilp_partition"}

    target_kinds = CLRS_KINDS | NPHARD_KINDS | FG_KINDS
    df1 = df[df["kind"].isin(target_kinds)].copy()

    if selected_models is not None:
        df1 = df1[df1["model"].isin(selected_models)]

    if df1.empty:
        print("[plot_combined] No data after filtering.")
        return

    # Create instance_id for cluster bootstrap
    df1 = create_instance_id(df1)
    print(f"[plot_combined] Created {df1['instance_id'].nunique()} unique instances for clustering")

    # Classify models
    closed_providers = {"anthropic", "openai", "google", "xai"}

    def _is_closed_model(model_name: str) -> bool:
        prefix = str(model_name).split("/")[0].lower()
        return prefix in closed_providers

    unique_models = sorted(df1["model"].unique())
    closed_models = [m for m in unique_models if _is_closed_model(m)]
    open_models = [m for m in unique_models if not _is_closed_model(m)]

    print(f"[plot_combined] Closed models: {closed_models}")
    print(f"[plot_combined] Open models: {open_models}")

    # Color palettes
    closed_palette = sns.color_palette("Blues", n_colors=max(len(closed_models), 3))
    open_palette = sns.color_palette("Oranges", n_colors=max(len(open_models), 3))

    palette_map = {}
    for i, m in enumerate(closed_models):
        palette_map[m] = closed_palette[min(i, len(closed_palette) - 1)]
    for i, m in enumerate(open_models):
        palette_map[m] = open_palette[min(i, len(open_palette) - 1)]

    # === LEFT PANEL: Accuracy across routes ===
    melted_df = pd.melt(df1, value_vars=cols, id_vars=["model", "kind"])
    melted_df["model_type"] = melted_df["model"].apply(lambda m: "closed" if _is_closed_model(m) else "open")

    x_positions = list(range(len(cols)))
    all_model_values = {i: [] for i in range(len(cols))}

    # Jitter offsets
    n_closed = len(closed_models)
    n_open = len(open_models)
    jitter_width = 0.35

    closed_offsets = np.linspace(-jitter_width/2, -0.02, n_closed) if n_closed > 1 else [(-jitter_width/4) if n_closed == 1 else 0]
    open_offsets = np.linspace(0.02, jitter_width/2, n_open) if n_open > 1 else [(jitter_width/4) if n_open == 1 else 0]

    model_offsets = {}
    for i, m in enumerate(closed_models):
        model_offsets[m] = closed_offsets[i] if i < len(closed_offsets) else 0
    for i, m in enumerate(open_models):
        model_offsets[m] = open_offsets[i] if i < len(open_offsets) else 0

    # Plot individual model points
    for model in unique_models:
        model_data = melted_df[melted_df["model"] == model]
        model_df = df1[df1["model"] == model]
        model_means = model_data.groupby("variable")["value"].mean().reindex(cols)
        marker = "x" if _is_closed_model(model) else "o"
        offset = model_offsets.get(model, 0)

        model_cis = []
        for col in cols:
            if len(model_df) > 1:
                ci_lo, ci_hi = cluster_bootstrap_ci(model_df, col, cluster_col='instance_id', n_bootstrap=500, ci=0.95)
            else:
                ci_lo, ci_hi = model_means[col], model_means[col]
            model_cis.append((ci_lo, ci_hi))

        model_yerr_lower = [max(0, model_means.values[i] - model_cis[i][0]) for i in range(len(cols))]
        model_yerr_upper = [max(0, model_cis[i][1] - model_means.values[i]) for i in range(len(cols))]

        x_jittered = [x + offset for x in x_positions]

        ax_acc.errorbar(
            x_jittered, model_means.values, yerr=[model_yerr_lower, model_yerr_upper],
            fmt=marker, markersize=10, color=palette_map[model], alpha=0.8,
            linewidth=0, elinewidth=1.5, capsize=3, capthick=1.2,
        )

        for i, val in enumerate(model_means.values):
            all_model_values[i].append(val)
            all_model_values[i].append(model_cis[i][1])

    # Compute and plot closed/open means
    df_closed = df1[df1["model"].isin(closed_models)]
    df_open = df1[df1["model"].isin(open_models)]

    closed_stats = []
    for col in cols:
        mean_val = df_closed[col].mean() if len(df_closed) > 0 else np.nan
        ci_lo, ci_hi = cluster_bootstrap_ci(df_closed, col, cluster_col='instance_id', n_bootstrap=1000, ci=0.95)
        closed_stats.append({"mean": mean_val, "ci_lower": ci_lo, "ci_upper": ci_hi})

    open_stats = []
    for col in cols:
        mean_val = df_open[col].mean() if len(df_open) > 0 else np.nan
        ci_lo, ci_hi = cluster_bootstrap_ci(df_open, col, cluster_col='instance_id', n_bootstrap=1000, ci=0.95)
        open_stats.append({"mean": mean_val, "ci_lower": ci_lo, "ci_upper": ci_hi})

    closed_means = np.array([s["mean"] for s in closed_stats])
    closed_yerr_lower = np.maximum(0, closed_means - np.array([s["ci_lower"] for s in closed_stats]))
    closed_yerr_upper = np.maximum(0, np.array([s["ci_upper"] for s in closed_stats]) - closed_means)

    ax_acc.errorbar(
        [x - 0.12 for x in x_positions], closed_means, yerr=[closed_yerr_lower, closed_yerr_upper],
        fmt="s-", color="steelblue", markersize=8, linewidth=2,
        capsize=4, capthick=1.5,
    )

    open_means = np.array([s["mean"] for s in open_stats])
    open_yerr_lower = np.maximum(0, open_means - np.array([s["ci_lower"] for s in open_stats]))
    open_yerr_upper = np.maximum(0, np.array([s["ci_upper"] for s in open_stats]) - open_means)

    ax_acc.errorbar(
        [x + 0.12 for x in x_positions], open_means, yerr=[open_yerr_lower, open_yerr_upper],
        fmt="^-", color="darkorange", markersize=8, linewidth=2,
        capsize=4, capthick=1.5,
    )

    # Overall stats for annotations
    overall_means = np.array([df1[col].mean() for col in cols])

    # Cochran's Q
    cochrans_q_test(df1, cols)

    # McNemar pairwise
    adjacent_pairs = [(cols[i], cols[i + 1]) for i in range(len(cols) - 1)]
    mcnemar_results = []
    p_values_raw = []

    for col_a, col_b in adjacent_pairs:
        result = mcnemar_one_sided(df1, col_a, col_b)
        mcnemar_results.append(result)
        p_values_raw.append(result['p_value'])

    _, p_values_holm, _, _ = multipletests(p_values_raw, method='holm')

    # Add p-value annotations
    max_data_y = max(max(all_model_values[i]) for i in range(len(cols)) if all_model_values[i])
    col_to_x = {col: i for i, col in enumerate(cols)}
    p_y_start = max_data_y + 0.05
    p_y_step = 0.05

    for i, (pair, p_holm) in enumerate(zip(adjacent_pairs, p_values_holm)):
        col1, col2 = pair
        x1, x2 = col_to_x[col1], col_to_x[col2]
        y = p_y_start + i * p_y_step
        h = 0.015
        ax_acc.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="steelblue")

        sig = "***" if p_holm < 0.001 else "**" if p_holm < 0.01 else "*" if p_holm < 0.05 else "ns"
        p_text = f"p<.001 {sig}" if p_holm < 0.001 else f"p={p_holm:.2f} {sig}"
        ax_acc.text((x1 + x2) * 0.5, y + h + 0.01, p_text, ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Mean annotations
    for i, (x, mean_val) in enumerate(zip(x_positions, overall_means)):
        y_pos = max(closed_means[i], open_means[i]) + max(closed_yerr_upper[i], open_yerr_upper[i]) + 0.02
        ax_acc.annotate(f"{mean_val:.1%}", xy=(x, y_pos), ha="center", va="bottom", fontsize=12, fontweight="bold")

    # (Cochran's Q removed per request)

    ax_acc.set_xticks(x_positions)
    ax_acc.set_xticklabels(arm_labels, fontsize=13, fontweight='bold')
    ax_acc.set_xlabel("Route", fontsize=14, fontweight='bold')
    ax_acc.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
    ax_acc.tick_params(axis='y', labelsize=11)

    y_max = p_y_start + len(adjacent_pairs) * p_y_step + 0.07
    ax_acc.set_ylim([0.08, max(0.65, y_max)])
    ax_acc.set_title("(A) Accuracy Across Routes", fontsize=14, fontweight="bold")

    # Build legend for accuracy panel
    handles_custom = []
    labels_custom = []

    for model in closed_models:
        handles_custom.append(Line2D([0], [0], marker="x", color=palette_map[model], linestyle="", markersize=7, markeredgewidth=1.5))
        labels_custom.append(model.split("/")[-1] if "/" in model else model)

    for model in open_models:
        handles_custom.append(Line2D([0], [0], marker="o", color=palette_map[model], linestyle="", markersize=7, markeredgewidth=1.5))
        labels_custom.append(model.split("/")[-1] if "/" in model else model)

    handles_custom.append(Line2D([0], [0], marker="s", color="steelblue", linestyle="-", markersize=7, linewidth=1.5))
    labels_custom.append("Closed ± 95% CI")
    handles_custom.append(Line2D([0], [0], marker="^", color="darkorange", linestyle="-", markersize=7, linewidth=1.5))
    labels_custom.append("Open ± 95% CI")

    ax_acc.legend(handles_custom, labels_custom, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=4, fontsize=7, handlelength=1.5, columnspacing=0.8, framealpha=0.95)

    # === RIGHT PANELS: Delta distributions ===

    # Delta 1: Sim - NL
    print("[plot_combined] Computing delta distributions (instance-level bootstrap)...")
    delta_sim_nl = cluster_bootstrap_delta(df1, 'nl_correct', 'sim_correct', n_bootstrap=2000)
    boot_vals_1 = delta_sim_nl['boot_deltas'] * 100

    # Use viridis colors for deltas (avoid blue/orange used for closed/open)
    viridis = plt.cm.viridis
    delta1_color = viridis(0.3)  # Teal-ish
    delta2_color = viridis(0.7)  # Yellow-green

    sns.histplot(boot_vals_1, kde=True, ax=ax_delta1, color=delta1_color, alpha=0.7, stat='density')
    ax_delta1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax_delta1.axvline(delta_sim_nl['delta'] * 100, color='black', linestyle='-', linewidth=2, label='Observed')
    ax_delta1.axvline(delta_sim_nl['delta_ci_low'] * 100, color='gray', linestyle=':', linewidth=1.5)
    ax_delta1.axvline(delta_sim_nl['delta_ci_high'] * 100, color='gray', linestyle=':', linewidth=1.5)

    ax_delta1.set_xlabel('Δ (Sim − NL) [%]', fontsize=10, fontweight='bold')
    ax_delta1.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax_delta1.set_title(f'(B) Δ = {delta_sim_nl["delta"]*100:+.2f}% [{delta_sim_nl["delta_ci_low"]*100:.2f}, {delta_sim_nl["delta_ci_high"]*100:.2f}]',
                        fontsize=10, fontweight='bold')
    ax_delta1.legend(loc='upper right', fontsize=7)
    ax_delta1.tick_params(labelsize=9)

    # Delta 2: Code Exec - Sim (zoomed in scale to see distribution clearly)
    delta_code_sim = cluster_bootstrap_delta(df1, 'sim_correct', 'code_correct', n_bootstrap=2000)
    boot_vals_2 = delta_code_sim['boot_deltas'] * 100

    sns.histplot(boot_vals_2, kde=True, ax=ax_delta2, color=delta2_color, alpha=0.7, stat='density')
    ax_delta2.axvline(delta_code_sim['delta'] * 100, color='black', linestyle='-', linewidth=2, label='Observed')
    ax_delta2.axvline(delta_code_sim['delta_ci_low'] * 100, color='gray', linestyle=':', linewidth=1.5)
    ax_delta2.axvline(delta_code_sim['delta_ci_high'] * 100, color='gray', linestyle=':', linewidth=1.5)

    # Zoom in on the distribution - center around observed delta with margin
    delta_center = delta_code_sim['delta'] * 100
    delta_range = (delta_code_sim['delta_ci_high'] - delta_code_sim['delta_ci_low']) * 100
    x_margin = max(delta_range * 2, 3)  # At least 3% margin
    ax_delta2.set_xlim(delta_center - x_margin, delta_center + x_margin)

    ax_delta2.set_xlabel('Δ (Code Exec − Sim) [%]', fontsize=10, fontweight='bold')
    ax_delta2.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax_delta2.set_title(f'(C) Δ = {delta_code_sim["delta"]*100:+.2f}% [{delta_code_sim["delta_ci_low"]*100:.2f}, {delta_code_sim["delta_ci_high"]*100:.2f}]',
                        fontsize=10, fontweight='bold')
    ax_delta2.legend(loc='upper left', fontsize=7)
    ax_delta2.tick_params(labelsize=9)

    print(f"[plot_combined] Delta (Sim - NL): {delta_sim_nl['delta']*100:+.2f}% [{delta_sim_nl['delta_ci_low']*100:.2f}, {delta_sim_nl['delta_ci_high']*100:.2f}]")
    print(f"[plot_combined] Delta (Code Exec - Sim): {delta_code_sim['delta']*100:+.2f}% [{delta_code_sim['delta_ci_low']*100:.2f}, {delta_code_sim['delta_ci_high']*100:.2f}]")

    plt.tight_layout()
    plt.savefig("figures/combined_accuracy_delta.png", bbox_inches="tight", pad_inches=0.05)
    plt.savefig("figures/combined_accuracy_delta.pdf", bbox_inches="tight", pad_inches=0.05)
    print("[plot_combined] Saved figures/combined_accuracy_delta.png and .pdf")


def wilcoxon_test(mdf: pd.DataFrame, complexity_pairs: List[tuple[str, str]]) -> List[float]:
    # Perform Wilcoxon test for each pair
    p_values = []
    for pair in complexity_pairs:
        complexity1 = pair[0]
        complexity2 = pair[1]
        x = mdf[mdf["variable"] == complexity1]["value"]
        y = mdf[mdf["variable"] == complexity2]["value"]
        try:
            stat, p_value = wilcoxon(x, y)
        except Exception:
            p_value = 1.0
        p_values.append(p_value)
    return p_values


def plot_p_vals(df: pd.DataFrame) -> None:
    from matplotlib import rcParams

    rcParams["figure.dpi"] = 500
    rcParams["savefig.dpi"] = 500
    rcParams["font.family"] = "Arial"
    rcParams["axes.labelsize"] = 18
    rcParams["axes.titlesize"] = 18
    rcParams["legend.fontsize"] = 18
    rcParams["figure.titlesize"] = 18
    rcParams["markers.fillstyle"] = "none"
    # df1 = df[df["model"] == "Qwen/Qwen2.5-14B-Instruct"]
    df1 = df
    # df1 = df[df["model"].isin(["Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501"])]
    df2 = df1[df1["kind"].isin(["gsm8k"])]
    name_map = {
        "nl_correct": "Arm 1 \n (NL)",
        "sim_correct": "Arm 2 \n (Code Sim)",
        "controlsim_correct": "Arm 2.5 \n (Controlled Code Sim)",
        "code_correct": "Arm 3 \n (Code Exec)",
    }
    dfnew = df2.rename(columns=name_map)

    # Debug: inspect incoming data
    print("[plot_p_vals] raw df shape:", df.shape)
    print("[plot_p_vals] models:", df["model"].unique())
    print("[plot_p_vals] kinds:", df["kind"].unique())
    print("[plot_p_vals] sample rows:\n", df.head(5))

    cols = list(name_map.values())
    mdf = pd.melt(dfnew, value_vars=cols, id_vars=["model", "kind"])
    fig, ax = plt.subplots(figsize=(6, 6))
    # mdf = mdf.sort_values(by=["variable"], key=lambda x: x.map({color: ind for color,ind in zip(range(len(colors)), list(colors.keys()))}))
    # import pdb; pdb.set_trace()
    mdf1 = mdf.groupby(["variable", "model", "kind"]).mean().reset_index()
    mdf2 = mdf1.drop(["model", "kind"], axis=1)

    # Drop non-finite values to avoid NaN/inf in downstream plotting
    print("[plot_p_vals] grouped sample before finite filter:\n", mdf2.head(10))
    mdf2 = mdf2[np.isfinite(mdf2["value"])]
    print("[plot_p_vals] grouped shape after finite filter:", mdf2.shape)
    print("[plot_p_vals] grouped sample after finite filter:\n", mdf2.head(10))
    if mdf2.empty:
        print("[plot_p_vals] No finite values available for p-value plot after filtering.")
        raise ValueError("No finite values available for p-value plot.")

    arm_pairs = list(itertools.combinations(cols, 2))
    # import pdb; pdb.set_trace()
    p_values = wilcoxon_test(mdf2, arm_pairs)
    sns.boxplot(x="variable", y="value", data=mdf2, gap=0.3, palette=sns.color_palette("vlag", n_colors=8)[:4], ax=ax)

    offset = 0
    max_val = mdf2["value"].max()
    if pd.isna(max_val):
        max_val = 0.0
    for i, pair in enumerate(arm_pairs):
        complexity1 = pair[0]
        complexity2 = pair[1]
        order = {x: i for i, x in enumerate(cols)}
        x1 = order[complexity1]
        x2 = order[complexity2]
        y = max_val + offset
        h = 0.03
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="steelblue")
        ax.text((x1 + x2) * 0.5, y + h, f"p={p_values[i]:.4f}", ha="center", va="bottom")
        offset += 0.08  # type: ignore[assignment]

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # Set y-axis ticks
    ax.set_xlabel("Arms")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, max(1.0, y + 0.5))
    plt.ylim(0, 1.5)
    plt.title("Accuracy across Arms on Fine-grained Tasks")
    plt.tight_layout()
    plt.show()
    plt.savefig("figures/pval.png", bbox_inches="tight")


models = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/gpt-5.1-codex",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/o3-mini",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-distill-llama-70b",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "xai/grok-code-fast-1",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-4-scout",
    "mistral/devstral-medium",
    "mistral/ministral-14b-2512",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "zhipuai/glm-4.6",
    "allenai/olmo-2-0325-32b-instruct",
    "microsoft/phi-4",
    "microsoft/phi-4-reasoning-plus",
]


def analysis() -> None:
    results_root = Path(__file__).parent / "results"
    jsonl_files = sorted(results_root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under {results_root}")
    df = create_big_df(jsonl_files)
    # Target models from scaled experiment (plan-1.md)
    df = df[
        df["model"].isin(
            [
                "mistralai/codestral-2508",
                "mistralai/mistral-large-2411",
                "google/gemini-2.0-flash-001",
                "mistralai/mixtral-8x22b-instruct",
            ]
        )
    ]
    plot_p_vals(df)
    plot_main_fig(df)
    plot_v_graph(df)

    # rows = df.to_dict("records")
    # import pdb

    # pdb.set_trace()
    # return rows


def analysis_closed_models() -> None:
    """Run analysis on closed models only with the new line_closed figure."""
    # Use local results directory
    results_root = Path(__file__).parent / "results"
    jsonl_files = sorted(results_root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under {results_root}")
    df = create_big_df(jsonl_files)
    print(f"[analysis_closed_models] Loaded {len(df)} rows from {len(jsonl_files)} files")
    print(f"[analysis_closed_models] Models: {df['model'].unique()}")
    print(f"[analysis_closed_models] Kinds: {df['kind'].unique()}")

    # Generate the closed models figure
    plot_v_graph_closed(df)


def analysis_all_models(parse_error_threshold: float = 50.0) -> None:
    """
    Run analysis on all models with parse error filtering.
    Generates three figures: closed only, open only, and combined.

    Args:
        parse_error_threshold: Exclude models with >threshold% parse error on any arm.
    """
    # Use local results directory
    results_root = Path(__file__).parent / "results"
    jsonl_files = sorted(results_root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under {results_root}")

    df = create_big_df(jsonl_files)
    print(f"[analysis_all_models] Loaded {len(df)} rows from {len(jsonl_files)} files")
    print(f"[analysis_all_models] Models before filtering: {sorted(df['model'].unique())}")

    # Apply parse error filtering
    df_filtered, excluded = filter_models_by_parse_error(df, threshold=parse_error_threshold)
    print(f"[analysis_all_models] Excluded {len(excluded)} models with >{parse_error_threshold}% parse error")
    if excluded:
        print(f"[analysis_all_models] Excluded models: {excluded}")

    # Exclude specific models from analysis
    EXCLUDED_MODELS = {
        "mistralai/ministral-14b-2512",
        "qwen/qwen-2.5-coder-32b-instruct",
        "anthropic/claude-opus-4",
        "deepseek/deepseek-chat-v3-0324",
    }
    df_filtered = df_filtered[~df_filtered["model"].isin(EXCLUDED_MODELS)]
    print(f"[analysis_all_models] Excluded {len(EXCLUDED_MODELS)} specific models: {EXCLUDED_MODELS}")
    print(f"[analysis_all_models] Models after filtering: {sorted(df_filtered['model'].unique())}")

    # Generate all three figures
    print("\n=== Generating Closed Models Figure ===")
    plot_v_graph_closed(df_filtered)

    print("\n=== Generating Open Models Figure ===")
    plot_v_graph_open(df_filtered)

    print("\n=== Generating Combined Figure ===")
    plot_v_graph_all(df_filtered)

    print("\n=== Generating Combined Accuracy + Delta Figure ===")
    plot_combined_accuracy_delta(df_filtered)

    print("\n=== Running GLMM Analysis ===")
    glmm_result = run_glmm_analysis(df_filtered)

    print("\n=== Generating Main Combined Figure ===")
    plot_main_combined(df_filtered, glmm_result)

    print("\n[analysis_all_models] Done! Generated figures:")
    print("  - figures/line_closed.png")
    print("  - figures/line_open.png")
    print("  - figures/line_all.png")
    print("  - figures/combined_accuracy_delta.png")
    print("  - figures/glmm_predicted_prob.png")
    print("  - figures/main_combined.png")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate reasoning benchmark reports and figures.")
    parser.add_argument("--parse-error-threshold", type=float, default=50.0)
    args = parser.parse_args()
    analysis_all_models(parse_error_threshold=args.parse_error_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
