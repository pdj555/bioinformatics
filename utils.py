"""Shared utilities for CDD ML notebooks (bioactivity, pIC50, stats, ML helpers)."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# --- Bioactivity labeling (Part 1) ---
ACTIVE_THRESHOLD_NM = 1000
INACTIVE_THRESHOLD_NM = 10000


def bioactivity_class(ic50_series):
    """Label IC50 (nM) as active (<=1e3), inactive (>=1e4), or intermediate."""
    def _label(v):
        v = float(v)
        if v >= INACTIVE_THRESHOLD_NM:
            return "inactive"
        if v <= ACTIVE_THRESHOLD_NM:
            return "active"
        return "intermediate"
    return pd.Series([_label(x) for x in ic50_series], name="class")


# --- pIC50 conversion (Part 2) ---
IC50_CAP_NM = 100_000_000  # cap so -log10(molar) stays non-negative


def norm_value(df, value_col="standard_value"):
    """Cap IC50 at 100M nM and add standard_value_norm column."""
    norm = df[value_col].clip(upper=IC50_CAP_NM)
    out = df.drop(columns=[value_col]).assign(standard_value_norm=norm)
    return out


def add_pic50(df, value_col="standard_value_norm"):
    """Convert normalized IC50 (nM) to pIC50; drop value_col."""
    molar = df[value_col] * 1e-9
    pic50 = -np.log10(molar)
    return df.drop(columns=[value_col]).assign(pIC50=pic50)


# --- Mann-Whitney U (Part 2) ---
def mannwhitney_test(
    df,
    descriptor,
    class_col="class",
    active_label="active",
    inactive_label="inactive",
    alpha=0.05,
    save_path=None,
):
    """Mann-Whitney U for active vs inactive. Returns result DataFrame; optionally saves CSV."""
    sel = df[[descriptor, class_col]]
    active = sel.loc[sel[class_col] == active_label, descriptor]
    inactive = sel.loc[sel[class_col] == inactive_label, descriptor]
    stat, p = mannwhitneyu(active, inactive)
    interpretation = (
        "Same distribution (fail to reject H0)"
        if p > alpha
        else "Different distribution (reject H0)"
    )
    results = pd.DataFrame(
        {
            "Descriptor": [descriptor],
            "Statistics": [stat],
            "p": [p],
            "alpha": [alpha],
            "Interpretation": [interpretation],
        }
    )
    if save_path:
        results.to_csv(save_path, index=False)
    return results


# --- ML feature / plot helpers (Parts 4, 5) ---
VARIANCE_THRESHOLD = 0.8 * (1 - 0.8)  # 0.16


def remove_low_variance(X, threshold=VARIANCE_THRESHOLD):
    """Drop features with variance below threshold. Returns array."""
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=threshold)
    return selector.fit_transform(X)


def plot_exp_vs_pred(y_test, y_pred, xlabel="Experimental pIC50", ylabel="Predicted pIC50"):
    """Scatter of experimental vs predicted values; returns axes."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_style("white")
    ax = sns.regplot(x=y_test, y=y_pred, scatter_kws={"alpha": 0.4})
    ax.set_xlabel(xlabel, fontsize="large", fontweight="bold")
    ax.set_ylabel(ylabel, fontsize="large", fontweight="bold")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.figure.set_size_inches(5, 5)
    plt.show()
    return ax
