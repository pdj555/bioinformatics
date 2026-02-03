"""Shared utilities for bioactivity and ML notebooks."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import VarianceThreshold


# Bioactivity thresholds (nM)
ACTIVE_THRESHOLD_NM = 1000
INACTIVE_THRESHOLD_NM = 10000
NORM_CAP_NM = 100_000_000


def bioactivity_class_from_ic50(standard_value_series):
    """Label IC50 (nM) as 'active' (<1000), 'inactive' (>10000), or 'intermediate'."""
    labels = []
    for v in standard_value_series:
        v = float(v)
        if v >= INACTIVE_THRESHOLD_NM:
            labels.append("inactive")
        elif v <= ACTIVE_THRESHOLD_NM:
            labels.append("active")
        else:
            labels.append("intermediate")
    return pd.Series(labels, name="class", index=standard_value_series.index)


def norm_value(df, value_col="standard_value", cap_nm=NORM_CAP_NM):
    """Cap IC50 at cap_nm (nM), add standard_value_norm, drop value_col. Returns new DataFrame."""
    norm = df[value_col].clip(upper=cap_nm)
    out = df.drop(columns=[value_col]).assign(standard_value_norm=norm)
    return out


def add_pIC50(df, norm_col="standard_value_norm"):
    """Convert nM (norm_col) to pIC50 = -log10(M). Drops norm_col. Returns new DataFrame."""
    molar = df[norm_col] * 1e-9
    pic50 = -np.log10(molar)
    return df.drop(columns=[norm_col]).assign(pIC50=pic50)


def ic50_to_pIC50(df, value_col="standard_value", cap_nm=NORM_CAP_NM):
    """Norm IC50 (cap at cap_nm) then add pIC50. Drops value_col. Returns new DataFrame."""
    return add_pIC50(norm_value(df, value_col=value_col, cap_nm=cap_nm))


def mannwhitney_test(df, descriptor, class_col="class", filename=None):
    """Mann-Whitney U: active vs inactive for descriptor. Returns results DataFrame."""
    sel = df[[descriptor, class_col]]
    active = sel.loc[sel[class_col] == "active", descriptor]
    inactive = sel.loc[sel[class_col] == "inactive", descriptor]
    stat, p = mannwhitneyu(active, inactive)
    alpha = 0.05
    interpretation = (
        "Different distribution (reject H0)"
        if p <= alpha
        else "Same distribution (fail to reject H0)"
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
    if filename is None:
        filename = f"mannwhitneyu_{descriptor}.csv"
    results.to_csv(filename, index=False)
    return results


def remove_low_variance_features(X, threshold=0.8 * (1 - 0.8)):
    """Drop features with variance below threshold. X: DataFrame or array. Returns array."""
    selector = VarianceThreshold(threshold=threshold)
    return selector.fit_transform(X)


def plot_experimental_vs_predicted(y_test, y_pred, ax=None):
    """Scatter + regression line: experimental vs predicted pIC50. Compatible with seaborn regplot API."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(color_codes=True)
    sns.set_style("white")
    ax = sns.regplot(x=y_test, y=y_pred, scatter_kws={"alpha": 0.4}, ax=ax)
    ax.set_xlabel("Experimental pIC50", fontsize="large", fontweight="bold")
    ax.set_ylabel("Predicted pIC50", fontsize="large", fontweight="bold")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.figure.set_size_inches(5, 5)
    plt.show()
    return ax
