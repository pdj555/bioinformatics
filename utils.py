"""Shared utilities for bioinformatics notebooks."""

import numpy as np
import pandas as pd


def label_bioactivity_class(standard_values, active_nM=1000, inactive_nM=10000):
    """Label compounds as active (<=active_nM), inactive (>=inactive_nM), or intermediate."""
    out = []
    for v in standard_values:
        x = float(v)
        if x >= inactive_nM:
            out.append("inactive")
        elif x <= active_nM:
            out.append("active")
        else:
            out.append("intermediate")
    return pd.Series(out, name="class")


def lipinski(smiles, verbose=False):
    """Compute Lipinski descriptors (MW, LogP, NumHDonors, NumHAcceptors) from SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    moldata = [Chem.MolFromSmiles(s) for s in smiles]
    baseData = None
    for i, mol in enumerate(moldata):
        if mol is None:
            continue
        row = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
        ])
        if baseData is None:
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
    if baseData is None:
        return pd.DataFrame(columns=["MW", "LogP", "NumHDonors", "NumHAcceptors"])
    return pd.DataFrame(data=baseData, columns=["MW", "LogP", "NumHDonors", "NumHAcceptors"])


def norm_value(input_df, cap_nM=100_000_000):
    """Cap standard_value at cap_nM and add standard_value_norm column."""
    norm = [min(float(i), cap_nM) for i in input_df["standard_value"]]
    out = input_df.copy()
    out["standard_value_norm"] = norm
    return out.drop(columns="standard_value")


def pIC50_from_norm(input_df):
    """Convert standard_value_norm (nM) to pIC50 column. Drops standard_value_norm."""
    out = input_df.copy()
    out["pIC50"] = [-np.log10(v * 1e-9) for v in out["standard_value_norm"]]
    return out.drop(columns="standard_value_norm")


def mannwhitney(descriptor, df, alpha=0.05, filename=None):
    """Mann-Whitney U test for descriptor between active and inactive classes."""
    from scipy.stats import mannwhitneyu

    sel = df[[descriptor, "class"]]
    active = sel[sel["class"] == "active"][descriptor]
    inactive = sel[sel["class"] == "inactive"][descriptor]
    stat, p = mannwhitneyu(active, inactive)
    interpretation = (
        "Same distribution (fail to reject H0)"
        if p > alpha
        else "Different distribution (reject H0)"
    )
    results = pd.DataFrame({
        "Descriptor": [descriptor],
        "Statistics": [stat],
        "p": [p],
        "alpha": [alpha],
        "Interpretation": [interpretation],
    })
    if filename is None:
        filename = f"mannwhitneyu_{descriptor}.csv"
    results.to_csv(filename, index=False)
    return results


def remove_low_variance_features(X, threshold=0.16):
    """Remove features with variance below threshold (default 0.8*(1-0.8)=0.16)."""
    from sklearn.feature_selection import VarianceThreshold

    sel = VarianceThreshold(threshold=threshold)
    return sel.fit_transform(X)


def plot_regression_fit(y_true, y_pred, xlim=(0, 12), ylim=(0, 12)):
    """Scatter plot of experimental vs predicted values with regression line."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(color_codes=True)
    sns.set_style("white")
    ax = sns.regplot(x=y_true, y=y_pred, scatter_kws={"alpha": 0.4})
    ax.set_xlabel("Experimental pIC50", fontsize="large", fontweight="bold")
    ax.set_ylabel("Predicted pIC50", fontsize="large", fontweight="bold")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.figure.set_size_inches(5, 5)
    plt.show()
    return ax
