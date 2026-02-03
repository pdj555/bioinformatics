"""Unit tests for utils.py."""

import numpy as np
import pandas as pd
import pytest


class TestLabelBioactivityClass:
    """Tests for label_bioactivity_class function."""

    def test_labels_active_when_below_threshold(self):
        """Values <= 1000 nM should be labeled 'active'."""
        from utils import label_bioactivity_class

        values = [100, 500, 1000]
        result = label_bioactivity_class(values)

        assert list(result) == ["active", "active", "active"]

    def test_labels_inactive_when_above_threshold(self):
        """Values >= 10000 nM should be labeled 'inactive'."""
        from utils import label_bioactivity_class

        values = [10000, 50000, 100000]
        result = label_bioactivity_class(values)

        assert list(result) == ["inactive", "inactive", "inactive"]

    def test_labels_intermediate_between_thresholds(self):
        """Values between 1000 and 10000 nM should be labeled 'intermediate'."""
        from utils import label_bioactivity_class

        values = [1001, 5000, 9999]
        result = label_bioactivity_class(values)

        assert list(result) == ["intermediate", "intermediate", "intermediate"]

    def test_returns_pandas_series_with_class_name(self):
        """Result should be a pandas Series named 'class'."""
        from utils import label_bioactivity_class

        result = label_bioactivity_class([500])

        assert isinstance(result, pd.Series)
        assert result.name == "class"

    def test_custom_thresholds(self):
        """Should use custom active_nM and inactive_nM thresholds."""
        from utils import label_bioactivity_class

        values = [50, 150, 500]
        result = label_bioactivity_class(values, active_nM=100, inactive_nM=200)

        assert list(result) == ["active", "intermediate", "inactive"]

    def test_handles_string_values(self):
        """Should convert string values to float."""
        from utils import label_bioactivity_class

        values = ["500", "5000", "50000"]
        result = label_bioactivity_class(values)

        assert list(result) == ["active", "intermediate", "inactive"]


class TestNormValue:
    """Tests for norm_value function."""

    def test_caps_values_at_default_threshold(self):
        """Values above 100M nM should be capped at 100M."""
        from utils import norm_value

        df = pd.DataFrame({"standard_value": [1000, 200_000_000]})
        result = norm_value(df)

        assert list(result["standard_value_norm"]) == [1000, 100_000_000]

    def test_removes_original_standard_value_column(self):
        """Original standard_value column should be dropped."""
        from utils import norm_value

        df = pd.DataFrame({"standard_value": [1000], "other": ["a"]})
        result = norm_value(df)

        assert "standard_value" not in result.columns
        assert "standard_value_norm" in result.columns
        assert "other" in result.columns

    def test_custom_cap_threshold(self):
        """Should use custom cap_nM threshold."""
        from utils import norm_value

        df = pd.DataFrame({"standard_value": [500, 1500]})
        result = norm_value(df, cap_nM=1000)

        assert list(result["standard_value_norm"]) == [500, 1000]

    def test_preserves_other_columns(self):
        """Should preserve all other columns in the DataFrame."""
        from utils import norm_value

        df = pd.DataFrame({
            "standard_value": [100],
            "molecule_chembl_id": ["CHEMBL123"],
            "canonical_smiles": ["CCO"],
        })
        result = norm_value(df)

        assert "molecule_chembl_id" in result.columns
        assert "canonical_smiles" in result.columns
        assert result["molecule_chembl_id"].iloc[0] == "CHEMBL123"


class TestPIC50FromNorm:
    """Tests for pIC50_from_norm function."""

    def test_converts_nm_to_pic50(self):
        """1 nM should give pIC50 of 9, 1000 nM should give pIC50 of 6."""
        from utils import pIC50_from_norm

        df = pd.DataFrame({"standard_value_norm": [1, 1000, 1_000_000]})
        result = pIC50_from_norm(df)

        # pIC50 = -log10(nM * 1e-9) = -log10(M) = 9 - log10(nM)
        expected = [9.0, 6.0, 3.0]
        np.testing.assert_array_almost_equal(result["pIC50"], expected)

    def test_removes_standard_value_norm_column(self):
        """standard_value_norm column should be dropped."""
        from utils import pIC50_from_norm

        df = pd.DataFrame({"standard_value_norm": [100], "other": ["a"]})
        result = pIC50_from_norm(df)

        assert "standard_value_norm" not in result.columns
        assert "pIC50" in result.columns
        assert "other" in result.columns

    def test_preserves_other_columns(self):
        """Should preserve all other columns in the DataFrame."""
        from utils import pIC50_from_norm

        df = pd.DataFrame({
            "standard_value_norm": [100],
            "molecule_chembl_id": ["CHEMBL123"],
        })
        result = pIC50_from_norm(df)

        assert result["molecule_chembl_id"].iloc[0] == "CHEMBL123"


class TestLipinski:
    """Tests for lipinski function (requires RDKit)."""

    def test_computes_lipinski_descriptors_for_valid_smiles(self):
        """Should compute MW, LogP, NumHDonors, NumHAcceptors for valid SMILES."""
        from utils import lipinski

        # Ethanol (CCO) and Methanol (CO)
        smiles = ["CCO", "CO"]
        result = lipinski(smiles)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
        assert len(result) == 2
        # Ethanol: MW ~46, 1 H-donor (OH), 1 H-acceptor (O)
        assert result["MW"].iloc[0] == pytest.approx(46.07, rel=0.01)
        assert result["NumHDonors"].iloc[0] == 1
        assert result["NumHAcceptors"].iloc[0] == 1

    def test_single_smiles_input(self):
        """Should handle single SMILES input correctly."""
        from utils import lipinski

        # Note: Currently known to fail due to array shape bug in utils.py
        # This test documents the expected behavior
        smiles = ["CCO"]
        result = lipinski(smiles)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert list(result.columns) == ["MW", "LogP", "NumHDonors", "NumHAcceptors"]

    def test_handles_multiple_smiles(self):
        """Should compute descriptors for multiple SMILES."""
        from utils import lipinski

        smiles = ["CCO", "CC(=O)O"]  # Ethanol, Acetic acid
        result = lipinski(smiles)

        assert len(result) == 2

    def test_skips_invalid_smiles(self):
        """Invalid SMILES should be skipped, not cause errors."""
        from utils import lipinski

        smiles = ["CCO", "INVALID_SMILES", "CC"]
        result = lipinski(smiles)

        # Should have 2 rows (valid ones)
        assert len(result) == 2

    def test_returns_empty_dataframe_for_all_invalid(self):
        """Should return empty DataFrame with correct columns if all SMILES are invalid."""
        from utils import lipinski

        smiles = ["INVALID1", "INVALID2"]
        result = lipinski(smiles)

        assert len(result) == 0
        assert list(result.columns) == ["MW", "LogP", "NumHDonors", "NumHAcceptors"]


class TestMannwhitney:
    """Tests for mannwhitney function."""

    def test_performs_mannwhitneyu_test(self, tmp_path):
        """Should perform Mann-Whitney U test between active and inactive classes."""
        from utils import mannwhitney

        df = pd.DataFrame({
            "pIC50": [9.0, 8.5, 8.0, 4.0, 3.5, 3.0],
            "class": ["active", "active", "active", "inactive", "inactive", "inactive"],
        })
        output_file = tmp_path / "mannwhitneyu_pIC50.csv"
        result = mannwhitney("pIC50", df, filename=str(output_file))

        assert isinstance(result, pd.DataFrame)
        assert "Descriptor" in result.columns
        assert "Statistics" in result.columns
        assert "p" in result.columns
        assert result["Descriptor"].iloc[0] == "pIC50"

    def test_writes_results_to_csv(self, tmp_path):
        """Should write results to CSV file."""
        from utils import mannwhitney

        df = pd.DataFrame({
            "descriptor": [9.0, 8.5, 4.0, 3.5],
            "class": ["active", "active", "inactive", "inactive"],
        })
        output_file = tmp_path / "test_output.csv"
        mannwhitney("descriptor", df, filename=str(output_file))

        assert output_file.exists()
        saved = pd.read_csv(output_file)
        assert "Descriptor" in saved.columns

    def test_interpretation_different_distribution(self, tmp_path):
        """Should interpret p < alpha as 'Different distribution'."""
        from utils import mannwhitney

        # Larger sample with clear separation for significant p-value
        df = pd.DataFrame({
            "value": [100, 101, 102, 103, 104, 1, 2, 3, 4, 5],
            "class": ["active"] * 5 + ["inactive"] * 5,
        })
        output_file = tmp_path / "mannwhitneyu_value.csv"
        result = mannwhitney("value", df, filename=str(output_file))

        assert "Different distribution" in result["Interpretation"].iloc[0]

    def test_interpretation_same_distribution(self, tmp_path):
        """Should interpret p > alpha as 'Same distribution'."""
        from utils import mannwhitney

        # Similar values should give p > 0.05
        df = pd.DataFrame({
            "value": [5, 6, 7, 5, 6, 7],
            "class": ["active", "active", "active", "inactive", "inactive", "inactive"],
        })
        output_file = tmp_path / "mannwhitneyu_value.csv"
        result = mannwhitney("value", df, alpha=0.05, filename=str(output_file))

        assert "Same distribution" in result["Interpretation"].iloc[0]


class TestRemoveLowVarianceFeatures:
    """Tests for remove_low_variance_features function."""

    def test_removes_low_variance_columns(self):
        """Should remove features with variance below threshold."""
        from utils import remove_low_variance_features

        # First column: all same values (variance = 0)
        # Second column: varying values (high variance)
        X = np.array([
            [1, 10],
            [1, 20],
            [1, 30],
            [1, 40],
        ])
        result = remove_low_variance_features(X, threshold=0.0)

        # Should keep only the high-variance column
        assert result.shape[1] == 1

    def test_keeps_high_variance_features(self):
        """Should keep features with variance above threshold."""
        from utils import remove_low_variance_features

        X = np.array([
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
        ])
        result = remove_low_variance_features(X, threshold=0.1)

        # Both columns have variance > 0.1, should keep both
        assert result.shape[1] == 2

    def test_default_threshold(self):
        """Default threshold should be 0.16 (0.8 * 0.2)."""
        from utils import remove_low_variance_features

        # Create data where one feature has variance just below 0.16
        # and another just above
        X = np.array([
            [0.9, 0.1],
            [0.9, 0.9],
            [0.9, 0.1],
            [0.9, 0.9],
        ])
        result = remove_low_variance_features(X)

        # First column has near-zero variance, second has variance = 0.16
        assert result.shape[1] == 1
