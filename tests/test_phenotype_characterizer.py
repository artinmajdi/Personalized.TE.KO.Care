import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from tekoa.utils.phenotype_characterizer import (
    _calculate_eta_squared,
    _calculate_cramers_v,
    compare_variable_across_clusters,
    characterize_phenotypes
)
import logging

# Get the logger for the module to be tested to check log messages
phenotype_logger = logging.getLogger('tekoa.utils.phenotype_characterizer')

class TestPhenotypeCharacterizer(unittest.TestCase):

    def setUp(self):
        self.numeric_var_data = pd.DataFrame({
            'NumericVar': [10, 12, 11, 20, 22, 21, 30, 32, 31, 15, 25, 35],
            'Cluster':    [0,  0,  0,  1,  1,  1,  2,  2,  2,  0,  1,  2 ]
        })
        self.categorical_var_data = pd.DataFrame({
            'CategoricalVar': ['A', 'B', 'A', 'B', 'A', 'B', 'C', 'A', 'C', 'B', 'C', 'A'],
            'Cluster':        [0,   0,   0,   1,   1,   1,   2,   2,   2,   0,   1,   2]
        })
        self.mixed_data_for_characterization = pd.DataFrame({
            'ID': range(12),
            'Age': [25, 30, 28, 45, 50, 48, 60, 65, 62, 20, 40, 55], # Numeric
            'Severity': ['Mild', 'Moderate', 'Mild', 'Severe', 'Moderate', 'Severe', 'Mild', 'Severe', 'Moderate', 'Mild', 'Moderate', 'Severe'], # Categorical
            'Outcome': [100, 150, 120, 200, 250, 220, 90, 180, 160, 110, 240, 190] # Numeric
        })
        self.labels = np.array([0,0,0,1,1,1,2,2,2,0,1,2])

    def test_calculate_eta_squared(self):
        self.assertAlmostEqual(_calculate_eta_squared(f_statistic=5, df_between=2, df_within=10), 0.5)
        self.assertTrue(np.isnan(_calculate_eta_squared(f_statistic=5, df_between=2, df_within=0)))
        self.assertTrue(np.isnan(_calculate_eta_squared(f_statistic=np.nan, df_between=2, df_within=10)))
        self.assertTrue(np.isnan(_calculate_eta_squared(f_statistic=1, df_between=1, df_within=-1))) # Denominator becomes 0

    def test_calculate_cramers_v(self):
        self.assertAlmostEqual(_calculate_cramers_v(chi2_statistic=10, n=100, k=3, r=2), np.sqrt(10/(100*1)))
        self.assertTrue(np.isnan(_calculate_cramers_v(chi2_statistic=10, n=100, k=1, r=2))) # min_dim = 0
        self.assertTrue(np.isnan(_calculate_cramers_v(chi2_statistic=10, n=100, k=3, r=1))) # min_dim = 0
        self.assertTrue(np.isnan(_calculate_cramers_v(chi2_statistic=10, n=0, k=3, r=2)))
        self.assertTrue(np.isnan(_calculate_cramers_v(chi2_statistic=np.nan, n=100, k=3, r=2)))

    def test_compare_variable_across_clusters_anova_success(self):
        result = compare_variable_across_clusters(self.numeric_var_data, 'NumericVar')
        self.assertEqual(result['TestType'], 'ANOVA')
        self.assertIsInstance(result['PValue'], float)
        self.assertLess(result['PValue'], 0.05) # Expect significance for this data
        self.assertIsInstance(result['Statistic'], float)
        self.assertGreater(result['Statistic'], 0)
        self.assertIsInstance(result['EffectSize'], float)
        self.assertTrue(0 <= result['EffectSize'] <= 1)

    def test_compare_variable_across_clusters_chi_square_success(self):
        result = compare_variable_across_clusters(self.categorical_var_data, 'CategoricalVar')
        self.assertEqual(result['TestType'], 'Chi-Square')
        self.assertIsInstance(result['PValue'], float)
        self.assertIsInstance(result['Statistic'], float)
        self.assertGreaterEqual(result['Statistic'], 0)
        self.assertIsInstance(result['EffectSize'], float)
        self.assertTrue(0 <= result['EffectSize'] <= 1)

    def test_compare_variable_insufficient_clusters(self):
        data = pd.DataFrame({'Var': [1,2,3], 'Cluster': [0,0,0]})
        result = compare_variable_across_clusters(data, 'Var')
        self.assertEqual(result['TestType'], 'InsufficientClusters')
        self.assertTrue(np.isnan(result['PValue']))

    def test_compare_variable_anova_not_enough_groups(self):
        # Test case: one group has only one sample after dropna
        data_test = pd.DataFrame({'NumericVar': [10, np.nan, 20, 21], 'Cluster': [0,0,1,1]})
        result = compare_variable_across_clusters(data_test, 'NumericVar')
        self.assertEqual(result['TestType'], 'ANOVA_NotEnoughGroups')
        self.assertTrue(np.isnan(result['PValue']))

        # Test case: only one distinct group remains after dropna
        data_test_one_group_left = pd.DataFrame({'NumericVar': [10, 11, np.nan, np.nan], 'Cluster': [0,0,1,1]})
        result_one_group = compare_variable_across_clusters(data_test_one_group_left, 'NumericVar')
        self.assertEqual(result_one_group['TestType'], 'ANOVA_NotEnoughGroups') # or InsufficientClusters if it drops to 1 group
        self.assertTrue(np.isnan(result_one_group['PValue']))


    def test_compare_variable_chi_square_small_table(self):
        # This data will result in a contingency table where one dimension is 1 (only 'A')
        data_test = pd.DataFrame({'CatVar': ['A', 'A', 'A'], 'Cluster': [0,0,1]})
        result = compare_variable_across_clusters(data_test, 'CatVar')
        self.assertEqual(result['TestType'], 'ChiSquare_SmallTable')
        self.assertTrue(np.isnan(result['PValue']))

        # This data will result in a 2x1 table for cluster 1 (only 'B') if we filter that way, but crosstab does not.
        # The actual crosstab will be CatVar (rows) x Cluster (cols).
        # CatVar A: C0=2, C1=0
        # CatVar B: C0=0, C1=1  -- This is a 2x2 table, should be fine for Chi2.
        # Let's make it such that one variable category appears in only one cluster,
        # and that cluster has only that category, making a row/col sum to zero effectively after filtering.
        # No, Chi-Square expects the full table. A more direct small table:
        data_small_row = pd.DataFrame({'CatVar': ['A', 'A', 'A', 'A'], 'Cluster': [0, 0, 1, 1]}) # 1 row CatVar
        result_small_row = compare_variable_across_clusters(data_small_row, 'CatVar')
        self.assertEqual(result_small_row['TestType'], 'ChiSquare_SmallTable')

        data_small_col = pd.DataFrame({'CatVar': ['A', 'B', 'C', 'D'], 'Cluster': [0,0,0,0]}) # 1 col Cluster (after unique) -> InsufficientClusters
        result_small_col = compare_variable_across_clusters(data_small_col, 'CatVar')
        self.assertEqual(result_small_col['TestType'], 'InsufficientClusters')


    def test_compare_variable_anova_no_variance_or_same_mean(self):
        data = pd.DataFrame({'NumericVar': [10, 10, 10, 10, 10, 10], 'Cluster': [0,0,0,1,1,1]})
        result = compare_variable_across_clusters(data, 'NumericVar')
        self.assertEqual(result['TestType'], 'ANOVA_NoVarianceOrSameMean')
        self.assertEqual(result['PValue'], 1.0)
        self.assertEqual(result['Statistic'], 0.0)
        self.assertEqual(result['EffectSize'], 0.0)

    def test_characterize_phenotypes_success_with_statsmodels(self):
        # This test assumes statsmodels is available.
        # If it's not, the function should still run but CorrectedPValue/RejectNullFDR will be NaN/NA.
        variables_to_compare = ['Age', 'Severity', 'Outcome']
        results_df = characterize_phenotypes(self.mixed_data_for_characterization, self.labels, variables_to_compare)

        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), len(variables_to_compare))
        self.assertListEqual(list(results_df.columns), ['Variable', 'TestType', 'Statistic', 'PValue', 'CorrectedPValue', 'RejectNullFDR', 'EffectSize'])

        self.assertEqual(results_df[results_df['Variable'] == 'Age']['TestType'].iloc[0], 'ANOVA')
        self.assertEqual(results_df[results_df['Variable'] == 'Severity']['TestType'].iloc[0], 'Chi-Square')
        self.assertEqual(results_df[results_df['Variable'] == 'Outcome']['TestType'].iloc[0], 'ANOVA')

        # Check if statsmodels was available and ran by seeing if corrected p-values are not all NaN
        # This is an indirect check. If statsmodels is truly missing, these will be NaN/NA.
        if not results_df['CorrectedPValue'].isna().all():
            self.assertFalse(results_df['RejectNullFDR'].isna().all())


    @patch('tekoa.utils.phenotype_characterizer.multipletests', None) # Simulate import error by making it None
    def test_characterize_phenotypes_no_statsmodels_via_none(self, mock_multipletests_is_none):
        # This test uses the fact that characterize_phenotypes has a try-except around the import.
        # Setting multipletests to None in the module's scope for this test will trigger the except.
        with self.assertLogs(logger=phenotype_logger, level='WARNING') as cm:
            results_df = characterize_phenotypes(self.mixed_data_for_characterization, self.labels, ['Age', 'Outcome'])

        self.assertTrue(any("statsmodels.stats.multitest not found" in message for message in cm.output))
        self.assertTrue(results_df['CorrectedPValue'].isna().all())
        self.assertTrue(results_df['RejectNullFDR'].isna().all()) # pd.NA makes this true

    def test_characterize_phenotypes_empty_inputs(self):
        expected_cols = ['Variable', 'TestType', 'Statistic', 'PValue', 'CorrectedPValue', 'RejectNullFDR', 'EffectSize']

        # Empty original_data
        df_empty = characterize_phenotypes(pd.DataFrame(), self.labels, ['Age'])
        self.assertTrue(df_empty.empty)
        self.assertListEqual(list(df_empty.columns), expected_cols)

        # Empty labels
        df_empty_labels = characterize_phenotypes(self.mixed_data_for_characterization, np.array([]), ['Age'])
        self.assertTrue(df_empty_labels.empty)
        self.assertListEqual(list(df_empty_labels.columns), expected_cols)

        # Empty variables_to_compare
        df_empty_vars = characterize_phenotypes(self.mixed_data_for_characterization, self.labels, [])
        self.assertTrue(df_empty_vars.empty)
        self.assertListEqual(list(df_empty_vars.columns), expected_cols)

    def test_characterize_phenotypes_variable_not_found(self):
        results_df = characterize_phenotypes(self.mixed_data_for_characterization, self.labels, ['NonExistentVar', 'Age'])
        self.assertEqual(len(results_df), 2)
        non_existent_res = results_df[results_df['Variable'] == 'NonExistentVar'].iloc[0]
        self.assertEqual(non_existent_res['TestType'], 'Error_VariableNotFoundInSource')
        self.assertTrue(np.isnan(non_existent_res['PValue']))
        age_res = results_df[results_df['Variable'] == 'Age'].iloc[0]
        self.assertEqual(age_res['TestType'], 'ANOVA')


if __name__ == '__main__':
    unittest.main()
