import pytest
import pandas as pd
import numpy as np
from tekoa.utils.dimensionality_reducer import DimensionalityReducer

def test_perform_famd_empty_after_dropna():
    """
    Test that perform_famd raises a ValueError if the DataFrame becomes empty
    after rows with NaN values are dropped.
    """
    data = pd.DataFrame({
        'numeric_col1': [1, 2, np.nan],
        'numeric_col2': [np.nan, 5, 6],
        'categorical_col1': ['A', np.nan, 'C']
    })

    reducer = DimensionalityReducer(data)

    with pytest.raises(ValueError, match="No data remaining for FAMD after removing rows with missing values."):
        reducer.perform_famd(n_components=2)
