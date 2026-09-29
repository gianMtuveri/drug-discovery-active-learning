import numpy as np
import pandas as pd
import pytest

from src.regression.simulation import concatenate_simulation_histories


def test_concatenation_clears_array_attrs_without_changing_history_data():
    first = pd.DataFrame({"seed": [0], "prediction": [1.25]})
    second = pd.DataFrame({"seed": [1], "prediction": [2.5]})
    first.attrs["test_indices"] = np.array([1, 2])
    second.attrs["test_indices"] = np.array([3, 4])

    with pytest.raises(ValueError, match="truth value of an array"):
        pd.concat([first, second], ignore_index=True)

    combined = concatenate_simulation_histories([first, second])

    expected = pd.DataFrame(
        {"seed": [0, 1], "prediction": [1.25, 2.5]}
    )
    pd.testing.assert_frame_equal(combined, expected)
    assert first.attrs == {}
    assert second.attrs == {}
    assert combined.attrs == {}
