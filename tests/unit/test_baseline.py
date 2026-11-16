import pandas as pd
from rice_fertilizer.evaluation.baseline import resolve_baseline

def test_observed_baseline():
    row = pd.Series({"N": 10.0, "P": 5.0, "K": 2.0})
    result = resolve_baseline(
        "observed", row, pd.DataFrame([row]),
        {"N": "N", "P": "P", "K": "K"},
    )
    assert result.total() == 17.0
