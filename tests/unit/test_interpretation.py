from types import SimpleNamespace

import numpy as np
import pytest

from rice_fertilizer.modeling.interpretation import built_in_feature_importance


def test_mismatched_importance_schema_is_not_silently_truncated():
    pipeline = SimpleNamespace(named_steps={
        "preprocessor": SimpleNamespace(get_feature_names_out=lambda: ["a", "b"]),
        "model": SimpleNamespace(feature_importances_=np.array([1.0])),
    })
    with pytest.raises(ValueError, match="schema mismatch"):
        built_in_feature_importance(pipeline)
