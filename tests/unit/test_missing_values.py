import numpy as np
import pandas as pd
from rice_fertilizer.data.missing_values import (
    parse_applied_nutrients,
    reconstruct_structural_zeroes,
)

def test_parse_simple_np_label():
    nutrients, status = parse_applied_nutrients("NP", ["CONTROL"])
    assert nutrients == {"N", "P"}
    assert status == "automatic_npk_pattern"

def test_missing_absent_nutrient_becomes_zero():
    df = pd.DataFrame({
        "Nutrients applied (NPKS)": ["NP"],
        "N.rate.kg_ha": [50.0],
        "P.rate.kg_ha": [20.0],
        "K.rate.kg_ha": [np.nan],
    })
    result, _, _ = reconstruct_structural_zeroes(
        df,
        "Nutrients applied (NPKS)",
        {"N": "N.rate.kg_ha", "P": "P.rate.kg_ha", "K": "K.rate.kg_ha"},
        ["CONTROL"],
    )
    assert result.loc[0, "K.rate.kg_ha"] == 0.0
    assert bool(result.loc[0, "K_was_structural_zero"])

def test_applied_but_missing_remains_missing():
    df = pd.DataFrame({
        "Nutrients applied (NPKS)": ["NPK"],
        "N.rate.kg_ha": [50.0],
        "P.rate.kg_ha": [20.0],
        "K.rate.kg_ha": [np.nan],
    })
    result, _, _ = reconstruct_structural_zeroes(
        df,
        "Nutrients applied (NPKS)",
        {"N": "N.rate.kg_ha", "P": "P.rate.kg_ha", "K": "K.rate.kg_ha"},
        ["CONTROL"],
    )
    assert pd.isna(result.loc[0, "K.rate.kg_ha"])
    assert result.loc[0, "K_missing_reason"] == "applied_but_rate_missing"
