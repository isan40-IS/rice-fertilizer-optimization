import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    rows = 120
    n = rng.uniform(0, 180, rows)
    p = rng.uniform(0, 80, rows)
    k = rng.uniform(0, 120, rows)
    rain = rng.normal(800, 120, rows)
    yield_value = (
        1200 + 12*n + 5*p + 4*k - 0.035*n**2 + 0.2*rain
        + rng.normal(0, 80, rows)
    )
    return pd.DataFrame({
        "record_id": range(rows),
        "id": np.repeat(np.arange(12), 10),
        "exp.id": np.tile(np.repeat([1, 2], 5), 12),
        "year": np.repeat(np.arange(2000, 2012), 10),
        "location": "Synthetic",
        "crop": "Rice",
        "Nutrients applied (NPKS)": "NPK",
        "N.rate.kg_ha": n,
        "P.rate.kg_ha": p,
        "K.rate.kg_ha": k,
        "avprec.mm": rain,
        "soil_type": rng.choice(["A", "B", "C"], rows),
        "fertilizer_type": rng.choice(["Mineral", "Mixed"], rows),
        "yield.wb.kg_ha": yield_value,
    })
