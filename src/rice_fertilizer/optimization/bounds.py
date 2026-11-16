from ..config import require_value
from ..schemas import NutrientBounds

def bounds_from_decisions(decisions: dict) -> NutrientBounds:
    values = {}
    for nutrient in ("N", "P", "K"):
        lower = require_value(decisions, f"optimization.bounds.{nutrient}.min")
        upper = require_value(decisions, f"optimization.bounds.{nutrient}.max")
        if float(lower) >= float(upper):
            raise ValueError(f"Invalid {nutrient} bounds")
        values[nutrient] = (float(lower), float(upper))
    return NutrientBounds(
        values["N"][0], values["N"][1],
        values["P"][0], values["P"][1],
        values["K"][0], values["K"][1],
    )
