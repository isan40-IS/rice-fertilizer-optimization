from rice_fertilizer.evaluation.fertilizer_efficiency import (
    average_efficiency,
    total_marginal_efficiency,
)
from rice_fertilizer.schemas import FertilizerVector

def test_average_efficiency():
    assert abs(average_efficiency(5000, FertilizerVector(50, 30, 20)) - 50) < 1e-6

def test_reduced_dose_status():
    result = total_marginal_efficiency(
        5200, 5000,
        FertilizerVector(40, 20, 20),
        FertilizerVector(50, 30, 30),
    )
    assert result["status"] == "increased_yield_reduced_dose"
