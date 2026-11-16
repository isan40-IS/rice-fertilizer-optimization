from rice_fertilizer.evaluation.productivity import yield_improvement

def test_yield_improvement():
    result = yield_improvement(5500, 5000)
    assert result["absolute_improvement"] == 500
    assert abs(result["percentage_improvement"] - 10) < 1e-6
    assert result["improved"]
