def yield_improvement(
    optimized_yield: float, baseline_yield: float, epsilon: float = 1e-8
) -> dict:
    absolute = float(optimized_yield - baseline_yield)
    return {
        "baseline_yield": float(baseline_yield),
        "optimized_yield": float(optimized_yield),
        "absolute_improvement": absolute,
        "percentage_improvement": float(
            100.0 * absolute / (abs(baseline_yield) + epsilon)
        ),
        "improved": absolute > 0,
    }
