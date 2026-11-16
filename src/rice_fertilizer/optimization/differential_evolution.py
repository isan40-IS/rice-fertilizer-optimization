from scipy.optimize import differential_evolution

def run_de(objective, bounds, config: dict, seed: int, callback=None):
    return differential_evolution(
        objective,
        bounds=bounds,
        strategy=config.get("strategy", "best1bin"),
        maxiter=int(config.get("maxiter", 300)),
        popsize=int(config.get("popsize", 15)),
        tol=float(config.get("tol", 1e-4)),
        mutation=tuple(config.get("mutation", [0.5, 1.0])),
        recombination=float(config.get("recombination", 0.7)),
        seed=int(seed),
        callback=callback,
        polish=bool(config.get("polish", False)),
        workers=int(config.get("workers", 1)),
        updating=config.get("updating", "immediate"),
    )
