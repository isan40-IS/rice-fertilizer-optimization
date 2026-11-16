from dataclasses import asdict
import numpy as np

class ConvergenceTracker:
    def __init__(self, objective):
        self.objective = objective
        self.history = []
        self.generation = 0

    def callback(self, xk: np.ndarray, convergence: float) -> bool:
        self.generation += 1
        self.history.append({
            "generation": self.generation,
            "convergence": float(convergence),
            "n": float(xk[0]), "p": float(xk[1]), "k": float(xk[2]),
            **asdict(self.objective.components(xk)),
        })
        return False
