"""Synthetic curves provided by the upstream kneed API."""

from __future__ import annotations

import numpy as np


class DataGenerator:
    @staticmethod
    def noisy_gaussian(mu: float = 50, sigma: float = 10, N: int = 100, seed=42):
        np.random.seed(seed)
        samples = np.random.normal(loc=mu, scale=sigma, size=N)
        x = np.sort(samples)
        y = np.array(range(N)) / float(N)
        return x, y

    @staticmethod
    def figure2():
        with np.errstate(divide="ignore"):
            x = np.linspace(0.0, 1, 10)
            return x, np.true_divide(-1, x + 0.1) + 5

    @staticmethod
    def convex_increasing():
        return np.arange(0, 10), np.array([1, 2, 3, 4, 5, 10, 15, 20, 40, 100])

    @staticmethod
    def convex_decreasing():
        return np.arange(0, 10), np.array([100, 40, 20, 15, 10, 5, 4, 3, 2, 1])

    @staticmethod
    def concave_decreasing():
        return np.arange(0, 10), np.array([99, 98, 97, 96, 95, 90, 85, 80, 60, 0])

    @staticmethod
    def concave_increasing():
        return np.arange(0, 10), np.array([0, 60, 80, 85, 90, 95, 96, 97, 98, 99])

    @staticmethod
    def bumpy():
        y = [
            7305.0, 6979.0, 6666.6, 6463.2, 6326.5, 6048.8, 6032.8, 5762.0,
            5742.8, 5398.2, 5256.8, 5227.0, 5001.7, 4942.0, 4854.2, 4734.6,
            4558.7, 4491.1, 4411.6, 4333.0, 4234.6, 4139.1, 4056.8, 4022.5,
            3868.0, 3808.3, 3745.3, 3692.3, 3645.6, 3618.3, 3574.3, 3504.3,
            3452.4, 3401.2, 3382.4, 3340.7, 3301.1, 3247.6, 3190.3, 3180.0,
            3154.2, 3089.5, 3045.6, 2989.0, 2993.6, 2941.3, 2875.6, 2866.3,
            2834.1, 2785.1, 2759.7, 2763.2, 2720.1, 2660.1, 2690.2, 2635.7,
            2632.9, 2574.6, 2556.0, 2545.7, 2513.4, 2491.6, 2496.0, 2466.5,
            2442.7, 2420.5, 2381.5, 2388.1, 2340.6, 2335.0, 2318.9, 2319.0,
            2308.2, 2262.2, 2235.8, 2259.3, 2221.0, 2202.7, 2184.3, 2170.1,
            2160.0, 2127.7, 2134.7, 2102.0, 2101.4, 2066.4, 2074.3, 2063.7,
            2048.1, 2031.9,
        ]
        return list(range(90)), y
