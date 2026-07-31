"""Automatic curve-shape classification."""

from __future__ import annotations

import numpy as np


def find_shape(x, y):
    x_values = np.asarray(x)
    y_values = np.asarray(y)
    if (
        x_values.ndim != 1
        or y_values.ndim != 1
        or x_values.size != y_values.size
    ):
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if x_values.size < 2:
        raise ValueError("x and y do not contain enough points")
    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    start, stop = int(len(x_values) * 0.2), int(len(x_values) * 0.8)
    residual = np.mean(y_values[start:stop]) - np.mean(
        x_values[start:stop] * slope + intercept
    )
    if slope > 0:
        return ("increasing", "concave" if residual > 0 else "convex")
    return ("decreasing", "concave" if residual > 0 else "convex")
