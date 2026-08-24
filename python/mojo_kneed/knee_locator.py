"""Upstream-compatible Python API backed by the Mojo Kneedle scan."""

from typing import Iterable, Optional, Tuple

import numpy as np

from ._lib import addr, f64, lib

VALID_CURVE = ["convex", "concave"]
VALID_DIRECTION = ["increasing", "decreasing"]

try:
    import matplotlib.pyplot as plt
except ImportError:
    _has_matplotlib = False
    _matplotlib_not_found_err = ModuleNotFoundError(
        "This function needs Matplotlib to be executed. Please install matplotlib."
    )
else:
    _has_matplotlib = True


class KneeLocator:
    """Find knee or elbow points with the Kneedle algorithm."""

    def __init__(
        self,
        x: Iterable[float],
        y: Iterable[float],
        S: float = 1.0,
        curve: str = "concave",
        direction: str = "increasing",
        interp_method: str = "interp1d",
        online: bool = False,
        polynomial_degree: int = 7,
    ):
        self.x = np.array(x)
        self.y = np.array(y)
        self.curve = curve
        self.direction = direction
        self.N = len(self.x)
        self.S = S
        self.all_knees = set()
        self.all_norm_knees = set()
        self.all_knees_y = []
        self.all_norm_knees_y = []
        self.online = online
        self.polynomial_degree = polynomial_degree

        if curve not in VALID_CURVE or direction not in VALID_DIRECTION:
            raise ValueError(
                "Please check that the curve and direction arguments are valid."
            )
        if self.x.ndim != 1 or self.y.ndim != 1 or len(self.x) != len(self.y):
            raise ValueError("x and y must be one-dimensional arrays of equal length")
        if self.N < 2:
            raise ValueError("x and y must contain at least two points")

        if interp_method == "interp1d":
            self.Ds_y = np.asarray(self.y, dtype=np.float64).copy()
        elif interp_method == "polynomial":
            polynomial = np.poly1d(
                np.polyfit(self.x, self.y, self.polynomial_degree)
            )
            self.Ds_y = polynomial(self.x)
        else:
            raise ValueError(
                f"{interp_method} is an invalid interp_method parameter, use either "
                "'interp1d' or 'polynomial'"
            )
        self.interp_method = interp_method

        x64 = f64(self.x)
        smooth64 = f64(self.Ds_y)
        diagnostic_scratch = np.empty((4, self.N), dtype=np.float64)
        (
            self.x_normalized,
            self.x_difference,
            self.y_normalized,
            self.y_difference,
        ) = diagnostic_scratch
        index_scratch = np.empty((4, self.N), dtype=np.int64)
        maxima, minima, knee_indices, norm_knee_indices = index_scratch
        thresholds = np.empty(self.N, dtype=np.float64)
        counts = np.empty(3, dtype=np.int64)

        curve_code = 0 if curve == "convex" else 1
        direction_code = 0 if direction == "increasing" else 1
        # These local arrays remain strongly referenced for the entire synchronous
        # ctypes call; Mojo neither retains nor frees any of their addresses.
        status = lib().mk_kneedle(
            addr(x64),
            addr(smooth64),
            self.N,
            float(S),
            curve_code,
            direction_code,
            0 if online is False else 1,
            addr(self.x_normalized),
            addr(self.x_difference),
            addr(self.y_normalized),
            addr(self.y_difference),
            addr(maxima),
            addr(minima),
            addr(thresholds),
            addr(knee_indices),
            addr(norm_knee_indices),
            addr(counts),
        )
        if status:
            reasons = {
                1: "x and y must contain at least two points",
                2: "an FFI buffer pointer was null",
                3: "an invalid kernel option was supplied",
            }
            raise ValueError(reasons.get(status, f"Kneedle kernel failed ({status})"))

        maxima_count, minima_count, knee_count = map(int, counts)
        self.maxima_indices = maxima[:maxima_count].copy()
        self.minima_indices = minima[:minima_count].copy()
        self.Tmx = thresholds[:maxima_count].copy()
        self.x_difference_maxima = self.x_difference[self.maxima_indices]
        self.y_difference_maxima = self.y_difference[self.maxima_indices]
        self.x_difference_minima = self.x_difference[self.minima_indices]
        self.y_difference_minima = self.y_difference[self.minima_indices]

        self.knee = None
        self.norm_knee = None
        for raw_index, normalized_index in zip(
            knee_indices[:knee_count], norm_knee_indices[:knee_count]
        ):
            knee = self.x[int(raw_index)]
            norm_knee = self.x_normalized[int(normalized_index)]
            if knee not in self.all_knees:
                y_at_knee = self.y[self.x == knee][0]
                y_norm_at_knee = self.y_normalized[
                    self.x_normalized == norm_knee
                ][0]
                self.all_knees_y.append(y_at_knee)
                self.all_norm_knees_y.append(y_norm_at_knee)
            self.all_knees.add(knee)
            self.all_norm_knees.add(norm_knee)
            self.knee = knee
            self.norm_knee = norm_knee

        self.knee_y = self.norm_knee_y = None
        if self.knee:
            self.knee_y = self.y[self.x == self.knee][0]
            self.norm_knee_y = self.y_normalized[
                self.x_normalized == self.norm_knee
            ][0]

    @staticmethod
    def transform_y(
        y: Iterable[float], direction: str, curve: str
    ) -> np.ndarray:
        transformed = np.asarray(y)
        if direction == "decreasing":
            if curve == "concave":
                transformed = np.flip(transformed)
            elif curve == "convex":
                transformed = transformed.max() - transformed
        elif direction == "increasing" and curve == "convex":
            transformed = np.flip(transformed.max() - transformed)
        return transformed

    def find_knee(self):
        return self.knee, self.norm_knee

    def plot_knee_normalized(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Normalized Knee Point",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ):
        if not _has_matplotlib:
            raise _matplotlib_not_found_err
        plt.figure(figsize=(6, 6) if figsize is None else figsize)
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.plot(
            self.x_normalized, self.y_normalized, "b", label="normalized curve"
        )
        plt.plot(
            self.x_difference, self.y_difference, "r", label="difference curve"
        )
        plt.xticks(
            np.arange(
                self.x_normalized.min(), self.x_normalized.max() + 0.1, 0.1
            )
        )
        plt.yticks(
            np.arange(
                self.y_difference.min(), self.y_normalized.max() + 0.1, 0.1
            )
        )
        plt.vlines(
            self.norm_knee,
            plt.ylim()[0],
            plt.ylim()[1],
            linestyles="--",
            label="knee/elbow",
        )
        plt.legend(loc="best")

    def plot_knee(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Knee Point",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ):
        if not _has_matplotlib:
            raise _matplotlib_not_found_err
        plt.figure(figsize=(6, 6) if figsize is None else figsize)
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.plot(self.x, self.y, "b", label="data")
        plt.vlines(
            self.knee,
            plt.ylim()[0],
            plt.ylim()[1],
            linestyles="--",
            label="knee/elbow",
        )
        plt.legend(loc="best")

    @property
    def elbow(self):
        return self.knee

    @property
    def norm_elbow(self):
        return self.norm_knee

    @property
    def elbow_y(self):
        return self.knee_y

    @property
    def norm_elbow_y(self):
        return self.norm_knee_y

    @property
    def all_elbows(self):
        return self.all_knees

    @property
    def all_norm_elbows(self):
        return self.all_norm_knees

    @property
    def all_elbows_y(self):
        return self.all_knees_y

    @property
    def all_norm_elbows_y(self):
        return self.all_norm_knees_y
