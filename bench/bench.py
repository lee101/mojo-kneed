"""Benchmarks against upstream kneed on identical NumPy arrays."""

from __future__ import annotations

import gc
import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_kneed
from kneed import KneeLocator as UpstreamKneeLocator
from kneed import __version__ as upstream_version


def best_time(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def locator_case(n, online=False, polynomial=False):
    x = np.linspace(0.0, 1.0, n)
    if online:
        y = 1.0 - np.exp(-8.0 * x) + 0.003 * np.sin(240.0 * np.pi * x)
    else:
        y = 1.0 - np.exp(-8.0 * x)
    options = {"online": online}
    if polynomial:
        options.update(interp_method="polynomial", polynomial_degree=7)

    def mojo_call():
        return mojo_kneed.KneeLocator(x, y, **options)

    def upstream_call():
        return UpstreamKneeLocator(x, y, **options)

    actual = mojo_call()
    expected = upstream_call()
    if actual.knee != expected.knee or actual.all_knees != expected.all_knees:
        raise AssertionError("benchmark inputs do not produce parity")
    return mojo_call, upstream_call


def main():
    cases = [
        ("KneeLocator offline, 10k", lambda: locator_case(10_000)),
        ("KneeLocator offline, 1M", lambda: locator_case(1_000_000)),
        ("KneeLocator online wavy, 2k", lambda: locator_case(2_000, online=True)),
        (
            "KneeLocator polynomial degree 7, 100k",
            lambda: locator_case(100_000, polynomial=True),
        ),
    ]

    print(f"Machine: {cpu_name()}")
    print(f"Platform: {platform.platform()}; Python {platform.python_version()}")
    print(f"Comparison: mojo-kneed {mojo_kneed.__version__} vs kneed {upstream_version}")
    print()
    print("| case | mojo-kneed | kneed | speedup | result |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, build_case in cases:
        mojo_call, upstream_call = build_case()
        mojo_call()
        mojo_seconds = best_time(mojo_call)
        upstream_seconds = best_time(upstream_call)
        ratio = upstream_seconds / mojo_seconds
        result = "faster" if ratio >= 1.0 else "slower"
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{upstream_seconds * 1e3:.2f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
