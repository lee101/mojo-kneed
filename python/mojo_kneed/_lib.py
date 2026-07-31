"""ctypes bridge to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "kneed.mojo")
LIB = os.environ.get("MOJO_KNEED_LIB") or os.path.join(
    ROOT, "dist", "libmojo-kneed.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mk_kneedle": ([I, I, I, F] + [I] * 12, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_KNEED_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= os.path.getmtime(SRC):
            return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo not found; run `pixi run build` first")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(values) -> np.ndarray:
    source = np.asarray(values)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex-valued inputs are not supported")
    result = np.ascontiguousarray(source, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError("FFI arrays must be one-dimensional")
    return result


def addr(values: np.ndarray) -> int:
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 1
        or not values.flags.c_contiguous
        or values.dtype not in (np.dtype(np.float64), np.dtype(np.int64))
        or values.size == 0
        or values.ctypes.data == 0
    ):
        raise ValueError("FFI buffers must be non-empty contiguous float64/int64 arrays")
    return int(values.ctypes.data)
