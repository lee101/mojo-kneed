# mojo-kneed

Knee-point and elbow detection with the linear-time part of the
[Kneedle algorithm](https://raghavan.usc.edu/papers/kneedle-simplex11.pdf)
implemented in Mojo. The Python API follows
[`kneed`](https://github.com/arvkevi/kneed) 0.8.6, while normalization,
curve transformation, local-extrema detection, threshold construction, the
knee scan runs in a compiled shared library.

For the covered API, changing an import is normally the only application
change:

```python
# from kneed import KneeLocator
from mojo_kneed import KneeLocator
```

## Coverage

Covered:

- `KneeLocator(x, y, S=1.0, curve="concave",
  direction="increasing", interp_method="interp1d", online=False,
  polynomial_degree=7)` with the upstream names, defaults, and behavior.
- Concave/convex and increasing/decreasing curves, sensitivity thresholds,
  offline first-knee detection, and online multi-knee detection.
- The upstream diagnostic attributes: normalized and difference curves,
  extrema indices and values, thresholds, knee values, normalized knee
  values, all-knee collections, and all elbow aliases.
- `KneeLocator.transform_y`, `find_knee`, `plot_knee`, and
  `plot_knee_normalized`.
- `DataGenerator` and all six upstream data sets, plus
  `DataGenerator.noisy_gaussian`.
- Both `interp1d` and `polynomial` modes. Polynomial fitting uses
  `numpy.polyfit`; its linear-time Kneedle pass still runs in Mojo.
- `find_shape(x, y)`. It uses NumPy's `polyfit`, as upstream does, because
  substituting a different regression algorithm can change classifications.

Not covered:

- The upstream package name is not shadowed. Import from `mojo_kneed`, which
  lets `kneed` remain installed beside it for parity testing.
- Matplotlib is not a base dependency. The two plotting helpers work when
  Matplotlib is installed; the locked development environment includes it.
- Inputs crossing the Mojo boundary are copied to contiguous float64 arrays.
  Complex inputs are rejected instead of silently discarding their imaginary
  component. Consequently, diagnostic arrays from float32 inputs can differ
  from upstream by float32 rounding, although detected knees are compatible.
- APIs outside `KneeLocator`, `DataGenerator`, and `find_shape` are not
  provided.
- This is a CPU implementation. No GPU path is included because Kneedle's
  linear passes stay below roughly two flops per byte moved; they are
  memory-bound, and device transfers would add overhead rather than remove it.

The test suite compares directly with the real `kneed==0.8.6` package, not a
reimplemented Python reference.

## Install

Install the pinned Mojo toolchain and Python dependencies, then build the
shared library:

```bash
pixi install
pixi run build
```

Run the parity suite and benchmarks with:

```bash
pixi run test
pixi run bench
```

## Usage

This example is also usable verbatim through `pixi run python` after the
build:

```python
from mojo_kneed import DataGenerator, KneeLocator

x, y = DataGenerator.figure2()
locator = KneeLocator(
    x,
    y,
    S=1.0,
    curve="concave",
    direction="increasing",
)

print(locator.knee)       # 0.2222222222222222
print(locator.knee_y)     # 1.8965517241379306
print([float(k) for k in locator.all_knees])  # [0.2222222222222222]
```

Automatic shape selection is available too:

```python
from mojo_kneed import KneeLocator, find_shape

direction, curve = find_shape(x, y)
locator = KneeLocator(x, y, direction=direction, curve=curve)
```

## Benchmarks

Measured on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic x86-64, Python 3.13.14. Each row is the best of three warm
runs of the complete Python call, including input copies and result
construction. These are the results printed by `pixi run bench` on
2026-08-24:

| case | mojo-kneed | kneed | speedup | result |
| --- | ---: | ---: | ---: | --- |
| KneeLocator offline, 10k | 0.51 ms | 6.24 ms | 12.29x | faster |
| KneeLocator offline, 1M | 31.14 ms | 544.31 ms | 17.48x | faster |
| KneeLocator online wavy, 2k | 2.31 ms | 17.14 ms | 7.41x | faster |
| KneeLocator polynomial degree 7, 100k | 24.62 ms | 72.02 ms | 2.92x | faster |

The polynomial row has the smallest gain because both implementations spend
part of the call in the same NumPy polynomial fit. Benchmark results depend on
the CPU, allocator, NumPy build, and curve shape; run the locked benchmark on
the target machine rather than assuming these ratios.

## How it works

The wrapper first creates the same fitted values that upstream uses.
`interp1d` is an identity when evaluated at its original sample positions, so
that mode needs no SciPy call. Polynomial mode deliberately uses the same
`numpy.polyfit` and `numpy.poly1d` operations as upstream.

The fitted values and x coordinates cross a C ABI as addresses to contiguous
float64 NumPy arrays. Python preallocates results in one diagnostic slab and
one index scratch allocation. Mojo reconstructs pointers from the integer
addresses. Fused SIMD passes perform range reduction, normalization,
direction/curvature transformation, difference construction, and extrema
comparisons, each with explicit scalar boundary and tail handling. The
normalization pass also writes the independent `x_difference` diagnostic,
avoiding a separate full-array copy. Inputs of at least 262,144 points divide
that independent pass into 65,536-point tasks across at most eight workers;
smaller inputs stay serial. Sensitivity threshold calculation and ordered
threshold traversal remain serial. NumPy storage is used directly across the
FFI call, and Mojo writes result buffers in place, so no Mojo allocation,
copy, or ownership crosses the boundary.

`build/build.sh` compiles `src/kneed.mojo` with
`mojo build --emit shared-lib` into `dist/libmojo-kneed.so`. The ctypes loader
uses that library and rebuilds it only if the source is newer.

## License

mojo-kneed is MIT licensed. The compatible `DataGenerator` vectors and API
behavior derive from BSD-3-Clause-licensed `kneed`; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
