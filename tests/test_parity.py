"""Behavioral and numerical parity with kneed 0.8.6."""

from __future__ import annotations

import inspect
import numpy as np
import pytest

import mojo_kneed as ours
from mojo_kneed._lib import addr, f64, lib
from kneed import DataGenerator as UpstreamDataGenerator
from kneed import KneeLocator as UpstreamKneeLocator
from kneed import find_shape as upstream_find_shape


CURVES = [
    ("concave", "increasing", "concave_increasing"),
    ("concave", "decreasing", "concave_decreasing"),
    ("convex", "increasing", "convex_increasing"),
    ("convex", "decreasing", "convex_decreasing"),
]


def assert_locator_parity(actual, expected):
    assert actual.knee == expected.knee
    assert actual.norm_knee == expected.norm_knee
    assert actual.knee_y == expected.knee_y
    assert actual.norm_knee_y == expected.norm_knee_y
    assert actual.all_knees == expected.all_knees
    assert actual.all_norm_knees == expected.all_norm_knees
    assert actual.all_knees_y == expected.all_knees_y
    assert actual.all_norm_knees_y == expected.all_norm_knees_y
    assert np.array_equal(actual.maxima_indices, expected.maxima_indices)
    assert np.array_equal(actual.minima_indices, expected.minima_indices)
    for name in (
        "Ds_y",
        "x_normalized",
        "y_normalized",
        "x_difference",
        "y_difference",
        "x_difference_maxima",
        "y_difference_maxima",
        "x_difference_minima",
        "y_difference_minima",
        "Tmx",
    ):
        assert np.allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=1e-13,
            atol=1e-13,
            equal_nan=True,
        ), name


def test_constructor_signature_matches_upstream():
    assert inspect.signature(ours.KneeLocator) == inspect.signature(
        UpstreamKneeLocator
    )


@pytest.mark.parametrize("curve,direction,generator", CURVES)
@pytest.mark.parametrize("interp_method", ["interp1d", "polynomial"])
def test_four_curve_shapes(curve, direction, generator, interp_method):
    x, y = getattr(UpstreamDataGenerator, generator)()
    actual = ours.KneeLocator(
        x,
        y,
        curve=curve,
        direction=direction,
        interp_method=interp_method,
    )
    expected = UpstreamKneeLocator(
        x,
        y,
        curve=curve,
        direction=direction,
        interp_method=interp_method,
    )
    assert_locator_parity(actual, expected)
    assert actual.elbow == expected.elbow
    assert actual.find_knee() == expected.find_knee()


@pytest.mark.parametrize("interp_method", ["interp1d", "polynomial"])
def test_published_figure2_vector(interp_method):
    x, y = UpstreamDataGenerator.figure2()
    actual = ours.KneeLocator(x, y, interp_method=interp_method)
    expected = UpstreamKneeLocator(x, y, interp_method=interp_method)
    assert_locator_parity(actual, expected)
    assert actual.knee == pytest.approx(0.2222222222222222)


@pytest.mark.parametrize("online", [False, True])
def test_bumpy_curve_online_and_offline(online):
    x, y = UpstreamDataGenerator.bumpy()
    options = {"curve": "convex", "direction": "decreasing", "online": online}
    actual = ours.KneeLocator(x, y, **options)
    expected = UpstreamKneeLocator(x, y, **options)
    assert_locator_parity(actual, expected)
    assert actual.all_elbows == expected.all_elbows
    assert actual.all_norm_elbows == expected.all_norm_elbows
    assert actual.all_elbows_y == expected.all_elbows_y
    assert actual.all_norm_elbows_y == expected.all_norm_elbows_y


@pytest.mark.parametrize("sensitivity", [0.0, 1.0, 3.0, 10.0, 100.0, 400.0])
def test_sensitivity(sensitivity):
    rng = np.random.RandomState(23)
    x = np.arange(1, 1001)
    y = np.sort(rng.gamma(0.5, 1.0, 1000))[::-1]
    options = {
        "S": sensitivity,
        "curve": "convex",
        "direction": "decreasing",
    }
    assert_locator_parity(
        ours.KneeLocator(x, y, **options),
        UpstreamKneeLocator(x, y, **options),
    )


@pytest.mark.parametrize(
    "direction,curve",
    [
        ("decreasing", "convex"),
        ("increasing", "convex"),
        ("increasing", "concave"),
        ("decreasing", "concave"),
    ],
)
def test_oscillating_curve(direction, curve):
    x = np.arange(0, 10, 0.1)
    y = np.sin(x)
    options = {
        "direction": direction,
        "curve": curve,
        "S": 1,
        "online": True,
    }
    assert_locator_parity(
        ours.KneeLocator(x, y, **options),
        UpstreamKneeLocator(x, y, **options),
    )


def test_polynomial_noisy_gaussian():
    x, y = UpstreamDataGenerator.noisy_gaussian(
        mu=50, sigma=10, N=1000, seed=42
    )
    options = {
        "interp_method": "polynomial",
        "polynomial_degree": 11,
        "online": True,
    }
    assert_locator_parity(
        ours.KneeLocator(x, y, **options),
        UpstreamKneeLocator(x, y, **options),
    )


@pytest.mark.parametrize("sensitivity", [0.0, 1.0])
def test_flat_maximum(sensitivity):
    x = np.arange(18, dtype=float)
    y = np.array(
        [
            1,
            0.787701317715959,
            0.7437774524158126,
            0.6559297218155198,
            0.5065885797950219,
            0.36749633967789164,
            0.2547584187408492,
            0.16251830161054173,
            0.10395314787701318,
            0.06734992679355783,
            0.043923865300146414,
            0.027818448023426062,
            0.01903367496339678,
            0.013177159590043924,
            0.010248901903367497,
            0.007320644216691069,
            0.005856515373352855,
            0.004392386530014641,
        ]
    )
    options = {
        "curve": "convex",
        "direction": "decreasing",
        "S": sensitivity,
    }
    assert_locator_parity(
        ours.KneeLocator(x, y, **options),
        UpstreamKneeLocator(x, y, **options),
    )


def test_no_knee_on_constant_curve():
    x = np.arange(10)
    y = np.ones(10)
    with np.errstate(invalid="ignore"):
        actual = ours.KneeLocator(x, y)
        expected = UpstreamKneeLocator(x, y)
    assert_locator_parity(actual, expected)
    assert actual.knee is None


@pytest.mark.parametrize("curve,direction,_", CURVES)
def test_simd_tail_parity(curve, direction, _):
    x = np.linspace(0.0, 1.0, 19)
    y = 1.0 - np.exp(-7.0 * x) + 0.01 * np.sin(11.0 * np.pi * x)
    options = {
        "curve": curve,
        "direction": direction,
        "online": True,
    }
    assert_locator_parity(
        ours.KneeLocator(x, y, **options),
        UpstreamKneeLocator(x, y, **options),
    )


@pytest.mark.parametrize("n", [262_143, 262_144])
def test_parallel_threshold_parity(n):
    x = np.linspace(0.0, 1.0, n)
    y = 1.0 - np.exp(-8.0 * x)
    assert_locator_parity(
        ours.KneeLocator(x, y),
        UpstreamKneeLocator(x, y),
    )


@pytest.mark.parametrize("curve,direction,generator", CURVES)
def test_transform_y(curve, direction, generator):
    _, y = getattr(UpstreamDataGenerator, generator)()
    assert np.array_equal(
        ours.KneeLocator.transform_y(y, direction, curve),
        UpstreamKneeLocator.transform_y(y, direction, curve),
    )


@pytest.mark.parametrize("curve,direction,generator", CURVES)
def test_find_shape(curve, direction, generator):
    x, y = getattr(UpstreamDataGenerator, generator)()
    assert ours.find_shape(x, y) == upstream_find_shape(x, y)
    assert ours.find_shape(x, y) == (direction, curve)


def test_find_shape_small_and_offset_inputs_match_polyfit_numerics():
    cases = [
        (np.array([1.0, 2.0]), np.array([0.4, -0.7])),
        (
            1.0e12 + np.arange(9, dtype=float),
            np.array([3.0, 2.0, 2.5, 1.0, 0.8, 0.7, 0.2, 0.1, -0.4]),
        ),
    ]
    for x, y in cases:
        assert ours.find_shape(x, y) == upstream_find_shape(x, y)


@pytest.mark.parametrize(
    "generator",
    [
        "figure2",
        "convex_increasing",
        "convex_decreasing",
        "concave_increasing",
        "concave_decreasing",
        "bumpy",
    ],
)
def test_data_generator_vectors(generator):
    actual = getattr(ours.DataGenerator, generator)()
    expected = getattr(UpstreamDataGenerator, generator)()
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


def test_noisy_gaussian_generator():
    actual = ours.DataGenerator.noisy_gaussian(N=256, seed=7)
    expected = UpstreamDataGenerator.noisy_gaussian(N=256, seed=7)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


@pytest.mark.parametrize(
    "options",
    [
        {"curve": "invalid"},
        {"direction": "invalid"},
        {"interp_method": "invalid"},
    ],
)
def test_invalid_options_match_exception_type(options):
    x, y = UpstreamDataGenerator.figure2()
    with pytest.raises(ValueError):
        ours.KneeLocator(x, y, **options)
    with pytest.raises(ValueError):
        UpstreamKneeLocator(x, y, **options)


def test_noncontiguous_float32_inputs_are_copied_safely():
    base = np.linspace(0.0, 1.0, 38, dtype=np.float32)
    x = base[::2]
    y = (1.0 - np.exp(-7.0 * base))[::2]
    assert not x.flags.c_contiguous
    actual = ours.KneeLocator(x, y)
    expected = UpstreamKneeLocator(x, y)
    assert actual.knee == expected.knee
    assert actual.all_knees == expected.all_knees
    assert np.allclose(actual.x_normalized, expected.x_normalized, rtol=1e-7)
    assert np.allclose(actual.y_normalized, expected.y_normalized, rtol=1e-7)
    assert ours.find_shape(x, y) == upstream_find_shape(x, y)


def test_ffi_helpers_reject_unsafe_buffers_and_complex_narrowing():
    with pytest.raises(TypeError, match="complex"):
        f64(np.array([1 + 2j, 3 + 4j]))
    with pytest.raises(ValueError, match="non-empty contiguous"):
        addr(np.arange(8, dtype=np.float64)[::2])
    with pytest.raises(ValueError, match="non-empty contiguous"):
        addr(np.empty(0, dtype=np.float64))


def test_exported_kernels_reject_null_pointers_without_dereferencing():
    kernel = lib().mk_kneedle
    args = [0, 0, 2, 1.0] + [0] * 12
    assert kernel(*args) == 2


def test_non_bool_online_matches_upstream_identity_semantics():
    x, y = UpstreamDataGenerator.bumpy()
    options = {"curve": "convex", "direction": "decreasing", "online": 0}
    assert_locator_parity(
        ours.KneeLocator(x, y, **options),
        UpstreamKneeLocator(x, y, **options),
    )


def test_plotting_helpers_execute_when_matplotlib_is_installed():
    pyplot = pytest.importorskip("matplotlib.pyplot")
    x, y = UpstreamDataGenerator.figure2()
    locator = ours.KneeLocator(x, y)
    locator.plot_knee()
    locator.plot_knee_normalized()
    assert len(pyplot.get_fignums()) == 2
    pyplot.close("all")
