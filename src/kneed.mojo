"""Kneedle's linear-time normalization, extrema, and threshold scan."""

from std.algorithm import parallelize
from std.sys.info import simd_width_of


comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 262144
comptime PARALLEL_CHUNK_SIZE = 65536


def normalize_forward(
    x: FPtr,
    smooth_y: FPtr,
    x_normalized: FPtr,
    y_normalized: FPtr,
    y_difference: FPtr,
    start: Int,
    stop: Int,
    x_min: Float64,
    x_span: Float64,
    y_min: Float64,
    y_span: Float64,
    invert_y: Bool,
):
    comptime W = simd_width_of[DType.float64]()
    var vector_stop = start + ((stop - start) // W) * W
    for i in range(start, vector_stop, W):
        var normalized_x = (x.load[width=W](i) - x_min) / x_span
        var normalized_y = (
            smooth_y.load[width=W](i) - y_min
        ) / y_span
        if invert_y:
            normalized_y = 1.0 - normalized_y
        x_normalized.store(i, normalized_x)
        y_normalized.store(i, normalized_y)
        y_difference.store(i, normalized_y - normalized_x)
    for i in range(vector_stop, stop):
        var normalized_x = (x[i] - x_min) / x_span
        var normalized_y = (smooth_y[i] - y_min) / y_span
        if invert_y:
            normalized_y = 1.0 - normalized_y
        x_normalized[i] = normalized_x
        y_normalized[i] = normalized_y
        y_difference[i] = normalized_y - normalized_x


def normalize_reverse(
    x: FPtr,
    smooth_y: FPtr,
    x_normalized: FPtr,
    y_normalized: FPtr,
    y_difference: FPtr,
    n: Int,
    start: Int,
    stop: Int,
    x_min: Float64,
    x_span: Float64,
    y_min: Float64,
    y_span: Float64,
    invert_y: Bool,
):
    comptime W = simd_width_of[DType.float64]()
    var vector_stop = start + ((stop - start) // W) * W
    for i in range(start, vector_stop, W):
        var normalized_x = (x.load[width=W](i) - x_min) / x_span
        var normalized_y = SIMD[DType.float64, W]()
        comptime for lane in range(W):
            normalized_y[lane] = (
                smooth_y[n - 1 - i - lane] - y_min
            ) / y_span
        if invert_y:
            normalized_y = 1.0 - normalized_y
        x_normalized.store(i, normalized_x)
        y_normalized.store(i, normalized_y)
        y_difference.store(i, normalized_y - normalized_x)
    for i in range(vector_stop, stop):
        var normalized_x = (x[i] - x_min) / x_span
        var normalized_y = (smooth_y[n - 1 - i] - y_min) / y_span
        if invert_y:
            normalized_y = 1.0 - normalized_y
        x_normalized[i] = normalized_x
        y_normalized[i] = normalized_y
        y_difference[i] = normalized_y - normalized_x


def local_maximum(values: FPtr, i: Int, n: Int) -> Bool:
    var left = i - 1
    var right = i + 1
    if left < 0:
        left = 0
    if right >= n:
        right = n - 1
    return values[i] >= values[left] and values[i] >= values[right]


def local_minimum(values: FPtr, i: Int, n: Int) -> Bool:
    var left = i - 1
    var right = i + 1
    if left < 0:
        left = 0
    if right >= n:
        right = n - 1
    return values[i] <= values[left] and values[i] <= values[right]


@export("mk_kneedle")
def mk_kneedle(
    x_addr: Int,
    smooth_y_addr: Int,
    n: Int,
    sensitivity: Float64,
    curve: Int,
    direction: Int,
    online: Int,
    x_normalized_addr: Int,
    y_normalized_addr: Int,
    y_difference_addr: Int,
    maxima_addr: Int,
    minima_addr: Int,
    thresholds_addr: Int,
    knee_indices_addr: Int,
    norm_knee_indices_addr: Int,
    counts_addr: Int,
) abi("C") -> Int:
    if n < 2:
        return 1
    if (
        x_addr == 0 or smooth_y_addr == 0 or x_normalized_addr == 0
        or y_normalized_addr == 0 or y_difference_addr == 0
        or maxima_addr == 0 or minima_addr == 0 or thresholds_addr == 0
        or knee_indices_addr == 0 or norm_knee_indices_addr == 0
        or counts_addr == 0
    ):
        return 2
    if (curve != 0 and curve != 1) or (
        direction != 0 and direction != 1
    ) or (online != 0 and online != 1):
        return 3

    var x = FPtr(unsafe_from_address=x_addr)
    var smooth_y = FPtr(unsafe_from_address=smooth_y_addr)
    var x_normalized = FPtr(unsafe_from_address=x_normalized_addr)
    var y_normalized = FPtr(unsafe_from_address=y_normalized_addr)
    var y_difference = FPtr(unsafe_from_address=y_difference_addr)
    var maxima = IPtr(unsafe_from_address=maxima_addr)
    var minima = IPtr(unsafe_from_address=minima_addr)
    var thresholds = FPtr(unsafe_from_address=thresholds_addr)
    var knee_indices = IPtr(unsafe_from_address=knee_indices_addr)
    var norm_knee_indices = IPtr(unsafe_from_address=norm_knee_indices_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)

    var x_min = x[0]
    var x_max = x[0]
    var y_min = smooth_y[0]
    var y_max = smooth_y[0]
    for i in range(1, n):
        if x[i] < x_min:
            x_min = x[i]
        if x[i] > x_max:
            x_max = x[i]
        if smooth_y[i] < y_min:
            y_min = smooth_y[i]
        if smooth_y[i] > y_max:
            y_max = smooth_y[i]

    var x_span = x_max - x_min
    var y_span = y_max - y_min
    var reverse_y = direction == 1 and curve == 1 or (
        direction == 0 and curve == 0
    )
    var invert_y = curve == 0
    if n >= PARALLEL_THRESHOLD:
        var task_count = (
            n + PARALLEL_CHUNK_SIZE - 1
        ) // PARALLEL_CHUNK_SIZE

        @parameter
        def normalize_task(task: Int):
            var start = task * PARALLEL_CHUNK_SIZE
            var stop = min(start + PARALLEL_CHUNK_SIZE, n)
            if reverse_y:
                normalize_reverse(
                    x,
                    smooth_y,
                    x_normalized,
                    y_normalized,
                    y_difference,
                    n,
                    start,
                    stop,
                    x_min,
                    x_span,
                    y_min,
                    y_span,
                    invert_y,
                )
            else:
                normalize_forward(
                    x,
                    smooth_y,
                    x_normalized,
                    y_normalized,
                    y_difference,
                    start,
                    stop,
                    x_min,
                    x_span,
                    y_min,
                    y_span,
                    invert_y,
                )

        parallelize[normalize_task](task_count, min(task_count, 8))
    elif reverse_y:
        normalize_reverse(
            x,
            smooth_y,
            x_normalized,
            y_normalized,
            y_difference,
            n,
            0,
            n,
            x_min,
            x_span,
            y_min,
            y_span,
            invert_y,
        )
    else:
        normalize_forward(
            x,
            smooth_y,
            x_normalized,
            y_normalized,
            y_difference,
            0,
            n,
            x_min,
            x_span,
            y_min,
            y_span,
            invert_y,
        )

    var mean_spacing = 0.0
    comptime W = simd_width_of[DType.float64]()
    var spacing_vector_stop = ((n - 1) // W) * W
    for i in range(0, spacing_vector_stop, W):
        mean_spacing += abs(
            x_normalized.load[width=W](i + 1)
            - x_normalized.load[width=W](i)
        ).reduce_add()
    for i in range(spacing_vector_stop, n - 1):
        mean_spacing += abs(x_normalized[i + 1] - x_normalized[i])
    mean_spacing /= Float64(n - 1)

    var maxima_count = 0
    var minima_count = 0
    for i in range(n):
        if local_maximum(y_difference, i, n):
            maxima[maxima_count] = Int64(i)
            thresholds[maxima_count] = (
                y_difference[i] - sensitivity * mean_spacing
            )
            maxima_count += 1
        if local_minimum(y_difference, i, n):
            minima[minima_count] = Int64(i)
            minima_count += 1

    counts[0] = Int64(maxima_count)
    counts[1] = Int64(minima_count)
    counts[2] = 0
    if maxima_count == 0:
        return 0

    var max_cursor = 0
    var min_cursor = 0
    var threshold = 0.0
    var threshold_index = 0
    var detection_active = True
    var knee_count = 0
    var i = Int(maxima[0])
    while i < n - 1:
        if max_cursor < maxima_count and Int(maxima[max_cursor]) == i:
            threshold = thresholds[max_cursor]
            threshold_index = i
            max_cursor += 1
            detection_active = True
        if min_cursor < minima_count and Int(minima[min_cursor]) == i:
            threshold = 0.0
            min_cursor += 1
            detection_active = False

        if detection_active and y_difference[i + 1] < threshold:
            var raw_index = threshold_index
            if (curve == 0 and direction == 0) or (
                curve == 1 and direction == 1
            ):
                raw_index = n - 1 - threshold_index
            knee_indices[knee_count] = Int64(raw_index)
            norm_knee_indices[knee_count] = Int64(threshold_index)
            knee_count += 1
            if online == 0:
                break
        i += 1

    counts[2] = Int64(knee_count)
    return 0
