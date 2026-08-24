"""Kneedle's linear-time normalization, extrema, and threshold scan."""

from std.runtime import initialize_runtime
from std.runtime.asyncrt import TaskGroup
from std.sys.info import simd_width_of


comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 262144
comptime PARALLEL_CHUNK_SIZE = 65536


@always_inline
def sync_parallelize[FuncType: def(Int) -> None](func: FuncType, count: Int):
    @__parameter
    @always_inline
    def wrapped(i: Int):
        func(i)

    @always_inline
    @__parameter
    async def task_fn(i: Int):
        wrapped(i)

    var tasks = TaskGroup()
    for i in range(count):
        tasks.create_task(task_fn(i))
    tasks.wait()


@always_inline
def parallelize[
    origins: OriginSet,
    //,
    func: def(Int) capturing[origins] -> None,
](num_work_items: Int, num_workers: Int):
    def unified_func(i: Int):
        func(i)

    var chunk_size, extra_items = divmod(num_work_items, num_workers)

    @always_inline
    def worker(worker_index: Int) {imm chunk_size, imm extra_items}:
        var start = worker_index * chunk_size + min(worker_index, extra_items)
        for i in range(chunk_size + Int(worker_index < extra_items)):
            unified_func(start + i)

    sync_parallelize(worker, num_workers)


def normalize_forward(
    x: FPtr,
    smooth_y: FPtr,
    x_normalized: FPtr,
    x_difference: FPtr,
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
        var normalized_x = (x.unsafe_load[width=W](i) - x_min) / x_span
        var normalized_y = (smooth_y.unsafe_load[width=W](i) - y_min) / y_span
        if invert_y:
            normalized_y = 1.0 - normalized_y
        x_normalized.unsafe_store(i, normalized_x)
        x_difference.unsafe_store(i, normalized_x)
        y_normalized.unsafe_store(i, normalized_y)
        y_difference.unsafe_store(i, normalized_y - normalized_x)
    for i in range(vector_stop, stop):
        var normalized_x = (x[unsafe_offset=i] - x_min) / x_span
        var normalized_y = (smooth_y[unsafe_offset=i] - y_min) / y_span
        if invert_y:
            normalized_y = 1.0 - normalized_y
        x_normalized[unsafe_offset=i] = normalized_x
        x_difference[unsafe_offset=i] = normalized_x
        y_normalized[unsafe_offset=i] = normalized_y
        y_difference[unsafe_offset=i] = normalized_y - normalized_x


def normalize_reverse(
    x: FPtr,
    smooth_y: FPtr,
    x_normalized: FPtr,
    x_difference: FPtr,
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
        var normalized_x = (x.unsafe_load[width=W](i) - x_min) / x_span
        var normalized_y = SIMD[DType.float64, W]()
        comptime for lane in range(W):
            normalized_y[lane] = (
                smooth_y[unsafe_offset=n - 1 - i - lane] - y_min
            ) / y_span
        if invert_y:
            normalized_y = 1.0 - normalized_y
        x_normalized.unsafe_store(i, normalized_x)
        x_difference.unsafe_store(i, normalized_x)
        y_normalized.unsafe_store(i, normalized_y)
        y_difference.unsafe_store(i, normalized_y - normalized_x)
    for i in range(vector_stop, stop):
        var normalized_x = (x[unsafe_offset=i] - x_min) / x_span
        var normalized_y = (smooth_y[unsafe_offset=n - 1 - i] - y_min) / y_span
        if invert_y:
            normalized_y = 1.0 - normalized_y
        x_normalized[unsafe_offset=i] = normalized_x
        x_difference[unsafe_offset=i] = normalized_x
        y_normalized[unsafe_offset=i] = normalized_y
        y_difference[unsafe_offset=i] = normalized_y - normalized_x


def local_maximum(values: FPtr, i: Int, n: Int) -> Bool:
    var left = i - 1
    var right = i + 1
    if left < 0:
        left = 0
    if right >= n:
        right = n - 1
    return values[unsafe_offset=i] >= values[unsafe_offset=left] and (
        values[unsafe_offset=i] >= values[unsafe_offset=right]
    )


def local_minimum(values: FPtr, i: Int, n: Int) -> Bool:
    var left = i - 1
    var right = i + 1
    if left < 0:
        left = 0
    if right >= n:
        right = n - 1
    return values[unsafe_offset=i] <= values[unsafe_offset=left] and (
        values[unsafe_offset=i] <= values[unsafe_offset=right]
    )


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
    x_difference_addr: Int,
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
        x_addr == 0
        or smooth_y_addr == 0
        or x_normalized_addr == 0
        or x_difference_addr == 0
        or y_normalized_addr == 0
        or y_difference_addr == 0
        or maxima_addr == 0
        or minima_addr == 0
        or thresholds_addr == 0
        or knee_indices_addr == 0
        or norm_knee_indices_addr == 0
        or counts_addr == 0
    ):
        return 2
    if (
        (curve != 0 and curve != 1)
        or (direction != 0 and direction != 1)
        or (online != 0 and online != 1)
    ):
        return 3

    var x = FPtr(unsafe_from_address=x_addr)
    var smooth_y = FPtr(unsafe_from_address=smooth_y_addr)
    var x_normalized = FPtr(unsafe_from_address=x_normalized_addr)
    var x_difference = FPtr(unsafe_from_address=x_difference_addr)
    var y_normalized = FPtr(unsafe_from_address=y_normalized_addr)
    var y_difference = FPtr(unsafe_from_address=y_difference_addr)
    var maxima = IPtr(unsafe_from_address=maxima_addr)
    var minima = IPtr(unsafe_from_address=minima_addr)
    var thresholds = FPtr(unsafe_from_address=thresholds_addr)
    var knee_indices = IPtr(unsafe_from_address=knee_indices_addr)
    var norm_knee_indices = IPtr(unsafe_from_address=norm_knee_indices_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)

    comptime W = simd_width_of[DType.float64]()
    var x_min = x[unsafe_offset=0]
    var x_max = x_min
    var y_min = smooth_y[unsafe_offset=0]
    var y_max = y_min
    if x_min != x_min or y_min != y_min:
        for i in range(1, n):
            if x[unsafe_offset=i] < x_min:
                x_min = x[unsafe_offset=i]
            if x[unsafe_offset=i] > x_max:
                x_max = x[unsafe_offset=i]
            if smooth_y[unsafe_offset=i] < y_min:
                y_min = smooth_y[unsafe_offset=i]
            if smooth_y[unsafe_offset=i] > y_max:
                y_max = smooth_y[unsafe_offset=i]
    else:
        var x_min_vector = SIMD[DType.float64, W](x_min)
        var x_max_vector = x_min_vector
        var y_min_vector = SIMD[DType.float64, W](y_min)
        var y_max_vector = y_min_vector
        var range_vector_stop = (n // W) * W
        for i in range(0, range_vector_stop, W):
            var x_values = x.unsafe_load[width=W](i)
            var y_values = smooth_y.unsafe_load[width=W](i)
            x_min_vector = min(x_min_vector, x_values)
            x_max_vector = max(x_max_vector, x_values)
            y_min_vector = min(y_min_vector, y_values)
            y_max_vector = max(y_max_vector, y_values)

        x_min = x_min_vector[0]
        x_max = x_max_vector[0]
        y_min = y_min_vector[0]
        y_max = y_max_vector[0]
        comptime for lane in range(1, W):
            x_min = min(x_min, x_min_vector[lane])
            x_max = max(x_max, x_max_vector[lane])
            y_min = min(y_min, y_min_vector[lane])
            y_max = max(y_max, y_max_vector[lane])
        for i in range(range_vector_stop, n):
            x_min = min(x_min, x[unsafe_offset=i])
            x_max = max(x_max, x[unsafe_offset=i])
            y_min = min(y_min, smooth_y[unsafe_offset=i])
            y_max = max(y_max, smooth_y[unsafe_offset=i])

    var x_span = x_max - x_min
    var y_span = y_max - y_min
    var reverse_y = (
        direction == 1 and curve == 1 or (direction == 0 and curve == 0)
    )
    var invert_y = curve == 0
    if n >= PARALLEL_THRESHOLD:
        var task_count = (n + PARALLEL_CHUNK_SIZE - 1) // PARALLEL_CHUNK_SIZE

        @__parameter
        def normalize_task(task: Int):
            var start = task * PARALLEL_CHUNK_SIZE
            var stop = min(start + PARALLEL_CHUNK_SIZE, n)
            if reverse_y:
                normalize_reverse(
                    x,
                    smooth_y,
                    x_normalized,
                    x_difference,
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
                    x_difference,
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

        initialize_runtime()
        parallelize[normalize_task](task_count, min(task_count, 8))
    elif reverse_y:
        normalize_reverse(
            x,
            smooth_y,
            x_normalized,
            x_difference,
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
            x_difference,
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
    var spacing_vector_stop = ((n - 1) // W) * W
    for i in range(0, spacing_vector_stop, W):
        mean_spacing += abs(
            x_normalized.unsafe_load[width=W](i + 1)
            - x_normalized.unsafe_load[width=W](i)
        ).reduce_add()
    for i in range(spacing_vector_stop, n - 1):
        mean_spacing += abs(
            x_normalized[unsafe_offset=i + 1] - x_normalized[unsafe_offset=i]
        )
    mean_spacing /= Float64(n - 1)

    var maxima_count = 0
    var minima_count = 0
    if local_maximum(y_difference, 0, n):
        maxima[unsafe_offset=maxima_count] = 0
        thresholds[unsafe_offset=maxima_count] = (
            y_difference[unsafe_offset=0] - sensitivity * mean_spacing
        )
        maxima_count += 1
    if local_minimum(y_difference, 0, n):
        minima[unsafe_offset=minima_count] = 0
        minima_count += 1

    var extrema_vector_stop = 1 + ((n - 2) // W) * W
    for i in range(1, extrema_vector_stop, W):
        var center = y_difference.unsafe_load[width=W](i)
        var is_maximum = center.ge(
            y_difference.unsafe_load[width=W](i - 1)
        ) & center.ge(y_difference.unsafe_load[width=W](i + 1))
        var is_minimum = center.le(
            y_difference.unsafe_load[width=W](i - 1)
        ) & center.le(y_difference.unsafe_load[width=W](i + 1))
        comptime for lane in range(W):
            if is_maximum[lane]:
                maxima[unsafe_offset=maxima_count] = Int64(i + lane)
                thresholds[unsafe_offset=maxima_count] = (
                    center[lane] - sensitivity * mean_spacing
                )
                maxima_count += 1
            if is_minimum[lane]:
                minima[unsafe_offset=minima_count] = Int64(i + lane)
                minima_count += 1

    for i in range(extrema_vector_stop, n - 1):
        if local_maximum(y_difference, i, n):
            maxima[unsafe_offset=maxima_count] = Int64(i)
            thresholds[unsafe_offset=maxima_count] = (
                y_difference[unsafe_offset=i] - sensitivity * mean_spacing
            )
            maxima_count += 1
        if local_minimum(y_difference, i, n):
            minima[unsafe_offset=minima_count] = Int64(i)
            minima_count += 1

    if local_maximum(y_difference, n - 1, n):
        maxima[unsafe_offset=maxima_count] = Int64(n - 1)
        thresholds[unsafe_offset=maxima_count] = (
            y_difference[unsafe_offset=n - 1] - sensitivity * mean_spacing
        )
        maxima_count += 1
    if local_minimum(y_difference, n - 1, n):
        minima[unsafe_offset=minima_count] = Int64(n - 1)
        minima_count += 1

    counts[unsafe_offset=0] = Int64(maxima_count)
    counts[unsafe_offset=1] = Int64(minima_count)
    counts[unsafe_offset=2] = 0
    if maxima_count == 0:
        return 0

    var max_cursor = 0
    var min_cursor = 0
    var threshold = 0.0
    var threshold_index = 0
    var detection_active = True
    var knee_count = 0
    var i = Int(maxima[unsafe_offset=0])
    while i < n - 1:
        if max_cursor < maxima_count and (
            Int(maxima[unsafe_offset=max_cursor]) == i
        ):
            threshold = thresholds[unsafe_offset=max_cursor]
            threshold_index = i
            max_cursor += 1
            detection_active = True
        if min_cursor < minima_count and (
            Int(minima[unsafe_offset=min_cursor]) == i
        ):
            threshold = 0.0
            min_cursor += 1
            detection_active = False

        if detection_active and (y_difference[unsafe_offset=i + 1] < threshold):
            var raw_index = threshold_index
            if (curve == 0 and direction == 0) or (
                curve == 1 and direction == 1
            ):
                raw_index = n - 1 - threshold_index
            knee_indices[unsafe_offset=knee_count] = Int64(raw_index)
            norm_knee_indices[unsafe_offset=knee_count] = Int64(threshold_index)
            knee_count += 1
            if online == 0:
                break
        i += 1

    counts[unsafe_offset=2] = Int64(knee_count)
    return 0
