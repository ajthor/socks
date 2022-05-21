from collections.abc import Sequence


def batch_generator(s: Sequence, size: int):
    """Generate batches.

    Batch generation function to split a sequence into smaller batches (slices).
    Generates `slice` objects, which can be iterated over in a for loop.

    Args:
        s: A sequence.
        size: Size of the batches. If the last batch is smaller than `size`, then the
            final batch will be the length of the remaining elements.

    Yields:
        A slice object.

    """

    start = 0

    for _ in range(len(s) // size):
        end = start + size

        yield slice(start, end)

        start = end

    if start < len(s):
        yield slice(start, len(s))
