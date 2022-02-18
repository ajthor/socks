def generate_batches(num_elements, batch_size):
    """Generate batches.

    Batch generation function to split a list into smaller batches (slices). Generates
    `slice` objects, which can be iterated over in a for loop.

    Args:
        num_elements: Length of the list.
        batch_size: Maximum size of the batches. If the last batch is smaller than
            `batch_size`, then the final batch will be the length of the remaining
            elements.

    Yields:
        A slice object.

    """

    start = 0

    for _ in range(num_elements // batch_size):
        end = start + batch_size

        yield slice(start, end)

        start = end

    if start < num_elements:
        yield slice(start, num_elements)
