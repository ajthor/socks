def generate_batches(num_elements=1, batch_size=1):

    start = 0

    for _ in range(num_elements // batch_size):
        end = start + batch_size

        yield slice(start, end)

        start = end

    if start < num_elements:
        yield slice(start, n)
