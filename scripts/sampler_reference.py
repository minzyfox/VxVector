"""An independent Python copy of the VxVector training-row sampler.

src/kmeans.vx chooses min(rows, k * sampling_factor) rows. It uses every row in
order when that is all of them, and otherwise SplitMix64 with Floyd's algorithm,
in the order of selection. The tests compare the native library with this copy.
"""

MASK = (1 << 64) - 1


def splitmix64(state):
    """Return (value, next state)."""
    state = (state + 0x9E3779B97F4A7C15) & MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31), state


def random_below(state, bound):
    threshold = ((1 << 64) - bound) % bound
    value, state = splitmix64(state)
    while value < threshold:
        value, state = splitmix64(state)
    return value % bound, state


def floyd(rows, count, seed):
    """count distinct rows of range(rows), in the order of selection."""
    chosen, taken, state = [], set(), seed
    for i in range(count):
        last = rows - count + i
        row, state = random_below(state, last + 1)
        if row in taken:
            row = last
        taken.add(row)
        chosen.append(row)
    return chosen


def training_rows(rows, k, sampling_factor, seed):
    count = min(rows, k * sampling_factor)
    return list(range(rows)) if count == rows else floyd(rows, count, seed)


def table_slots(count):
    """The sampler table capacity that vxvector_kmeans_v2 requires."""
    slots = 1
    while slots < 2 * count:
        slots *= 2
    return slots
