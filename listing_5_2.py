from random import Random
from collections import Counter
from math import log, ceil

def categorical_release(weights, count, seed):
    if count < 0 or not weights or any(w < 0 for w in weights):
        raise ValueError("Недопустимый план")
    if sum(weights) <= 0:
        raise ValueError("Пустое распределение")
    rng = Random(seed)
    indices = list(range(len(weights)))
    counts = Counter(rng.choices(indices, weights=weights, k=count))
    missing = [i for i, w in enumerate(weights)
               if w > 0 and counts[i] == 0]
    return dict(counts), missing

counts, missing = categorical_release([0.999, 0.001], 1000, 17)
assert sum(counts.values()) == 1000
assert categorical_release([1, 0], 10, 17)[1] == []
required = ceil(log(0.05) / log(0.999))
assert required == 2995

