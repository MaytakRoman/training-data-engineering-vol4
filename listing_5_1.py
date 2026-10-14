from fractions import Fraction
from random import Random

def linear_tasks(count, seed=20261013):
    rng = Random(seed)
    tasks = []
    for index in range(count):
        a = rng.choice([v for v in range(-9, 10) if v != 0])
        b = rng.randint(-20, 20)
        x = Fraction(rng.randint(-15, 15), rng.randint(1, 9))
        c = a * x + b
        answer = (c - b) / a
        assert a * answer + b == c
        tasks.append({
            "id": f"linear-{index:04d}",
            "family": "linear-rational-v1",
            "a": a, "b": b,
            "c": str(c), "answer": str(answer),
        })
    return tasks

tasks = linear_tasks(100)
assert len(tasks) == 100
assert all(Fraction(t["answer"]) * t["a"] + t["b"]
           == Fraction(t["c"]) for t in tasks)

