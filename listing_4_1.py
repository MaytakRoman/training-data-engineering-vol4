from math import isclose

def agreement(matrix):
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("Матрица должна быть квадратной")
    if any(type(v) is not int or v < 0
           for row in matrix for v in row):
        raise ValueError("Нужны неотрицательные целые счётчики")
    total = sum(sum(row) for row in matrix)
    if total == 0:
        raise ValueError("Нет совместных суждений")
    rows = [sum(row) / total for row in matrix]
    cols = [sum(matrix[i][j] for i in range(size)) / total
            for j in range(size)]
    observed = sum(matrix[i][i] for i in range(size)) / total
    expected = sum(a * b for a, b in zip(rows, cols))
    kappa = None if isclose(expected, 1.0) else (
        (observed - expected) / (1.0 - expected)
    )
    return observed, expected, kappa

result = agreement([[130, 20], [10, 40]])
assert isclose(result[0], 0.85)
assert isclose(result[1], 0.60)
assert isclose(result[2], 0.625)
assert agreement([[20, 0], [0, 0]])[2] is None

