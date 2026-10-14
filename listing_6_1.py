from decimal import Decimal, ROUND_FLOOR

def allocate(total, weights):
    if type(total) is not int or total < 0:
        raise ValueError("Бюджет должен быть целым и неотрицательным")
    values = [Decimal(str(w)) for w in weights]
    if not values or any(not w.is_finite() or w < 0 for w in values):
        raise ValueError("Недопустимые веса")
    weight_sum = sum(values)
    if weight_sum <= 0:
        raise ValueError("Сумма весов должна быть положительной")
    quotas = [Decimal(total) * w / weight_sum for w in values]
    result = [int(q.to_integral_value(rounding=ROUND_FLOOR))
              for q in quotas]
    left = total - sum(result)
    order = sorted(range(len(values)),
                   key=lambda i: (-(quotas[i] - result[i]), i))
    for i in order[:left]:
        result[i] += 1
    assert sum(result) == total
    return result

assert allocate(101, [0.50, 0.20, 0.15, 0.15]) == [51, 20, 15, 15]
assert allocate(0, [1, 2]) == [0, 0]

