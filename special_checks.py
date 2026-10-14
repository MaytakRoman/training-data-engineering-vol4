"""Контроль специальных наборов без внешних зависимостей.
Метки времени задаются целыми шагами в одной согласованной шкале.
Запуск: python3 special_checks.py
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Measurement:
    id: str
    asset: str
    event_at: int
    available_at: int
    value: float


@dataclass(frozen=True)
class Window:
    id: str
    asset: str
    observed_ids: tuple[str, ...]
    forecast_at: int
    target_end: int
    split: str


@dataclass(frozen=True)
class Preference:
    id: str
    family: str
    prompt: str
    answer_a: str
    answer_b: str
    verdict: str
    instruction_version: str
    split: str


@dataclass(frozen=True)
class Event:
    id: str
    episode: str
    sequence: int
    kind: str
    parent: str | None
    intent: str | None = None
    effect: str | None = None


def validate_measurements(rows: list[Measurement]) -> dict[str, Measurement]:
    indexed = {}
    for row in rows:
        if row.id in indexed:
            raise ValueError("duplicate measurement id")
        if row.available_at < row.event_at:
            raise ValueError("availability precedes event")
        indexed[row.id] = row
    return indexed


def validate_windows(
    rows: list[Window], measurements: dict[str, Measurement],
    train_end: int, strict_disjoint: bool = True,
) -> dict:
    used_by_split = defaultdict(set)
    ids = set()
    for row in rows:
        if row.id in ids:
            raise ValueError("duplicate window id")
        ids.add(row.id)
        if row.split not in ("train", "test"):
            raise ValueError("unknown window split")
        if row.target_end <= row.forecast_at:
            raise ValueError("invalid target interval")
        if not row.observed_ids:
            raise ValueError("empty input window")
        if len(set(row.observed_ids)) != len(row.observed_ids):
            raise ValueError("repeated input measurement")
        for mid in row.observed_ids:
            if mid not in measurements:
                raise ValueError("missing input measurement")
            item = measurements[mid]
            if item.asset != row.asset:
                raise ValueError("mixed assets")
            if item.available_at > row.forecast_at:
                raise ValueError("future information")
            used_by_split[row.split].add(mid)
        if row.split == "train" and row.target_end > train_end:
            raise ValueError("training target crosses boundary")
        if row.split == "test" and row.forecast_at <= train_end:
            raise ValueError("test forecast before boundary")
    overlap = used_by_split["train"] & used_by_split["test"]
    if strict_disjoint and overlap:
        raise ValueError("shared input measurements")
    return {"n_windows": len(rows), "shared_inputs": len(overlap)}


def validate_preferences(rows: list[Preference]) -> dict:
    allowed = {"a", "b", "tie", "both_bad", "insufficient_context"}
    ids, families, counts = set(), {}, defaultdict(int)
    for row in rows:
        if row.id in ids:
            raise ValueError("duplicate preference id")
        ids.add(row.id)
        if not row.prompt.strip():
            raise ValueError("empty prompt")
        if not row.answer_a.strip() or not row.answer_b.strip():
            raise ValueError("empty alternative")
        if row.answer_a == row.answer_b:
            raise ValueError("identical alternatives")
        if row.verdict not in allowed:
            raise ValueError("unknown verdict")
        if not row.instruction_version:
            raise ValueError("missing instruction version")
        if row.split not in ("train", "validation", "test"):
            raise ValueError("unknown preference split")
        old = families.get(row.family)
        if old is not None and old != row.split:
            raise ValueError("preference family crosses split")
        families[row.family] = row.split
        counts[row.verdict] += 1
    return {"n_pairs": len(rows), "n_families": len(families),
            "verdicts": dict(counts)}


def validate_events(rows: list[Event]) -> dict:
    ids = {}
    sequences = set()
    effects = defaultdict(set)
    attempts = defaultdict(int)
    allowed = {"observe", "decide", "call", "result", "finish"}
    ordered = sorted(rows, key=lambda r: (r.episode, r.sequence))
    for row in ordered:
        if row.id in ids:
            raise ValueError("duplicate event id")
        pair = row.episode, row.sequence
        if pair in sequences:
            raise ValueError("duplicate sequence")
        sequences.add(pair)
        if row.kind not in allowed:
            raise ValueError("invalid event kind")
        if row.parent is not None:
            if row.parent not in ids:
                raise ValueError("missing or later parent")
            parent = ids[row.parent]
            if parent.episode != row.episode:
                raise ValueError("cross-episode parent")
        if row.kind == "call":
            if not row.intent:
                raise ValueError("call without intent")
            attempts[row.intent] += 1
        if row.effect:
            if row.kind != "result" or not row.intent:
                raise ValueError("effect without result and intent")
            effects[row.intent].add(row.effect)
        ids[row.id] = row
    return {"n_events": len(rows), "attempts": dict(attempts),
            "effects": {key: len(value) for key, value in effects.items()}}


def must_fail(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


def self_test() -> None:
    measurements = validate_measurements([
        Measurement("m1", "A", 1, 1, 10.0),
        Measurement("m2", "A", 2, 8, 11.0),
        Measurement("m3", "A", 12, 12, 12.0),
    ])
    train = Window("w1", "A", ("m1",), 2, 4, "train")
    test = Window("w2", "A", ("m3",), 13, 15, "test")
    assert validate_windows([train, test], measurements, 10)["shared_inputs"] == 0
    late = Window("w3", "A", ("m2",), 4, 6, "train")
    must_fail(validate_windows, [late], measurements, 10)
    crossing = Window("w4", "A", ("m1",), 9, 11, "train")
    must_fail(validate_windows, [crossing], measurements, 10)
    same_input = Window("w5", "A", ("m1",), 13, 15, "test")
    must_fail(validate_windows, [train, same_input], measurements, 10)
    p1 = Preference("p1", "f1", "Вопрос", "Ответ А", "Ответ Б",
                    "both_bad", "v1", "train")
    p2 = Preference("p2", "f1", "Другой вопрос", "Иной А", "Иной Б",
                    "a", "v1", "test")
    assert validate_preferences([p1])["verdicts"]["both_bad"] == 1
    must_fail(validate_preferences, [p1, p2])
    events = [
        Event("e1", "ep1", 1, "observe", None),
        Event("e2", "ep1", 2, "call", "e1", "intent1"),
        Event("e3", "ep1", 3, "result", "e2", "intent1", "object1"),
        Event("e4", "ep1", 4, "call", "e3", "intent1"),
        Event("e5", "ep1", 5, "result", "e4", "intent1", "object1"),
    ]
    result = validate_events(events)
    assert result["attempts"]["intent1"] == 2
    assert result["effects"]["intent1"] == 1
    invalid = [Event("x", "ep1", 1, "result", "absent")]
    must_fail(validate_events, invalid)
    print("special-data checks passed")


if __name__ == "__main__":
    self_test()
