"""Воспроизводимый синтетический стенд подготовки технического корпуса.
Запуск: python3 corpus_lab.py --out lab_results
Только стандартная библиотека Python. Сетевых обращений нет.
Все документы вымышлены. Классификатор: multinomial naive Bayes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

DOMAINS = {
    "pump": ("насос", "крыльчатка", "расход", "кавитация", "уплотнение"),
    "motor": ("двигатель", "обмотка", "ротор", "статор", "изоляция"),
    "valve": ("клапан", "седло", "затвор", "герметичность", "привод"),
    "sensor": ("датчик", "калибровка", "сигнал", "погрешность", "диапазон"),
    "bearing": ("подшипник", "смазка", "вибрация", "сепаратор", "зазор"),
}
NAMES = tuple(DOMAINS)
SEEDS = (17, 29, 43, 71, 101)
STAGES = ("raw", "extract", "dedup", "quality")


@dataclass(frozen=True)
class Record:
    id: str
    family: str
    domain: str
    body: str
    navigation: str
    damaged: bool
    split: str
    copy_of: str | None = None

    def text(self, stage: str) -> str:
        if stage == "raw":
            return self.navigation + " " + self.body
        return self.body


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokens(text: str) -> list[str]:
    return re.findall(r"[а-яё]+", text.lower())


def generate(seed: int) -> tuple[list[Record], list[Record]]:
    """Пять классов; семейства train/test создаются раздельно.
    Повреждение подменяет техническую область тела. Флаг известен
    генератору и имитирует идеальную проверку типа документа.
    """
    rng = random.Random(seed)
    train, test = [], []
    for domain in NAMES:
        for split, count in (("train", 400), ("test", 120)):
            for i in range(count):
                rid = f"{split}-{domain}-{i:04d}"
                damaged = split == "train" and rng.random() < 0.30
                actual = rng.choice(NAMES) if damaged else domain
                words = list(DOMAINS[actual])
                body_words = rng.choices(words, k=rng.randint(3, 9))
                if rng.random() < 0.45:
                    other = rng.choice(NAMES)
                    body_words += rng.choices(DOMAINS[other], k=rng.randint(5, 11))
                body = (
                    f"Документ {rid}. Перед проверкой отключить питание. "
                    + " ".join(body_words)
                    + f". Предел {rng.randint(10, 99)}."
                )
                # Условный шаблон сайта противопоставлен классу страницы.
                nav_domain = NAMES[(NAMES.index(domain) + 1) % len(NAMES)]
                nav = " ".join(rng.choices(DOMAINS[nav_domain], k=40))
                rec = Record(rid, rid, domain, body, nav, damaged, split)
                (train if split == "train" else test).append(rec)
                # Один домен имеет больше зеркал: пример смещения весов.
                copies = 4 if domain == "pump" and i % 3 == 0 else 0
                if split == "train":
                    for j in range(copies):
                        train.append(Record(
                            f"{rid}-copy-{j}", rid, domain, body, nav,
                            damaged, split, rid,
                        ))
    return train, test


def prepare(records: list[Record], stage: str) -> list[Record]:
    if stage not in STAGES:
        raise ValueError("unknown stage")
    if stage in ("raw", "extract"):
        return list(records)
    seen = set()
    result = []
    for record in records:
        key = digest(record.body)
        if key in seen:
            continue
        seen.add(key)
        if stage == "quality" and record.damaged:
            continue
        result.append(record)
    return result


class NB:
    """Прозрачный мультиномиальный классификатор со сглаживанием 1."""
    def __init__(self) -> None:
        self.counts = {name: Counter() for name in NAMES}
        self.documents = Counter()
        self.vocabulary = set()
        self.total = {}
        self.n = 0

    def fit(self, records: list[Record], stage: str) -> "NB":
        for record in records:
            ts = tokens(record.text(stage))
            self.counts[record.domain].update(ts)
            self.documents[record.domain] += 1
            self.vocabulary.update(ts)
        self.n = len(records)
        if not self.n:
            raise ValueError("empty training corpus")
        self.total = {
            name: sum(self.counts[name].values()) for name in NAMES
        }
        return self

    def predict(self, text: str) -> str:
        x = Counter(t for t in tokens(text) if t in self.vocabulary)
        v = len(self.vocabulary)
        scores = {}
        for name in NAMES:
            prior = (self.documents[name] + 1) / (self.n + len(NAMES))
            score = math.log(prior)
            denominator = self.total[name] + v
            for token, frequency in x.items():
                probability = (self.counts[name][token] + 1) / denominator
                score += frequency * math.log(probability)
            scores[name] = score
        return max(NAMES, key=lambda name: scores[name])


def metrics(truth: list[str], pred: list[str]) -> dict:
    if not truth or len(truth) != len(pred):
        raise ValueError("invalid evaluation arrays")
    matrix = {name: Counter() for name in NAMES}
    for y, p in zip(truth, pred):
        matrix[y][p] += 1
    f1, recall = [], {}
    for name in NAMES:
        tp = matrix[name][name]
        fn = sum(matrix[name].values()) - tp
        fp = sum(matrix[other][name] for other in NAMES if other != name)
        den = 2 * tp + fp + fn
        f1.append(2 * tp / den if den else 0.0)
        recall[name] = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": sum(a == b for a, b in zip(truth, pred)) / len(truth),
        "macro_f1": statistics.mean(f1),
        "recall": recall,
        "matrix": {k: dict(v) for k, v in matrix.items()},
    }


def wilson(success: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0 or not 0 <= success <= n:
        raise ValueError("invalid binomial count")
    p = success / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - radius, centre + radius


def bootstrap_difference(
    a: list[bool], b: list[bool], seed: int = 2026, repeats: int = 5000,
) -> tuple[float, float]:
    """Парный bootstrap по независимым тестовым семействам.
    В стенде на семейство приходится один объект.
    """
    if len(a) != len(b) or not a:
        raise ValueError("invalid paired arrays")
    rng = random.Random(seed)
    diff = [int(y) - int(x) for x, y in zip(a, b)]
    n = len(diff)
    samples = sorted(
        sum(diff[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(repeats)
    )
    return samples[int(0.025 * repeats)], samples[int(0.975 * repeats) - 1]


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in SEEDS:
        train, test = generate(seed)
        assert not {r.family for r in train} & {r.family for r in test}
        predictions = {}
        for stage in STAGES:
            corpus = prepare(train, stage)
            model = NB().fit(corpus, stage)
            # Общий тест содержит очищенные тела для всех вариантов.
            pred = [model.predict(r.body) for r in test]
            truth = [r.domain for r in test]
            predictions[stage] = [p == y for p, y in zip(pred, truth)]
            result = metrics(truth, pred)
            result.update({
                "seed": seed, "stage": stage, "n_train": len(corpus),
                "n_test": len(test),
                "n_tokens": sum(len(tokens(r.text(stage))) for r in corpus),
                "n_damaged": sum(r.damaged for r in corpus),
                "n_families": len({r.family for r in corpus}),
            })
            runs.append(result)
        if seed == SEEDS[0]:
            for name, records in (("raw_train", train), ("test", test)):
                path = out / f"{name}.jsonl"
                with path.open("w", encoding="utf-8") as file:
                    for record in records:
                        file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            paired = {
                "raw_to_quality": bootstrap_difference(
                    predictions["raw"], predictions["quality"],
                )
            }
    summary = []
    for stage in STAGES:
        rows = [r for r in runs if r["stage"] == stage]
        summary.append({
            "stage": stage,
            "mean_accuracy": statistics.mean(r["accuracy"] for r in rows),
            "sd_accuracy": statistics.stdev(r["accuracy"] for r in rows),
            "mean_f1": statistics.mean(r["macro_f1"] for r in rows),
            "mean_train": statistics.mean(r["n_train"] for r in rows),
            "mean_tokens": statistics.mean(r["n_tokens"] for r in rows),
        })
    report = {
        "design": {"seeds": SEEDS, "test_per_class": 120,
                   "train_families_per_class": 400,
                   "model": "multinomial NB alpha=1", "synthetic": True},
        "runs": runs, "summary": summary, "paired_first_seed": paired,
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def self_test() -> None:
    train, test = generate(17)
    assert len(test) == 600
    assert len({r.family for r in train}) == 2000
    dedup = prepare(train, "dedup")
    assert len(dedup) == 2000
    assert all(not r.damaged for r in prepare(train, "quality"))
    assert prepare(dedup, "dedup") == dedup
    assert not {r.family for r in train} & {r.family for r in test}
    low, high = wilson(0, 100)
    assert abs(low) < 1e-12 and 0.03 < high < 0.04
    result = metrics(["pump", "motor"], ["pump", "motor"])
    assert result["accuracy"] == 1.0
    assert digest("а") != digest("a")
    try:
        prepare(train, "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown stage must fail")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("lab_results"))
    args = parser.parse_args()
    self_test()
    report = run(args.out)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
