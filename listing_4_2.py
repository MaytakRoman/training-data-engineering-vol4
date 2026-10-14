def validate_label(record, sources, versions):
    labels = {"sufficient", "clarify", "absent"}
    if record.get("label") not in labels:
        raise ValueError("Неизвестная метка")
    if record.get("instruction_version") not in versions:
        raise ValueError("Неизвестная инструкция")
    source_id = record.get("source_id")
    if source_id not in sources:
        raise ValueError("Источник не восстановлен")
    evidence = record.get("evidence", [])
    text = sources[source_id]
    for span in evidence:
        start, end = span
        if type(start) is not int or type(end) is not int:
            raise ValueError("Координаты должны быть целыми")
        if not 0 <= start < end <= len(text):
            raise ValueError("Диапазон вне снимка")
    if record["label"] == "sufficient" and not evidence:
        raise ValueError("Нет свидетельства достаточности")
    if record["label"] == "clarify" and not record.get("missing_parameter"):
        raise ValueError("Не указан параметр уточнения")
    if record["label"] == "absent" and not record.get("reason"):
        raise ValueError("Не указана причина отсутствия")
    return True

sources = {"manual-v1": "Для модификации А момент равен 12 Н·м."}
record = {
    "label": "sufficient", "instruction_version": "1.0",
    "source_id": "manual-v1", "evidence": [(0, 37)],
}
assert validate_label(record, sources, {"1.0"})

