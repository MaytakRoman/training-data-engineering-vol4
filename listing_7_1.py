def split_conflicts(records):
    allowed = {"train", "development", "test"}
    groups = {}
    for record in records:
        group = record.get("group_id")
        split = record.get("split")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Не определена группа происхождения")
        if split not in allowed:
            raise ValueError("Неизвестный раздел")
        groups.setdefault(group, set()).add(split)
    return {g: sorted(parts) for g, parts in groups.items()
            if len(parts) > 1}

records = [
    {"id": "a", "group_id": "manual-17", "split": "train"},
    {"id": "b", "group_id": "manual-17", "split": "test"},
    {"id": "c", "group_id": "manual-18", "split": "test"},
]
assert split_conflicts(records) == {
    "manual-17": ["test", "train"]
}

