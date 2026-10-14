"""Локальный учебный каталог неизменяемых объектов и выпусков.
Запуск: python3 release_store.py
Требования: стандартная библиотека Python, одна файловая система.
Согласованная блокировка SQLite охватывает публикацию через этот API.
Обход API сторонним процессом не защищён данной реализацией.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from typing import Iterator
from dataclasses import dataclass
from pathlib import Path


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_key(key: str) -> str:
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("expected complete hexadecimal SHA256")
    return key


@dataclass(frozen=True)
class Stored:
    digest: str
    size: int


class Conflict(RuntimeError):
    pass


class IntegrityError(RuntimeError):
    pass


class ReleaseStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "catalog.sqlite"
        with self.connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                    digest TEXT PRIMARY KEY,
                    size INTEGER NOT NULL CHECK(size >= 0)
                );
                CREATE TABLE IF NOT EXISTS releases (
                    digest TEXT PRIMARY KEY,
                    manifest BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS current_release (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    digest TEXT REFERENCES releases(digest)
                );
                INSERT OR IGNORE INTO current_release VALUES (1, NULL);
                CREATE TABLE IF NOT EXISTS publication_log (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    previous_digest TEXT,
                    next_digest TEXT NOT NULL,
                    reason TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.db, timeout=30)) as con:
            con.execute("PRAGMA foreign_keys = ON")
            with con:
                yield con

    def object_path(self, key: str) -> Path:
        return self.objects / safe_key(key)

    def put(self, data: bytes) -> Stored:
        key = sha256(data)
        path = self.object_path(key)
        if path.exists():
            self.verify(key, len(data))
        else:
            fd, name = tempfile.mkstemp(prefix="pending-", dir=self.objects)
            temp = Path(name)
            try:
                with os.fdopen(fd, "wb") as file:
                    file.write(data)
                    file.flush()
                    os.fsync(file.fileno())
                # Одновременные авторы по одному ключу пишут одинаковые байты.
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
        with self.connect() as con:
            row = con.execute(
                "SELECT size FROM objects WHERE digest = ?", (key,),
            ).fetchone()
            if row is not None and row[0] != len(data):
                raise IntegrityError("catalog size conflict")
            con.execute(
                "INSERT OR IGNORE INTO objects VALUES (?, ?)",
                (key, len(data)),
            )
        return Stored(key, len(data))

    def verify(self, key: str, size: int | None = None) -> bytes:
        path = self.object_path(key)
        if not path.is_file():
            raise IntegrityError("missing object")
        data = path.read_bytes()
        if sha256(data) != key:
            raise IntegrityError("hash mismatch")
        if size is not None and len(data) != size:
            raise IntegrityError("size mismatch")
        return data

    def prepare(self, manifest: dict) -> str:
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported manifest schema")
        records = manifest.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("nonempty records required")
        ids = set()
        groups = {}
        for record in records:
            rid = record["id"]
            if rid in ids:
                raise ValueError("duplicate logical id")
            ids.add(rid)
            split = record["split"]
            if split not in ("train", "validation", "test"):
                raise ValueError("invalid split")
            family = record["family"]
            if family in groups and groups[family] != split:
                raise ValueError("family crosses split boundary")
            groups[family] = split
            self.verify(record["digest"], record["size"])
        data = canonical(manifest)
        key = sha256(data)
        with self.connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO releases VALUES (?, ?)",
                (key, data),
            )
        return key

    def current(self) -> str | None:
        with self.connect() as con:
            return con.execute(
                "SELECT digest FROM current_release WHERE singleton = 1",
            ).fetchone()[0]

    def publish(self, key: str, expected: str | None, reason: str) -> None:
        safe_key(key)
        if not reason.strip():
            raise ValueError("publication reason required")
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT manifest FROM releases WHERE digest = ?", (key,),
            ).fetchone()
            if row is None:
                raise ValueError("release not prepared")
            current = con.execute(
                "SELECT digest FROM current_release WHERE singleton = 1",
            ).fetchone()[0]
            if current == key:
                # Уже исполненный повтор не создаёт вторую публикацию.
                return
            if current != expected:
                raise Conflict("current release changed")
            # Повторная проверка наличия выполняется перед публикацией.
            manifest = json.loads(row[0])
            for record in manifest["records"]:
                self.verify(record["digest"], record["size"])
            con.execute(
                "UPDATE current_release SET digest = ? WHERE singleton = 1",
                (key,),
            )
            con.execute(
                "INSERT INTO publication_log "
                "(previous_digest, next_digest, reason) VALUES (?, ?, ?)",
                (current, key, reason),
            )

    def read_release(self, key: str | None = None) -> dict:
        key = key if key is not None else self.current()
        if key is None:
            raise ValueError("no release published")
        with self.connect() as con:
            row = con.execute(
                "SELECT manifest FROM releases WHERE digest = ?", (key,),
            ).fetchone()
        if row is None or sha256(row[0]) != key:
            raise IntegrityError("invalid manifest")
        return json.loads(row[0])

    def audit(self) -> dict:
        missing, corrupt, valid = [], [], []
        with self.connect() as con:
            rows = con.execute("SELECT digest, size FROM objects").fetchall()
        for key, size in rows:
            try:
                self.verify(key, size)
            except IntegrityError as exc:
                target = missing if not self.object_path(key).exists() else corrupt
                target.append({"digest": key, "reason": str(exc)})
            else:
                valid.append(key)
        known = {key for key, _ in rows}
        orphans = [
            path.name for path in self.objects.iterdir()
            if path.is_file() and path.name not in known
        ]
        return {"valid": valid, "missing": missing,
                "corrupt": corrupt, "unregistered": orphans}


def demonstration() -> None:
    with tempfile.TemporaryDirectory() as folder:
        store = ReleaseStore(Path(folder))
        a = store.put("Первая инструкция".encode("utf-8"))
        b = store.put("Вторая инструкция".encode("utf-8"))
        assert store.put("Первая инструкция".encode("utf-8")) == a
        base = {"schema_version": 1, "transform": "demo-v1", "records": [
            {"id": "a", "family": "fa", "split": "train",
             "digest": a.digest, "size": a.size},
            {"id": "b", "family": "fb", "split": "test",
             "digest": b.digest, "size": b.size},
        ]}
        first = store.prepare(base)
        store.publish(first, None, "initial release")
        store.publish(first, None, "retry")
        assert store.current() == first
        with store.connect() as con:
            assert con.execute("SELECT count(*) FROM publication_log").fetchone()[0] == 1
        other = dict(base, transform="demo-v2")
        second = store.prepare(other)
        try:
            store.publish(second, None, "stale writer")
        except Conflict:
            pass
        else:
            raise AssertionError("stale publication accepted")
        store.publish(second, first, "update")
        assert store.read_release()["transform"] == "demo-v2"
        store.object_path(a.digest).write_bytes(b"damage")
        assert len(store.audit()["corrupt"]) == 1
        print("release-store checks passed")


if __name__ == "__main__":
    demonstration()
