"""内存数据仓库：给每个业务模块准备一份可筛选、可流转的示例数据。

真实项目里这里会换成数据库访问层；当前实现只依赖标准库，保证克隆下来就能起。

读取链路约定：
- ``snapshot()`` 在同一把锁内深拷贝若干模块，返回同一提交批次的数据截面；
- ``transaction()`` 在同一把锁内完成多表读取与写入，退出时整批提交；
- ``commit_batch()`` 把同一模块的多行更新作为一批原子提交，失败整批不落库。
"""
from __future__ import annotations

import copy
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from app.seed import SEED_ROWS


@dataclass(frozen=True)
class StoreSnapshot:
    """某一提交批次上的只读数据截面。

    行内容在生成截面时深拷贝，后续写入不会反向污染已在读取中的结果。
    """

    generation: int
    tables: dict[str, tuple[dict[str, Any], ...]]

    def table(self, module: str) -> tuple[dict[str, Any], ...]:
        return self.tables.get(module, ())

    def find(self, module: str, entry_id: int) -> dict[str, Any] | None:
        for row in self.table(module):
            if int(row.get("id", 0)) == entry_id:
                return row
        return None


@dataclass(frozen=True)
class BatchCommit:
    """一次原子批量写入的结果。"""

    generation: int
    updated: tuple[int, ...]
    missing: tuple[str, ...]


class StoreTransaction:
    """持锁事务：读取与写入落在同一个提交批次内。"""

    def __init__(self, store: "Store") -> None:
        self._store = store
        self.dirty = False
        self._next_ids: dict[str, int] = {}

    def rows(self, module: str) -> list[dict[str, Any]]:
        return self._store._tables.setdefault(module, [])

    @property
    def generation(self) -> int:
        """本次事务进入时的提交批次号。"""
        return self._store._generation

    def find(self, module: str, entry_id: int) -> dict[str, Any] | None:
        for row in self.rows(module):
            if int(row.get("id", 0)) == entry_id:
                return row
        return None

    def next_id(self, module: str) -> int:
        if module not in self._next_ids:
            self._next_ids[module] = max(
                (int(row.get("id", 0)) for row in self.rows(module)), default=0
            ) + 1
        else:
            self._next_ids[module] += 1
        return self._next_ids[module]

    def insert(self, module: str, row: Mapping[str, Any]) -> dict[str, Any]:
        entry = {"id": row["id"]} if "id" in row else {"id": self.next_id(module)}
        entry.update({key: value for key, value in row.items() if key != "id"})
        self.rows(module).append(entry)
        self.dirty = True
        return entry


class Store:
    def __init__(self) -> None:
        self._tables: dict[str, list[dict[str, Any]]] = {
            name: [dict(row) for row in rows] for name, rows in SEED_ROWS.items()
        }
        self._lock = threading.RLock()
        self._generation = 1

    def module_names(self) -> list[str]:
        return sorted(self._tables)

    def reset(self) -> None:
        """把仓库恢复为种子数据（测试与本地演示用）。"""
        with self._lock:
            self._tables = {
                name: [dict(row) for row in rows]
                for name, rows in SEED_ROWS.items()
            }
            self._generation = 1

    def rows(self, module: str) -> list[dict[str, Any]]:
        with self._lock:
            return self._tables.setdefault(module, [])

    def find(self, module: str, entry_id: int) -> dict[str, Any] | None:
        with self._lock:
            for row in self._tables.get(module, ()):
                if int(row.get("id", 0)) == entry_id:
                    return row
        return None

    @property
    def generation(self) -> int:
        """当前已提交批次号；每次原子写入提交后递增。"""
        with self._lock:
            return self._generation

    def snapshot(self, modules: Iterable[str]) -> StoreSnapshot:
        """在同一把锁内读取多个模块，返回同一提交批次的数据截面。"""
        with self._lock:
            tables = {
                module: tuple(
                    copy.deepcopy(row) for row in self._tables.get(module, ())
                )
                for module in modules
            }
            return StoreSnapshot(generation=self._generation, tables=tables)

    @contextmanager
    def transaction(self) -> Iterator[StoreTransaction]:
        """多表读写在同一个提交截面内完成；有写入时退出后整批可见。

        事务体抛出异常时丢弃本次写入，不留半批数据。
        """
        with self._lock:
            backup = copy.deepcopy(self._tables)
            txn = StoreTransaction(self)
            try:
                yield txn
            except BaseException:
                self._tables = backup
                raise
            if txn.dirty:
                self._generation += 1

    def commit_batch(
        self,
        module_or_changes: str | Mapping[str, Sequence[Mapping[str, Any]]],
        updates: Sequence[Mapping[str, Any]] | None = None,
        *,
        match_keys: Sequence[str] = ("id",),
    ) -> BatchCommit:
        """把一组行更新作为同一批次原子提交。

        调用方可以传 ``(module, updates)``，也可以传 ``{module: updates}``。
        所有更新在同一把锁内完成；读取方只会看到整批之前或整批之后的数据。
        存在无法匹配的行时整批不落库，避免出现半批字段。
        """
        if isinstance(module_or_changes, str):
            changes: Mapping[str, Sequence[Mapping[str, Any]]] = {
                module_or_changes: updates or (),
            }
        else:
            changes = module_or_changes

        with self._lock:
            prepared: list[tuple[dict[str, Any], dict[str, Any]]] = []
            missing: list[str] = []

            for module, batch in changes.items():
                rows = self._tables.setdefault(module, [])
                for update in batch:
                    row, matched_key = self._match_row(rows, update, match_keys)
                    if row is None:
                        marker = next(
                            (str(update[key]) for key in match_keys if key in update),
                            "<未提供匹配键>",
                        )
                        missing.append(f"{module}:{marker}")
                        continue
                    prepared.append((row, {
                        key: value for key, value in update.items()
                        if key != matched_key
                    }))

            if missing:
                return BatchCommit(self._generation, (), tuple(missing))

            for row, fields in prepared:
                row.update(fields)
            if prepared:
                self._generation += 1
            updated = tuple(int(row.get("id", 0)) for row, _ in prepared)
            return BatchCommit(self._generation, updated, ())

    @staticmethod
    def _match_row(
        rows: Sequence[dict[str, Any]],
        update: Mapping[str, Any],
        match_keys: Sequence[str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        for key in match_keys:
            if key not in update:
                continue
            expected = str(update[key])
            for row in rows:
                if str(row.get(key)) == expected:
                    return row, key
        return None, None

    def overview(self) -> dict[str, object]:
        with self._lock:
            modules: list[dict[str, object]] = []
            for name in self.module_names():
                rows = self._tables.get(name, ())
                modules.append({
                    "name": name,
                    "created": len(rows),
                    "pending": sum(1 for row in rows if row.get("pending")),
                    "abnormal": sum(1 for row in rows if row.get("abnormal")),
                })
            cards = [
                {"label": "业务模块", "value": len(modules)},
                {"label": "今日新增", "value": sum(int(item["created"]) for item in modules)},
                {"label": "待处理", "value": sum(int(item["pending"]) for item in modules)},
                {"label": "异常量", "value": sum(int(item["abnormal"]) for item in modules)},
            ]
            return {"cards": cards, "modules": modules}


store = Store()
