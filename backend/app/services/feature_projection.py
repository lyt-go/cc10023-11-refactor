"""特征定义投影层：一次读取定义字段，按用途投影成目录、组清单和训练输入。

下游的数据目录、特征组导出、训练准备都从这里取数，避免各自实现一遍字段映射：
- ``read_definition(row)`` 把一行特征定义读成唯一一份领域视图（状态、类型、逻辑同源）；
- ``project_definition(view, purpose)`` 按用途输出字段，键顺序固定；
- 组清单与训练准备共用同一套“组 + 已生效定义”的匹配逻辑。
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

ProjectionPurpose = Literal["catalog", "group", "training"]

# 各用途输出的字段顺序就是接口里的字段顺序，改动会直接影响导出契约。
CATALOG_FIELDS: tuple[str, ...] = (
    "feature_code",
    "feature_name",
    "feature_type",
    "data_source",
    "compute_logic",
    "version",
)
GROUP_FIELDS: tuple[str, ...] = ("name", "type", "logic")
TRAINING_FIELDS: tuple[str, ...] = ("name", "type", "logic", "version")

# 定义行 → 领域视图。状态与类型、计算逻辑都从同一行读取，后续不再二次查表。
DEFINITION_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("feature_code", "特征编号"),
    ("feature_name", "特征名称"),
    ("feature_type", "特征类型"),
    ("data_source", "数据来源"),
    ("compute_logic", "计算逻辑"),
    ("version", "特征版本"),
)


@dataclass(frozen=True)
class DefinitionView:
    """一份特征定义的只读视图；同一实例中的状态与内容来自同一行数据。"""

    id: Any
    status: Any
    feature_code: Any
    feature_name: Any
    feature_type: Any
    data_source: Any
    compute_logic: Any
    version: Any

    @property
    def effective(self) -> bool:
        return self.status == "已生效"


@dataclass(frozen=True)
class GroupView:
    """一份特征组的只读视图，包含解析后的特征名称列表。"""

    id: Any
    status: Any
    name: Any
    version: Any
    feature_names: tuple[str, ...]


def parse_feature_names(raw: Any) -> tuple[str, ...]:
    """把组里的“包含特征”文本解析为有序特征名列表。"""
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def read_definition(row: Mapping[str, Any]) -> DefinitionView:
    """把特征定义行读成唯一视图，保留字段缺失时的 None 默认值。"""
    values = {key: row.get(source) for key, source in DEFINITION_FIELD_MAP}
    return DefinitionView(
        id=row.get("id"),
        status=row.get("status"),
        **values,
    )


def read_group(row: Mapping[str, Any]) -> GroupView:
    return GroupView(
        id=row.get("id"),
        status=row.get("status"),
        name=row.get("组名称"),
        version=row.get("组版本"),
        feature_names=parse_feature_names(row.get("包含特征")),
    )


def project_definition(definition: DefinitionView, purpose: ProjectionPurpose) -> dict[str, Any]:
    """按用途投影同一份定义视图；输出键顺序由用途字段表固定。"""
    if purpose == "catalog":
        fields = CATALOG_FIELDS
        values = {
            "feature_code": definition.feature_code,
            "feature_name": definition.feature_name,
            "feature_type": definition.feature_type,
            "data_source": definition.data_source,
            "compute_logic": definition.compute_logic,
            "version": definition.version,
        }
    elif purpose in ("group", "training"):
        fields = GROUP_FIELDS if purpose == "group" else TRAINING_FIELDS
        values = {
            "name": definition.feature_name,
            "type": definition.feature_type,
            "logic": definition.compute_logic,
            "version": definition.version,
        }
    else:
        raise ValueError(f"未知的定义投影用途：{purpose}")
    return {field: values[field] for field in fields}


def project_catalog(definition_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """全量定义目录：保留原行顺序，不做状态过滤。"""
    return [
        project_definition(read_definition(row), "catalog")
        for row in definition_rows
    ]


def match_effective_definitions(
    group: GroupView,
    definition_rows: Iterable[Mapping[str, Any]],
) -> list[DefinitionView]:
    """按组的特征名匹配已生效定义；顺序沿用定义表的原顺序。"""
    wanted = set(group.feature_names)
    return [
        definition
        for definition in (read_definition(row) for row in definition_rows)
        if definition.feature_name in wanted and definition.effective
    ]


def project_group_export(
    group_id: int,
    group: GroupView,
    definition_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """特征组导出视图：只含已生效定义，字段与原接口一致。"""
    matched = match_effective_definitions(group, definition_rows)
    return _group_export_from_matched(group_id, group, matched)


def project_group_views(
    group_id: int,
    group: GroupView,
    definition_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """一次匹配同时产出组导出与训练准备清单，两者必然来自同一份定义。"""
    matched = match_effective_definitions(group, definition_rows)
    export = _group_export_from_matched(group_id, group, matched)
    manifest = [
        project_definition(definition, "training") for definition in matched
    ]
    return export, manifest


def _group_export_from_matched(
    group_id: int,
    group: GroupView,
    matched: Sequence[DefinitionView],
) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "group_name": group.name,
        "feature_count": len(matched),
        "features": [
            project_definition(definition, "group") for definition in matched
        ],
    }


def read_training_manifest(pipeline_row: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """读取流水线创建时保存的训练准备清单。

    返回 ``None`` 表示该流水线没有组投影（历史流水线），调用方应沿用原有特征集配置。
    """
    raw = pipeline_row.get("训练准备清单")
    if raw in (None, ""):
        return None
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except ValueError:
            return None
    else:
        decoded = raw
    if not isinstance(decoded, list):
        return None
    return [
        {
            "name": item.get("name"),
            "type": item.get("type"),
            "logic": item.get("logic"),
            "version": item.get("version"),
        }
        for item in decoded
        if isinstance(item, Mapping)
    ]


def serialize_training_manifest(manifest: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(list(manifest), ensure_ascii=False)
