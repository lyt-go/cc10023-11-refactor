"""特征定义读取链路测试：目录、组导出、训练准备清单的一致性。"""
from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services import service_for
from app.store import store


def assert_ok(result, msg=""):
    ok = result.get("ok") if isinstance(result, dict) else getattr(result, "ok", False)
    assert ok, f"{msg}: {result}"
    return result


def _definition(name: str, feature_type: str, logic: str, version: str) -> int:
    service = service_for("feature_define")
    entry, missing = service.create_entry({
        "特征编号": f"FEAT-{name.upper()}",
        "特征名称": name,
        "特征类型": feature_type,
        "计算逻辑": logic,
        "数据来源": f"source_{name}",
        "缺失填充": "mean",
        "取值范围": "all",
        "特征版本": version,
    })
    assert not missing, f"创建特征定义失败: {missing}"
    return int(entry["id"])


def _group(feature_names: str) -> int:
    service = service_for("feature_group")
    entry, missing = service.create_entry({
        "组编号": "GROUP-AGE-INCOME",
        "组名称": "age_income_group",
        "包含特征": feature_names,
        "业务主题": "risk",
        "默认权重": "1.0",
        "互斥规则": "",
        "依赖关系": "",
        "组版本": "v1",
    })
    assert not missing, f"创建特征组失败: {missing}"
    return int(entry["id"])


def _reset_domain() -> None:
    store.reset()
    for module in ("feature_define", "feature_group", "train_pipeline", "model_registry"):
        store.rows(module).clear()


def _seed_pipeline_module() -> None:
    """只恢复训练流水线种子数据，用于验证历史流水线。"""
    store.reset()
    for module in ("feature_define", "feature_group", "model_registry"):
        store.rows(module).clear()


def _setup_group_with_effective_age_and_draft_income():
    """组内 age 已生效、income 仍是草稿。"""
    _reset_domain()
    age_id = _definition("age", "numeric", "age", "v1")
    income_id = _definition("income", "numeric", "income", "v1")
    store.commit_batch("feature_define", [
        {"id": age_id, "status": "已生效", "pending": False},
    ])
    group_id = _group("age,income")
    return group_id, age_id, income_id


def _updates(age_id: int, income_id: int):
    """同一批定义更新：age 改为分箱，income 变为已生效。"""
    return [
        {
            "id": age_id,
            "特征类型": "category",
            "计算逻辑": "pd.cut(age, [0, 18, 35, 60, 100])",
            "特征版本": "v2",
        },
        {
            "id": income_id,
            "status": "已生效",
            "pending": False,
            "计算逻辑": "income",
            "特征版本": "v2",
        },
    ]


def test_catalog_and_group_export_share_definition_view():
    """目录仍列出全部两行，组只导出已生效的 age，字段与顺序保持不变。"""
    group_id, _, _ = _setup_group_with_effective_age_and_draft_income()

    catalog = service_for("feature_define").export_feature_catalog()
    assert [row["feature_name"] for row in catalog] == ["age", "income"], catalog
    assert list(catalog[0].keys()) == [
        "feature_code", "feature_name", "feature_type",
        "data_source", "compute_logic", "version",
    ], "目录字段与顺序必须保持原契约"

    export = service_for("feature_group").export_group_features(group_id)
    assert_ok(export, "export_group_features")
    data = export["data"]
    assert list(data.keys()) == ["group_id", "group_name", "feature_count", "features"], data
    assert data["feature_count"] == 1, data
    assert data["features"] == [
        {"name": "age", "type": "numeric", "logic": "age"},
    ], data["features"]


def test_group_export_is_single_batch_snapshot():
    """组导出与目录读取不会把两个提交批次的字段拼进同一份结果。"""
    group_id, age_id, income_id = _setup_group_with_effective_age_and_draft_income()
    pre = {
        "catalog": ("numeric", "v1"),
        "group": (("age", "numeric", "age"),),
    }
    post = {
        "catalog": ("category", "v2"),
        "group": (
            ("age", "category", "pd.cut(age, [0, 18, 35, 60, 100])"),
            ("income", "numeric", "income"),
        ),
    }

    results: list[tuple[tuple, tuple]] = []
    errors: list[BaseException] = []
    stop = threading.Event()
    start = threading.Barrier(3)

    def reader() -> None:
        try:
            start.wait(timeout=5)
            while not stop.is_set():
                export = service_for("feature_group").export_group_features(group_id)
                assert_ok(export, "并发 export_group_features")
                features = tuple(
                    (item["name"], item["type"], item["logic"])
                    for item in export["data"]["features"]
                )
                catalog = service_for("feature_define").export_feature_catalog()
                by_name = {row["feature_name"]: row for row in catalog}
                # age 的类型与 income 的版本必须来自同一批定义
                catalog_state = (
                    by_name["age"]["feature_type"], by_name["income"]["version"],
                )
                results.append((catalog_state, features))
        except BaseException as exc:  # pragma: no cover - 并发失败时向上报
            errors.append(exc)
            start.abort()

    threads = [threading.Thread(target=reader) for _ in range(2)]
    for thread in threads:
        thread.start()
    try:
        start.wait(timeout=5)
        time.sleep(0.05)
        commit = store.commit_batch("feature_define", _updates(age_id, income_id))
        assert not commit.missing, commit
        time.sleep(0.05)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)

    assert not errors, errors
    assert results, "并发读取应至少得到一次结果"
    allowed_groups = (pre["group"], post["group"])
    allowed_catalogs = (pre["catalog"], post["catalog"])
    for catalog_state, features in results:
        assert features in allowed_groups, f"组导出混入了两个批次: {features}"
        assert catalog_state in allowed_catalogs, f"目录混入了两个批次: {catalog_state}"


def test_old_pipeline_keeps_snapshot_after_definition_batch():
    """提交新的定义批次后，旧流水线仍按创建时的清单训练。"""
    group_id, age_id, income_id = _setup_group_with_effective_age_and_draft_income()
    group_service = service_for("feature_group")
    pipeline_service = service_for("train_pipeline")

    old = assert_ok(group_service.create_training_pipeline(group_id), "建旧流水线")
    old_id = old["pipeline_id"]
    assert old["feature_count"] == 1

    commit = store.commit_batch("feature_define", _updates(age_id, income_id))
    assert not commit.missing, commit

    new = assert_ok(group_service.create_training_pipeline(group_id), "建新流水线")
    new_id = new["pipeline_id"]
    assert new["feature_count"] == 2

    # 新流水线导出与目录一致：分箱 age + 已生效 income
    export = group_service.export_group_features(group_id)["data"]
    assert [(f["name"], f["type"]) for f in export["features"]] == [
        ("age", "category"), ("income", "numeric"),
    ], export["features"]

    old_training = assert_ok(
        pipeline_service.train_and_register_model(old_id), "启动旧流水线")
    assert old_training["training_features"] == "age", old_training
    assert old_training["features"] == [
        {"name": "age", "type": "numeric", "logic": "age", "version": "v1"},
    ], old_training["features"]

    new_training = assert_ok(
        pipeline_service.train_and_register_model(new_id), "启动新流水线")
    assert new_training["training_features"] == "age,income", new_training
    assert new_training["features"] == [
        {
            "name": "age", "type": "category",
            "logic": "pd.cut(age, [0, 18, 35, 60, 100])", "version": "v2",
        },
        {"name": "income", "type": "numeric", "logic": "income", "version": "v2"},
    ], new_training["features"]

    # 训练产物里记录的字段也来自各自清单
    models = store.rows("model_registry")
    old_model = next(m for m in models if m["模型文件"] == f"models/pipeline_{old_id}.pkl")
    new_model = next(m for m in models if m["模型文件"] == f"models/pipeline_{new_id}.pkl")
    assert old_model["训练特征"] == "age", old_model
    assert new_model["训练特征"] == "age,income", new_model


def test_legacy_pipeline_without_group_projection():
    """没有组投影的历史流水线仍按原有特征集准备训练输入。"""
    _seed_pipeline_module()
    pipeline_service = service_for("train_pipeline")
    legacy = store.find("train_pipeline", 1)
    original_feature_set = legacy["特征集"]
    assert "训练准备清单" not in legacy

    prepared = assert_ok(
        pipeline_service.prepare_training_input(1), "历史流水线训练准备")
    assert prepared["source"] == "legacy", prepared
    assert prepared["feature_set"] == original_feature_set, prepared
    assert prepared["features"] == [], prepared

    trained = assert_ok(
        pipeline_service.train_and_register_model(1), "启动历史流水线")
    assert trained["training_features"] == original_feature_set, trained
    assert "features" not in trained, trained


def test_batch_commit_is_atomic():
    """一批定义更新要么整批生效，要么整批保持原样。"""
    group_id, age_id, income_id = _setup_group_with_effective_age_and_draft_income()
    before = {
        row["id"]: (row["特征类型"], row["status"], row["特征版本"])
        for row in store.rows("feature_define")
    }

    failed = store.commit_batch("feature_define", [
        {"id": age_id, "特征类型": "category", "特征版本": "v2"},
        {"id": 999999, "特征类型": "category", "特征版本": "v2"},
    ])
    assert failed.missing, "缺少匹配行时整批应失败"
    after = {
        row["id"]: (row["特征类型"], row["status"], row["特征版本"])
        for row in store.rows("feature_define")
    }
    assert after == before, "失败的批次不得留下半批字段"

    export = service_for("feature_group").export_group_features(group_id)["data"]
    assert export["features"] == [{"name": "age", "type": "numeric", "logic": "age"}], export


if __name__ == "__main__":
    tests = [
        test_catalog_and_group_export_share_definition_view,
        test_group_export_is_single_batch_snapshot,
        test_old_pipeline_keeps_snapshot_after_definition_batch,
        test_legacy_pipeline_without_group_projection,
        test_batch_commit_is_atomic,
    ]
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {test.__name__}: {exc}")
            failed += 1
    print(f"\n总测试: {len(tests)}, 通过: {passed}, 失败: {failed}")
    if failed:
        sys.exit(1)
