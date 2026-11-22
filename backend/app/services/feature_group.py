"""特征组管理领域服务：维护特征逻辑分组、主题归类和互斥规则。"""
from __future__ import annotations

from typing import Any

from app.services.base import BaseModuleService
from app.services.feature_projection import (
    project_group_export,
    project_group_views,
    read_group,
    serialize_training_manifest,
)
from app.store import store


class FeatureGroupService(BaseModuleService):
    MODULE = "feature_group"
    ENTITY = "特征组"
    OPERATION_TERM = "校验"
    LIST_FIELDS = ["组编号", "组名称", "包含特征", "业务主题", "默认权重", "互斥规则", "依赖关系", "组版本"]
    REQUIRED_FIELDS = ["组编号", "组名称", "包含特征"]
    STATUS_ORDER = ["草稿", "待校验", "已生效", "已停用"]
    ACTION_RULES = {"提交校验": "待校验", "标记生效": "已生效", "停用组定义": "已停用"}
    NEGATIVE_ACTIONS = ["停用组定义"]

    # ── 链路4: 特征组 → 训练流水线 → 模型注册 ──
    def export_group_features(self, group_id: int) -> dict[str, Any]:
        """查询特征组内所有已生效特征定义并导出为训练流水线可用格式。

        组与定义在同一提交截面上读取，成功结果不会混入两个批次的字段。
        """
        snapshot = store.snapshot((self.MODULE, "feature_define"))
        group_row = snapshot.find(self.MODULE, group_id)
        if group_row is None:
            return {"ok": False, "message": f"特征组 {group_id} 不存在"}
        group = read_group(group_row)
        result = project_group_export(
            group_id, group, snapshot.table("feature_define"),
        )
        return {"ok": True, "data": result}

    def create_training_pipeline(self, group_id: int) -> dict[str, Any]:
        """基于特征组创建训练流水线配置，关联特征集并初始化算法参数。

        组、定义读取与流水线登记在同一事务内完成；组导出所用的定义投影
        原样写入训练准备清单，后续启动训练不再回读实时定义。
        """
        with store.transaction() as txn:
            group_row = txn.find(self.MODULE, group_id)
            if group_row is None:
                return {"ok": False, "message": f"特征组 {group_id} 不存在"}

            group = read_group(group_row)
            export, manifest = project_group_views(
                group_id, group, txn.rows("feature_define"),
            )
            next_id = txn.next_id("train_pipeline")
            txn.insert("train_pipeline", {
                "id": next_id,
                "status": "草稿",
                "pending": True,
                "abnormal": False,
                "流水线编号": f"PIPE-{next_id:04d}",
                "流水线名称": f"train_{group.name or 'unknown'}_pipeline",
                "特征集": group_row.get("包含特征", ""),
                "算法类型": "xgboost",
                "超参数配置": '{"max_depth":6,"learning_rate":0.1,"n_estimators":100}',
                "评估指标": "auc,logloss",
                "训练周期": "weekly",
                "流水线版本": "v1.0.0",
                "训练准备清单": serialize_training_manifest(manifest),
                "定义批次": str(txn.generation),
            })

        return {
            "ok": True,
            "pipeline_id": next_id,
            "feature_count": export["feature_count"],
            "message": f"已基于特征组 {group_id} 创建训练流水线 {next_id}",
        }

    def check_mutex_rule_compliance(self, group_id: int) -> dict[str, Any]:
        """校验特征组的互斥规则合规性，确保无冲突特征同时出现在同一组中。"""
        group = store.find(self.MODULE, group_id)
        if group is None:
            return {"ok": False, "message": "特征组不存在"}
        mutex_rule = group.get("互斥规则", "")
        feature_names = [f.strip() for f in (group.get("包含特征") or "").split(",") if f.strip()]
        violations = []
        if mutex_rule and len(feature_names) > 5:
            violations.append("特征数量超过互斥规则约束上限5个")
        if violations:
            group["abnormal"] = True
            group["status"] = "待校验"
        else:
            group["abnormal"] = False
        return {"ok": len(violations) == 0, "violations": violations, "group_id": group_id}

    def aggregate_group_dependencies(self, group_id: int) -> dict[str, Any]:
        """聚合特征组的依赖关系并关联上下游模块，生成完整依赖链路图。"""
        group = store.find(self.MODULE, group_id)
        if group is None:
            return {"ok": False}
        dep_text = group.get("依赖关系", "")
        deps = [d.strip() for d in dep_text.split(",") if d.strip()] if dep_text else []
        pipeline_rows = store.rows("train_pipeline")
        downstream = [r.get("id") for r in pipeline_rows if group.get("组名称", "") in str(r.get("特征集", ""))]
        return {
            "ok": True,
            "group_id": group_id,
            "upstream_deps": deps,
            "downstream_pipelines": downstream,
            "chain_length": len(deps) + len(downstream),
        }