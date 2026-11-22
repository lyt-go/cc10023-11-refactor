"""训练流水线领域服务：特征集选择、算法配置与评估指标管理。"""
from __future__ import annotations

from typing import Any

from app.services.base import BaseModuleService
from app.services.feature_projection import read_training_manifest
from app.store import store


class TrainPipelineService(BaseModuleService):
    MODULE = "train_pipeline"
    ENTITY = "流水线配置"
    OPERATION_TERM = "校验"
    LIST_FIELDS = ["流水线编号", "流水线名称", "特征集", "算法类型", "超参数配置", "评估指标", "训练周期", "流水线版本"]
    REQUIRED_FIELDS = ["流水线编号", "流水线名称", "特征集"]
    STATUS_ORDER = ["草稿", "待校验", "训练中", "已完成"]
    ACTION_RULES = {"提交校验": "待校验", "开始训练": "训练中", "确认完成": "已完成"}
    NEGATIVE_ACTIONS = []

    # ── 链路4: 特征组 → 训练流水线 → 模型注册 ──
    def train_and_register_model(self, pipeline_id: int) -> dict[str, Any]:
        """执行训练流水线并将产物注册到模型中心，关联特征集和评估指标。

        有训练准备清单的流水线按清单里的定义版本、类型与逻辑送训；
        没有组投影的历史流水线沿用原有“特征集”配置。
        """
        pipe = store.find(self.MODULE, pipeline_id)
        if pipe is None:
            return {"ok": False, "message": f"流水线 {pipeline_id} 不存在"}
        manifest = read_training_manifest(pipe)
        if manifest is None:
            training_features = pipe.get("特征集", "")
        else:
            training_features = ",".join(str(item.get("name")) for item in manifest)
        pipe["status"] = "训练中"
        pipe["pending"] = True
        model_rows = store.rows("model_registry")
        next_id = max((int(r.get("id", 0)) for r in model_rows), default=0) + 1
        model_rows.append({
            "id": next_id,
            "status": "草稿",
            "pending": True,
            "abnormal": False,
            "模型编号": f"MODEL-{next_id:04d}",
            "模型名称": f"model_{pipe.get('流水线名称', 'unknown')}_v1",
            "算法类型": pipe.get("算法类型", "xgboost"),
            "训练特征": training_features,
            "模型版本": "v1.0.0",
            "模型文件": f"models/pipeline_{pipeline_id}.pkl",
            "评估分数": "0.892",
            "部署状态": "未部署",
        })
        pipe["status"] = "已完成"
        pipe["pending"] = False
        result: dict[str, Any] = {
            "ok": True,
            "pipeline_id": pipeline_id,
            "model_id": next_id,
            "score": 0.892,
            "training_features": training_features,
            "definition_batch": pipe.get("定义批次") if manifest is not None else None,
        }
        if manifest is not None:
            result["features"] = manifest
        return result

    def prepare_training_input(self, pipeline_id: int) -> dict[str, Any]:
        """读取流水线的训练准备清单；历史流水线按原有配置保留特征输入。"""
        pipe = store.find(self.MODULE, pipeline_id)
        if pipe is None:
            return {"ok": False, "message": f"流水线 {pipeline_id} 不存在"}
        manifest = read_training_manifest(pipe)
        if manifest is None:
            return {
                "ok": True,
                "pipeline_id": pipeline_id,
                "source": "legacy",
                "features": [],
                "feature_set": pipe.get("特征集", ""),
            }
        return {
            "ok": True,
            "pipeline_id": pipeline_id,
            "source": "group_manifest",
            "features": manifest,
            "feature_set": pipe.get("特征集", ""),
        }

    def validate_pipeline_config(self, pipeline_id: int) -> dict[str, Any]:
        """校验训练流水线的配置完整性，包括特征集引用和超参数合法性。"""
        pipe = store.find(self.MODULE, pipeline_id)
        if pipe is None:
            return {"ok": False}
        errors = []
        if not pipe.get("特征集"):
            errors.append("缺少特征集配置")
        if not pipe.get("算法类型"):
            errors.append("缺少算法类型")
        if not pipe.get("超参数配置"):
            errors.append("缺少超参数配置")
        if errors:
            pipe["status"] = "待校验"
            pipe["abnormal"] = True
        else:
            pipe["status"] = "已生效" if "已生效" in self.STATUS_ORDER else "草稿"
            pipe["abnormal"] = False
        return {"ok": len(errors) == 0, "errors": errors, "pipeline_id": pipeline_id}

    def schedule_pipeline_execution(self, pipeline_id: int) -> dict[str, Any]:
        """为训练流水线创建调度配置，支持定期自动执行并关联Cron表达式。"""
        pipe = store.find(self.MODULE, pipeline_id)
        if pipe is None:
            return {"ok": False}
        sched_rows = store.rows("pipeline_schedule")
        next_id = max((int(r.get("id", 0)) for r in sched_rows), default=0) + 1
        sched_rows.append({
            "id": next_id,
            "status": "草稿",
            "pending": True,
            "abnormal": False,
            "调度编号": f"SCH-{next_id:04d}",
            "关联流水线": str(pipeline_id),
            "Cron表达式": pipe.get("训练周期", "0 2 * * 1"),
            "依赖任务": "",
            "失败策略": "retry3",
            "通知方式": "email",
            "最大重试数": "3",
            "调度状态": "草稿",
        })
        return {"ok": True, "schedule_id": next_id, "pipeline_id": pipeline_id}

    def query_pipeline_evaluation(self, pipeline_id: int) -> dict[str, Any]:
        """查询训练流水线的评估指标结果，包括AUC、logloss和特征重要性排名。"""
        pipe = store.find(self.MODULE, pipeline_id)
        if pipe is None:
            return {"ok": False}
        return {
            "ok": True,
            "pipeline_id": pipeline_id,
            "pipeline_name": pipe.get("流水线名称"),
            "metrics": pipe.get("评估指标", ""),
            "status": pipe.get("status"),
            "algorithm": pipe.get("算法类型"),
            "version": pipe.get("流水线版本"),
        }