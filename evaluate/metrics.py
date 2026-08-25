"""
公共指标计算模块：基于 eval_results_v2.json 的原始字段，
派生 4 个新交叉指标 + 7 个保留指标，组成新 11 项。
"""
from statistics import mean

THRESH = 0.5            # semantic / content 阈值
SCHEMA_THRESH = 0.85    # E2E 的 schema 阈值（≥ 12/14 项通过即可，允许 ≤2 项瑕疵）

# 新 11 项的统一键
NEW_METRICS = [
    ("SCJ", "Source-Content联合"),
    ("E2E", "端到端严格通过"),
    ("SV",  "Schema结构"),
    ("SC",  "语义正确"),
    ("IC",  "信息完整"),
    ("IV",  "信息有效"),
    ("INR", "信息非冗余"),
    ("CF",  "内容一致"),
    ("SSC", "Schema-Semantic"),
    ("MSC", "Meta语义"),
    ("MSJ", "Meta-Sample联合"),
]


def geom3(a, b, c):
    a = max(a, 1e-6); b = max(b, 1e-6); c = max(c, 1e-6)
    return (a * b * c) ** (1.0 / 3.0)


def compute_new_dims(info: dict) -> dict | None:
    """从 eval_results_v2.json 中某个 model 的 info，派生新 11 项分数（0-1）"""
    if not isinstance(info, dict) or "error" in info:
        return None
    ps = info.get("per_sample", []) or []
    dims = info.get("dimensions", {}) or {}
    n = len(ps)
    if n == 0:
        return None

    # SCJ
    scj = sum(float(s.get("source_score", 0)) * float(s.get("content_fidelity_score", 0))
              for s in ps) / n

    # E2E（schema 用连续分数阈值，允许少量瑕疵，避免单项卡死全归零）
    e2e_cnt = 0
    for s in ps:
        if (s.get("pair_valid") and
            float(s.get("source_score", 0)) >= 1.0 and
            float(s.get("schema_valid_score", 0)) >= SCHEMA_THRESH and
            float(s.get("semantic_score", 0)) >= THRESH and
            float(s.get("content_fidelity_score", 0)) >= THRESH):
            e2e_cnt += 1
    e2e = e2e_cnt / n

    # SSC
    ssc = sum(float(s.get("schema_valid_score", 0)) * float(s.get("semantic_score", 0))
              for s in ps) / n

    # 保留 7 项（直接读 dimensions）
    def _d(k):
        v = dims.get(k, 0)
        if isinstance(v, dict):
            v = v.get("mean", v.get("score", 0))
        return float(v) if v is not None else 0.0

    sv  = _d("schema_validity")
    sc  = _d("semantic")
    ic  = _d("information_completeness")
    iv  = _d("information_validity")
    inr = _d("information_non_redundancy")
    cf  = _d("content_fidelity")
    msc = _d("meta_semantic_correctness")
    msv = _d("meta_schema_validity")
    mcf = _d("meta_content_fidelity")

    # MSJ
    msj = geom3(msv, msc, mcf) * cf

    return {
        "SCJ": scj, "E2E": e2e, "SV": sv, "SC": sc,
        "IC": ic, "IV": iv, "INR": inr, "CF": cf,
        "SSC": ssc, "MSC": msc, "MSJ": msj,
    }


def overall(new_dims: dict) -> float:
    if not new_dims: return 0.0
    return mean(new_dims[k] for k, _ in NEW_METRICS)


# 方案 A：任务核心性加权（与 main_table.py 完全一致）
CATEGORIES = {
    "联合通过":  (["SCJ", "E2E"], 0.30),
    "内容质量":  (["IC", "IV", "INR", "CF"], 0.25),
    "语义正确":  (["SC"], 0.20),
    "结构合法":  (["SV", "SSC"], 0.15),
    "Meta质量":  (["MSC", "MSJ"], 0.10),
}


def weighted_score(dims: dict) -> float:
    """任务核心性加权总分（方案 A）：5 大类按 30/25/20/15/10 加权，类内子项均分"""
    if not dims:
        return 0.0
    s = 0.0
    for _cat, (keys, w) in CATEGORIES.items():
        vals = [dims[k] for k in keys if k in dims]
        if vals:
            s += w * mean(vals)
    return s


def simple_avg(dims: dict) -> float:
    """11 项简单算术平均"""
    if not dims:
        return 0.0
    return mean(dims[k] for k, _ in NEW_METRICS)


def build_stage_result(raw: dict) -> dict:
    """将底层评估原始结果压缩为主表新 11 项输出（不含 v2 dimensions 字段）。"""
    if not isinstance(raw, dict):
        return {"error": "评估结果无效"}
    if "error" in raw:
        return {"error": raw["error"]}
    n = len(raw.get("per_sample") or [])
    dims = compute_new_dims(raw)
    if dims is None:
        return {
            "error": "无法计算主表 11 项（缺少 per_sample 或样本数为 0）",
            "n_samples": n,
        }
    return {
        "n_samples": n,
        "metrics": dims,
        "avg": overall(dims),
        "weighted": weighted_score(dims),
    }
