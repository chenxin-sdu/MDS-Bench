#!/usr/bin/env python3
"""
主表新指标完整版 + 加权平均（方案 A：任务核心性加权）
  1. 简单算术平均 (Avg)         —— 每项 1/11
  2. 任务核心性加权 (Weighted)  —— 端到端 > 字段精度 > 语义 > 结构 > Meta
       联合通过 = (SCJ + E2E)/2          × 0.30
       内容质量 = (IC + IV + INR + CF)/4 × 0.25
       语义正确 = SC                     × 0.20
       结构合法 = (SV + SSC)/2           × 0.15
       Meta 质量 = (MSC + MSJ)/2         × 0.10
"""
import json, sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_new_dims, NEW_METRICS

CONFIGS = [
    Path("/data3/chenxin/Agent_test/origin_dataset"),
    Path("/data3/chenxin/Agent_test/second_dataset"),
    Path("/data3/chenxin/Agent_test/third_dataset"),
    Path("/data4/chenxin/Med-Agent/forth_dataset"),
    Path("/data4/chenxin/Med-Agent/fifth_dataset"),
]
MODELS = ["Codex5.2", "Composer1.5", "GPT5.2", "Gemini3",
          "Grok4.20", "Haiku4.5", "KimiK2.5", "Opus4.6", "Sonnet4.6"]

CATEGORIES = {
    "联合通过":  (["SCJ", "E2E"], 0.30),
    "内容质量":  (["IC", "IV", "INR", "CF"], 0.25),
    "语义正确":  (["SC"], 0.20),
    "结构合法":  (["SV", "SSC"], 0.15),
    "Meta质量":  (["MSC", "MSJ"], 0.10),
}


def weighted_score(dims: dict) -> float:
    """类别等权：5 类各 20%，类内子项均分"""
    s = 0.0
    for cat, (keys, w) in CATEGORIES.items():
        vals = [dims[k] for k in keys if k in dims]
        if vals:
            s += w * mean(vals)
    return s


def collect():
    by_model = {m: {k: [] for k, _ in NEW_METRICS} for m in MODELS}
    for base in CONFIGS:
        for jf in sorted(base.glob("*/VLM/eval_results_v2.json")):
            data = json.loads(jf.read_text())
            for m in MODELS:
                dims = compute_new_dims(data.get("models", {}).get(m))
                if dims is None: continue
                for k, _ in NEW_METRICS:
                    by_model[m][k].append(dims[k])
    return by_model


def main():
    by_model = collect()

    # 计算每模型 11 项均值
    model_avg = {}
    for m in MODELS:
        d = {k: mean(by_model[m][k]) for k, _ in NEW_METRICS}
        model_avg[m] = d

    # 1) 11 项 + 简单平均 + 加权平均
    print("=" * 175)
    print("方案 A · 新 11 项主表（含两种汇总：简单平均 / 类别加权）")
    print("=" * 175)
    col_keys = [k for k, _ in NEW_METRICS]
    head = f"{'Rk':<3}{'Model':<14}|" + "|".join(f"{k:^8}" for k in col_keys) + "| Avg(均) | Wgt(加权)"
    print(head); print("-" * len(head))

    rows = []
    for m in MODELS:
        d = model_avg[m]
        simple = mean(d[k] for k in col_keys)
        weighted = weighted_score(d)
        rows.append((m, d, simple, weighted))

    # 按加权排
    rows.sort(key=lambda r: -r[3])
    for rk, (m, d, sim, wgt) in enumerate(rows, 1):
        cells = "|".join(f"{d[k]*100:^7.1f}%" for k in col_keys)
        print(f"{rk:<3}{m:<14}|" + cells + f"| {sim*100:^6.1f}% | {wgt*100:^7.1f}%")

    # 2) 5 大类别得分（便于看加权来源）
    print(f"\n{'='*100}")
    print("5 大类别得分（用于类别等权汇总）")
    print(f"{'='*100}")
    cats = list(CATEGORIES.keys())
    head = f"{'Rk':<3}{'Model':<14}|" + "|".join(f"{c:^12}" for c in cats) + "| 加权总分"
    print(head); print("-" * len(head))
    cat_rows = []
    cat_weights = [w for _, (_, w) in CATEGORIES.items()]
    for m in MODELS:
        d = model_avg[m]
        cvs = []
        for cat, (ks, _) in CATEGORIES.items():
            cvs.append(mean(d[k] for k in ks))
        wgt = sum(cvs[i] * cat_weights[i] for i in range(len(cats)))
        cat_rows.append((m, cvs, wgt))
    cat_rows.sort(key=lambda r: -r[2])
    for rk, (m, cvs, wgt) in enumerate(cat_rows, 1):
        cells = "|".join(f"{v*100:^11.1f}%" for v in cvs)
        print(f"{rk:<3}{m:<14}|" + cells + f"| {wgt*100:.1f}%")

    # 显示权重
    print("\n类别权重: " + ", ".join(f"{c}={w*100:.0f}%" for c, (_, w) in CATEGORIES.items()))


if __name__ == "__main__":
    main()
