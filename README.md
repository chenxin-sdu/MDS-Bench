# MDS-Bench

**Medical Data Standardization Benchmark**

> Paper: *[Solve the Missing First Step: Can VLMs Standardize Raw Heterogeneous Medical Data?](https://arxiv.org/abs/2607.04694)*  
> arXiv: [2607.04694](https://arxiv.org/pdf/2607.04694)  
> **Accepted to EMNLP 2026.**  
> Raw data (Hugging Face): [sdu-chenxin/MDS-Bench-data](https://huggingface.co/datasets/sdu-chenxin/MDS-Bench-data)

现有医学 VLM 评测大多默认输入已是整理好的图像 / 文本 / QA。真实临床与科研场景中，数据往往是**原始、异构、目录分散**的。MDS-Bench 评测的正是被忽略的上游步骤：**原始医学数据标准化（raw medical data standardization）**——模型拿到原始数据集文件夹后，能否识别信息源、将原始影像转为 VLM 可用图像、抽取文本与标注，并组织为统一的 **image + JSON**。

人工标注 **1,939** 个标准化任务，覆盖 **100** 个数据集、分类 / 分割 / 检测、多种模态与原始格式。实验表明，即便最强模型 Gemini 3 Flash，端到端成功率（E2E）也仅 **48.6%**。

---

## 主实验 Prompt

主实验使用的 staged reasoning prompt 为：

**[`datasets/standard.md`](datasets/standard.md)**

该文件即论文中的 Data Organization Agent 提示词：引导 agentic VLM 分阶段完成 source grounding → 视觉标准化 → 结构化标注 → 一致性检查，并规定统一的 image + JSON 输出 schema。

复现主表结果时，请将该 prompt 与对应数据集的原始 `data/` 目录、`VLM/random.txt` 样本清单一起提供给模型；**不要**读取 `ground_truth*.json`。

---

每个数据集在本仓库中的公开结构：

```text
datasets/<group>/<dataset_id>/VLM/
├── ground_truth_*.json   # 人工核验的结构化 GT
└── random.txt            # 目标样本清单（index + source path）
```

完整本地实验目录还需放置原始数据（从 Hugging Face 下载，见下文）：

```text
<dataset_id>/
├── data/                 # 原始异构数据（Hugging Face 下载）
├── VLM/
│   ├── random.txt
│   └── ground_truth_*.json
└── agent1_data_organization/   # 模型输出
    ├── images/
    ├── standardized_annotations/
    └── data_meta.json
```

---

## 数据规模

| 分组               |      数据集数 |
| ------------------ | ------------: |
| `origin_dataset` |             8 |
| `second_dataset` |            34 |
| `third_dataset`  |            22 |
| `forth_dataset`  |            22 |
| `fifth_dataset`  |            14 |
| **合计**     | **100** |

- 样本数：**1,939**
- 任务构成：约 51 分割 / 43 分类 / 6 检测
- 模态：CT、MRI、显微病理、X-ray、超声、眼科、内镜等
- 原始格式：DICOM、NIfTI、TIFF、数组、视频及专用格式等

`random.txt` 格式：

```text
<index> <path_to_source_file>
```

评测按索引对齐 GT；路径可能需映射到你本地的数据根目录。

---

## 原始数据下载

100 个数据集的原始 `data/` 目录（约 2 TB）托管在 Hugging Face：
[sdu-chenxin/MDS-Bench-data](https://huggingface.co/datasets/sdu-chenxin/MDS-Bench-data)。
Hub 上的 `<group>/<dataset_id>/data/` 与本仓库 `datasets/<group>/<dataset_id>/` 一一对应，建议按需下载：

```bash
# 下载单个数据集到 datasets/<group>/<dataset_id>/data/
hf download sdu-chenxin/MDS-Bench-data --repo-type dataset \
  --include "fifth_dataset/LCTSC/data/*" --local-dir datasets
```

- `second_dataset/fomo-mri`：受 Hub 单仓库文件数上限限制，部分文件打包在 `data/fomo-60k-shards/shard_*.tar`，
  需在 `data/` 下执行 `tar -xf shard_XXX.tar -C .` 还原，`INDEX.tsv` 记录每个文件所在分片。
- `third_dataset/brats2020-training-data`、`fifth_dataset/rsna-pneumonia-detection-challenge`：
  因单目录文件数上限，部分文件被移入子目录，`random.txt` 中的路径需相应调整。

---

## 评测协议（11 指标）

| 组别      | 指标            |
| --------- | --------------- |
| Structure | SV, SSC         |
| Semantic  | SC              |
| Content   | IC, IV, INR, CF |
| Metadata  | MSC, MSJ        |
| Joint     | SCJ, E2E        |

- **Source-Gate**：`source_score = 0` 的样本，不计入 SC / IC / IV / INR / CF 的有效平均。
- **E2E**：图像–JSON 对有效 ∧ source 完全正确 ∧ schema ≥ 0.85 ∧ semantic ≥ 0.5 ∧ content ≥ 0.5。
- 分数先在每个数据集内计算，再对 **100 个数据集等权平均**。

相关代码：

```bash
python evaluate/evaluate_source_gate.py   # 严格 11 指标（含 Source-Gate）
python evaluate/metrics.py                # SCJ / E2E 等交叉指标派生
python evaluate/main_table_11metrics.py   # 跨数据集主表汇总
```

> 汇总脚本内默认路径可能仍指向内部机器目录，公开复现前请改为本仓库 `datasets/*_dataset`。

---

## 消融 / 扩展实验（`prompt_experiment/`）

主结果对应 `datasets/standard.md`。以下为额外设定，**不是**论文主表默认设定：

| 设定 | 说明                                                                      |
| ---- | ------------------------------------------------------------------------- |
| S0   | `Data_Organization_Agent.md`（与 `standard.md` 同源的整理提示词副本） |
| S1   | 全局自检修订                                                              |
| S2   | 高风险字段白名单修补                                                      |
| S3   | Best-of-n 候选选择脚本                                                    |
| S4   | 窄域字段投票脚本                                                          |

S1–S4 均禁止读取 GT 与评测结果文件。

---

## 快速开始

1. 从 [Hugging Face](https://huggingface.co/datasets/sdu-chenxin/MDS-Bench-data) 下载某个数据集的原始文件到 `<dataset_id>/data/`，并修正 `VLM/random.txt` 中的路径。
2. 将 **`datasets/standard.md`** 作为主 prompt，在可访问文件系统的 agentic VLM 环境中运行。
3. 产出 `agent1_data_organization/{images,standardized_annotations,data_meta.json}`。
4. 用 `evaluate/` 脚本对照 `VLM/ground_truth_*.json` 打分。

---

## 许可与数据版权

- 本仓库公开：**主实验 prompt、GT JSON、样本清单、评测与消融代码**；原始数据见 Hugging Face 仓库。
- 各数据集原始影像的版权与许可归原发布方；使用前请遵守其条款并正确引用。

---
