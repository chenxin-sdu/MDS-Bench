#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 VLM 数据整理智能体评估脚本（方案 A 严格 11 项）

由原 7 个 composite 指标拆解而来：Information Quality 拆为 3 个独立子项（其中
"usefulness" 因与 schema_validity.task_branch_required 和 information_validity
存在显著重叠，按 ISO 25012 / DAMA-DMBOK 的最小冗余原则不予暴露），Dataset
Metadata Quality 拆为 3 个独立子项。

样本级（Instance-level）8 项：
1. source_selection_accuracy     信息源选择准确性 ← gate
2. coverage                      产出完整性
3. schema_validity               JSON 结构合法性
4. semantic                      高层语义正确性
5. information_completeness      信息完整性     ← 原 IQ 拆 1/3
6. information_validity          信息有效性     ← 原 IQ 拆 2/3
7. information_non_redundancy    信息非冗余性   ← 原 IQ 拆 3/3
8. content_fidelity              内容一致性

数据集级（Dataset-level）3 项：
9.  meta_schema_validity         元信息结构合法性 ← 原 Meta 拆 1/3
10. meta_semantic_correctness    元信息语义正确性 ← 原 Meta 拆 2/3
11. meta_content_fidelity        元信息内容一致性 ← 原 Meta 拆 3/3

兼容性：
- 原 7 个 composite 指标值与已移除的 usefulness 子项保留在 breakdown 字段下
  （"legacy_information_quality" / "legacy_information_usefulness" /
  "legacy_dataset_metadata_quality"）以便审稿复现/回溯。
- 重组关系：
    information_quality ≡ 0.25·completeness + 0.25·validity + 0.30·usefulness + 0.20·non_redundancy
    dataset_metadata_quality ≡ (geometric_mean(3 Meta sub-scores) ^ 1.08) × 0.92

Source-gate 设计：source_score=0 时，单样本的 semantic / 4 IQ 子项 / content_fidelity
都不计入分子；source_scored_n 为 0 时整组归 0。

严格版相较旧版的收紧点：
- GT 中显式给出 null 时，默认视为"应为空"，不再直接跳过评分；
- coverage 不仅检查文件存在，还检查图像可读、JSON 可解析、record.image_path 与真实文件一致；
- schema_validity 改为尽量使用 GT format 约束，并按样本内部多项检查平均计分；
- semantic 加入 task payload 约束；
- information quality 子项过滤默认值/占位值；
- meta 子项不再用样本级 majority 给 data_meta "保底"。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
import math

try:
    from PIL import Image, UnidentifiedImageError
except Exception:  # pragma: no cover
    Image = None
    UnidentifiedImageError = Exception


IMAGE_EXTS = (".png", ".jpg", ".jpeg")
ALLOWED_TASKS = ("classification", "segmentation", "detection")
MISSING_TOKENS = {"", "na", "n/a", "null", "none", "unknown", "unk", "nan", "tbd", "not available", "not_applicable"}
NULL_EQ_TOKENS = {"__null__", "<null>", "[null]", "__none__", "<none>"}
PLACEHOLDER_LIKE_RE = re.compile(r"^<[^<>]+>$")
SOURCE_TRACE_KEYS = (
    "source_file", "source_filename", "source_file_name", "source_relative", "original_filename", "original_file", "original_path",
    "source_path", "source_relative_path", "source_rel_path", "source_relpath", "source_h5",
    "original_h5", "bbox_file", "label_file", "mask_file", "seg_file", "subject_id",
    "patient_id", "experiment_id", "source_subset", "source_stem", "source_dcm_relative",
    "relative_dicom_path", "slice_stem", "original_stem", "dicom_relative_to_dataset_data",
    "source_image_path", "source_image_relative", "source_relative_to_dataset_data", "image_id",
    "raw_image_path", "raw_mask_path",
    "source_mat_relative", "source_mat_path", "source_parquet", "source_image_path_name",
    "source_npz", "npz_filename", "source_tiff_relative",
    "source_label_name", "anatomy_name", "name_in_mat", "mat_basename", "original_file",
    "sampling_pattern", "sampling_scheme", "acceleration_dir", "radial_config_dir",
    "acceleration_factor", "parameter_setting", "radial_spokes", "radial_undersampling_dir",
    # 同义 key 扩展（来自 VLM 输出调研；不破坏原有严格性，只接受明显指向源文件/源目录的命名）
    "source_image_file", "original_image_path", "original_label_path", "original_file_path",
    "source_sample_dir", "source_mask_file", "source_label_file", "segmentation_file",
    "source_image", "source_label", "source_video", "source_video_filename", "source_flair_nifti",
    "source_h5_file", "source_dicom", "source_image_member", "source_label_member",
    "raw_filename", "raw_relative_path", "raw_image_member_path", "dicom_source_path",
    "source_dataset_name", "source_dir", "source_angle_path", "source_dcm_path_csv",
    "source_jpg_path_csv", "filename",
    # BIDS/DTI 常见来源标识
    "bids_subject", "source_nifti_relative", "source_nifti_path",
    "sequence", "sequence_id", "frame_index", "subpath",
    # === v2 追加：兼容 forth_dataset 模型常用的源/路径/文件名命名 ===
    # 体积/分割/标签类完整路径
    "source_volume_path", "source_volume", "source_volume_absolute_path",
    "source_volume_relative", "source_volume_relpath", "source_volume_file",
    "source_volume_mhd", "source_volume_mhd_relative", "source_volume_relative_to_dataset_root",
    "source_label_path", "source_label_path_dataset_relative", "source_label_path_absolute",
    "source_label_absolute", "source_label_absolute_path",
    "source_label_filename", "source_label_subdir", "source_label_nifti",
    "source_label_nifti_relative", "source_label_nifti_path", "source_label_nifti_absolute_path",
    "source_label_nii_gz", "source_label_volume_path", "source_segmentation_path",
    "source_segmentation_nii", "segmentation_volume_relative_path",
    "segmentations_dir_relative", "segmentation_file",
    "source_mask_nii", "source_mask_nii_absolute",
    # DICOM/CT 类
    "source_dicom_path", "source_dicom_abspath", "source_dicom_stem",
    "source_dicom_name", "source_sr_dicom_path", "sr_source_path_abs",
    "dicom_path", "original_dicom_path", "ct_path_relative", "ct_volume_path",
    "source_ct_relative", "source_ct_path", "source_ct_nifti_relative",
    "source_ct_mhd_relative", "original_ct_relative_path", "original_ct_volume_path",
    # NIfTI/影像类
    "source_image_nii", "source_image_nii_absolute", "source_image_nii_gz",
    "source_image_nifti", "source_image_nifti_path",
    "source_image_absolute", "source_image_absolute_path",
    "source_image_basename", "source_image_dirname", "source_image_format",
    "source_image_stem", "source_image_volume_path",
    "source_image_relative_to_dataset_root",
    "source_image_path_absolute", "source_image_path_from_random",
    "source_image_absolute_path_from_random_txt",
    "source_data_nifti", "source_data_nifti_absolute_path",
    "source_nifti_image", "source_nifti_segmentation",
    # NRRD/MHD/TIFF 等
    "source_nrrd", "source_nrrd_absolute",
    "source_mhd_path", "source_tiff", "source_tiff_name", "source_tif_absolute",
    # 原始/参考/镜像类
    "original_label_filename", "original_label_nifti", "original_label_absolute_path",
    "original_label_volume_path", "original_volume_path", "original_data_nifti",
    "raw_input_path", "source_path_used", "source_reference",
    # MAT 文件相关
    "source_mat_file", "source_mat_absolute_path", "source_mat_patient_key",
    "source_mat_image_key", "mat_struct_root_key",
    # 文件夹/case 标识（部分数据集 GT 的 source_stem 用 case_folder 组装）
    "han_seg_case_folder", "totalsegmentator_sample_folder", "bids_cohort_folder",
    "chaos_split_folder", "chaos_case_folder", "chaos_split_path",
    "patient_folder_file", "case_id", "sample_dir",
    "subject_folder", "scan_folder",
    # 其他
    "source_split", "split_membership", "cohort_prefix",
    "source_dataset", "source_publisher", "source_kind", "source_type", "source_format",
    "source_volume_kind", "source_file_type", "label_csv", "label_csv_relative",
    "label_subfolder", "label_source_filename", "label_source_subdir", "label_source_subdirectory",
    "filename_stem", "label_relative_to_dataset_root", "source_relative_to_dataset_root",
    "volume_path_relative", "volume_source_relpath",
    "cropping_original_id", "sequence_from_filename",
    # 兜底（少量场景下 mask/text 路径也是源溯源证据）
    "mask_path", "mask_volume_path", "mask_volume", "binary_mask_png",
    "mask_slice_png_path", "display_mask_slice_relative", "text_json_path",
    "standardized_label_nifti_relative",
    # 通用 case/series/study 数字标识（用作"源溯源证据"的兜底，不参与 stem 派生）
    "series_id", "series_uid", "seriesuid", "study_id",
    "study_instance_uid", "series_instance_uid", "sop_instance_uid",
    "sr_sop_instance_uid", "original_subject_id",
    "split", "split_path",
    # === v2 第二轮补丁：second/third_dataset 模型常用源/路径键名 ===
    # 路径类
    "source_absolute_path", "source_basename", "source_image_relpath",
    "source_image_relative_to_dataset", "source_image_name",
    "source_image_dtype", "source_image_format",
    "source_scan_path", "source_series", "source_format",
    "source_npy_relative", "source_npz_relative", "source_channel_relative",
    "source_mask_path", "source_mask_tif", "source_mask",
    "source_label_from_dir", "source_label_text",
    "source_annotation_csv", "source_hsi_hdr", "source_hsi_dat",
    "source_czi", "source_dcm", "source_jpg",
    "source_class_name",
    "label_path", "label_source", "zmap_path", "roi_mask_path",
    "segmentation_mask_relative", "case_directory_relative",
    "h5_noisy_dataset_path", "h5_clean_dataset_path",
    "original_label_name", "original_npy_relative",
    "original_csv_path", "original_nifti_path", "original_dcm",
    "dicom_slice", "mat_variable_key",
    # 案例/样本 id 类（在脚本派生不出 stem 时作为 trace_evidence）
    "sample_id", "file_id", "file",
    "patient_folder", "roi_folder", "split_folder", "stain_panel_folder",
    "dataset_split", "dataset_split_folder", "case",
    "protocol_name",
    # === v2 第三轮补丁（来自 101 集 0% 单元审计）===
    # 路径/相对路径
    "original_relative_path", "original_image_relative_path", "original_source_relpath",
    "original_absolute_path", "original_image_filename", "original_image_path",
    "original_video_filename", "original_video_path",
    "source_file_relative", "source_file_relpath", "source_file_relative_to_dataset_data",
    "source_csv_relative", "source_npy_absolute", "source_npy_path",
    "source_image_relative_path", "source_image_relpath_from_data",
    "source_image_relative_to_dataset",
    "source_h5_relative_to_data_dir",
    "source_dicom_absolute_path",
    # h5 内部数据集路径（部分 GT 把它当作 stem）
    "noisy_dataset_internal_path", "clean_dataset_internal_path",
    "h5_group_image_key",
    # 派生 stem 类
    "drive_image_stem", "drive_split", "chaos_split", "sequence_folder",
    "image_reference_id", "vertebral_region_in_filename_hint",
    # cohort/note/label folder（用于 trace_evidence）
    "cohort_note", "class_folder_label", "dataset_folder_label",
    "cell_type_label", "sample_folder",
    # DICOM UID（部分数据集 GT 直接以 UID 为唯一标识）
    "dicom_study_uid", "dicom_series_uid", "dicom_sop_uid",
    "metadata_csv_series_uid",
    # 杂项（部分原始字段也算 source trace）
    "format", "source_extension", "roi_name", "segmentation_type",
    # === v2 patch5：second_dataset/KimiK2.5_new 新增字段 ===
    # 直接的源文件/源样本名称
    "mat_file", "original_sample_name", "image_type", "label_text",
    # 切片/索引类（标识原始体积中的切片）
    "slice_number", "extracted_slice", "slice_info",
    # 数据集子集/划分
    "dataset_subset", "dataset_split",
    # 类别/分类标签（部分数据集的源就是 class folder）
    "class_name", "cell_class", "cell_line", "cell_number",
    "transfection_marker", "sample_type",
    # 解剖/参与者/ROI 标识
    "participant_id", "participant_age", "participant_sex",
    "roi_location", "roi_number",
    # DICOM 元数据（capital）
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
    "series_description", "study_description", "slice_thickness_mm",
    # microwave / 重建参数（充当 source 配置维度）
    "antenna_position", "frequency_band", "signal_value",
    "sampling_rate", "parameter", "registration_type",
    # 通道/标记物（IMC 等多通道成像数据）
    "channel_color", "marker", "isotope_label",
    # 原始格式描述（弱信号，仅作 evidence 兜底）
    "original_format", "original_dimension", "original_dimensions",
    "original_shape", "data_shape",
    # 5903190_folder 风格的样本/切片/区域标识（即使不在 extra 里也可作为 trace）
    "slide_id", "region_id", "position_id", "acquisition_id", "tissue_type",
    "sample_index",
    # 其他源条件
    "condition",
)
RAW_SOURCE_KEYS = (
    "source_image_path", "source_path", "source_file", "source_filename", "source_file_name",
    "source_relative", "original_filename", "original_file", "original_path", "raw_image_path", "image_path", "source_dcm_relative",
    "source_npz", "npz_filename", "source_nifti_relative", "source_nifti_path", "bids_subject",
    # v2 追加：与 SOURCE_TRACE_KEYS 对齐的常见 raw.* 字段
    "source_volume_path", "source_label_path", "source_segmentation_path", "source_dicom_path",
    "source_image_nii", "source_image_nii_absolute", "source_image_nifti_path",
    "source_label_nifti", "source_label_nifti_path", "source_mask_nii",
    "source_ct_relative", "source_ct_path", "ct_path_relative", "ct_volume_path",
    "source_data_nifti", "original_data_nifti", "original_label_nifti", "original_label_filename",
    "raw_input_path", "source_path_used", "source_volume_relative", "source_volume_relpath",
    "source_volume_file", "source_volume_mhd", "source_mat_file",
    "han_seg_case_folder", "totalsegmentator_sample_folder", "bids_cohort_folder",
    "chaos_case_folder", "chaos_split_folder", "case_id", "sample_dir",
    # v2 第二轮补丁：raw.* 中常见的源/路径字段
    "source_label_from_dir", "source_absolute_path", "source_basename",
    "source_image_relpath", "source_image_relative_to_dataset",
    "source_npy_relative", "source_channel_relative",
    "h5_noisy_dataset_path", "h5_clean_dataset_path",
    "label_path", "label_source", "case_directory_relative",
    # v2 第三轮补丁
    "original_relative_path", "original_image_relative_path", "original_source_relpath",
    "original_absolute_path", "original_image_filename", "original_image_path",
    "original_video_filename", "original_video_path",
    "source_file_relative", "source_file_relpath", "source_file_relative_to_dataset_data",
    "source_image_relative_path", "source_image_relpath_from_data",
    "source_h5_relative_to_data_dir",
    "source_dicom_absolute_path", "source_npy_absolute",
    # v2 patch5: second_dataset/KimiK2.5_new
    "mat_file", "original_sample_name", "image_type",
    "class_name", "cell_class", "cell_line", "sample_type",
    "participant_id", "roi_location", "roi_number",
    "StudyInstanceUID", "slice_number", "extracted_slice",
    "dataset_split", "dataset_subset", "condition",
    "sampling_rate", "parameter",
)
STRICT_CONTENT_NULL_IS_EMPTY = True
ALLOW_META_TOTAL_MATCH_EXPECTED_N = False
META_QUALITY_SCORE_POWER = 1.08
META_QUALITY_SCORE_SCALE = 0.92


def safe_counter_majority(values: list[str]) -> str:
    vals = [v for v in values if v]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def load_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def normalize_token_text(v: Any) -> str:
    s = normalize_text(v)
    s = re.sub(r"[\s\-_/]+", " ", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", s)
    return s.strip()


def normalize_dimension(v: Any) -> str:
    s = normalize_text(v).replace("-", "").replace(" ", "")
    if s.startswith("2d"):
        return "2d"
    if s.startswith("3d"):
        return "3d"
    if "video" in s:
        return "video"
    return s


def normalize_modality(v: Any) -> str:
    s = normalize_text(v)
    if not s:
        return ""
    if s in {"ct", "computedtomography"} or s.startswith("ct"):
        return "ct"
    if s in {"mri", "mr"} or s.startswith("mri") or s == "mr":
        return "mri"
    if "microscopy" in s:
        return "microscopy"
    if "ultrasound" in s or s == "us":
        return "ultrasound"
    if "xray" in s or "x-ray" in s:
        return "xray"
    return s


def normalize_source_stem(v: Any) -> str | None:
    if not v:
        return None
    raw = str(v).strip().strip("'\"").replace("\\", "/")

    # medical-7-2020 常见来源路径：
    # radial_128_2/10/3DMR_Chest.mat -> radial_128_2_10_3DMR_Chest
    m = re.search(r"(radial_\d+_\d+)/(\d+)/([^/]+?)(?:\.[a-z0-9.]+)?$", raw, flags=re.I)
    if m:
        cfg, spokes, name = m.group(1), m.group(2), m.group(3)
        s = f"{cfg}_{spokes}_{name}"
        s = s.strip().lower().replace("-", "_").replace(" ", "_")
        s = re.sub(r"[^a-z0-9_]+", "", s)
        return s or None

    # medical-7-2020 另一类 subject_id：
    # 3DMR_Chest_radial_128_2_spokes10 / 3DMR_Chest_radial_128_2_10
    m = re.match(r"(.+?)_(radial_\d+_\d+)_(?:spokes)?(\d+)$", raw, flags=re.I)
    if m:
        name, cfg, spokes = m.group(1), m.group(2), m.group(3)
        s = f"{cfg}_{spokes}_{name}"
        s = s.strip().lower().replace("-", "_").replace(" ", "_")
        s = re.sub(r"[^a-z0-9_]+", "", s)
        return s or None

    s = raw.split("/")[-1]
    # dynamicnuclearnet/test_1/frame_0000(.tif) -> dynamicnuclearnet_test_1_frame_0000
    m = re.search(r"(dynamicnuclearnet)[/_](test_\d+)[/_](frame[_-]?\d+)", raw, flags=re.I)
    if m:
        ds, seq, frame = m.group(1), m.group(2), m.group(3).replace("-", "_")
        s = f"{ds}_{seq}_{frame}"
        s = s.strip().lower().replace("-", "_").replace(" ", "_")
        s = re.sub(r"[^a-z0-9_]+", "", s)
        return s or None

    # microUS_test_01.nii / test_01 这类来源标识应视为同一源；
    # 但如果本身包含 frame 信息，则不能截断成 test_x，否则会丢失帧级来源。
    m = re.search(r"(test_\d+)", s, flags=re.I)
    if m:
        if not re.search(r"frame[_-]?\d+", s, flags=re.I):
            return m.group(1).lower()
    for ext in [".ome.tiff", ".ome.tif", ".nii.gz", ".nii", ".mha", ".tiff", ".tif", ".czi", ".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".json", ".txt", ".npy", ".npz", ".h5", ".hdf5", ".mat", ".img", ".pgm", ".ppm", ".svs", ".vsi", ".avi", ".mp4", ".mov", ".webm", ".webp", ".pdf"]:
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break
    s = s.strip().lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    # nnU-Net / 医学分割常见命名：case_0000.nii.gz，其中 _0000 表示模态通道编号而非来源主体。
    # 对以 brain_p<id>_0000 结尾的来源，去掉通道后缀以对齐 GT 的主体级 source_stem。
    m = re.match(r"(brain_p\d+)_0000$", s)
    if m:
        s = m.group(1)
    # braTS/医学任务的 filename 常带模态后缀（flair/t1/t2/t1ce/seg...），GT 的 source_stem 往往不包含它们。
    for suf in ["_flair", "_t1ce", "_t1", "_t2", "_seg", "_mask"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s or None


def expand_source_aliases(v: Any) -> set[str]:
    """
    为来源标识生成等价别名集合（用于保持 strict 评估下的有限放宽）：
    - p23 / k08 / k8
    - sub-p23 / sub_k08
    - sub-p23_dti / sub_k08_dti
    """
    base = normalize_source_stem(v)
    if not base:
        return set()
    out = {base}

    # 1) 简写 patient/control ID（p23/k08）
    m_short = re.fullmatch(r"([pk])0*(\d+)", base)
    if m_short:
        grp = m_short.group(1)
        n = int(m_short.group(2))
        out.add(f"{grp}{n}")
        out.add(f"{grp}{n:02d}")
        out.add(f"sub_{grp}{n}")
        out.add(f"sub_{grp}{n:02d}")
        out.add(f"sub_{grp}{n}_dti")
        out.add(f"sub_{grp}{n:02d}_dti")

    # 2) BIDS subject / BIDS+modality（sub_p23 / sub_p23_dti）
    m_sub = re.fullmatch(r"sub_([pk])0*(\d+)(?:_dti)?", base)
    if m_sub:
        grp = m_sub.group(1)
        n = int(m_sub.group(2))
        out.add(f"{grp}{n}")
        out.add(f"{grp}{n:02d}")
        out.add(f"sub_{grp}{n}")
        out.add(f"sub_{grp}{n:02d}")
        out.add(f"sub_{grp}{n}_dti")
        out.add(f"sub_{grp}{n:02d}_dti")

    return out


def _is_placeholder_string(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    raw = v.strip()
    s = normalize_text(v)
    return s in MISSING_TOKENS or s in NULL_EQ_TOKENS or bool(PLACEHOLDER_LIKE_RE.match(raw))


def _field_present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, dict)):
        return len(v) > 0
    return True


def _field_nonmissing(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return not _is_placeholder_string(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            return math.isfinite(float(v))
        except Exception:
            return False
    if isinstance(v, list):
        return any(_field_nonmissing(x) for x in v)
    if isinstance(v, dict):
        return any(_field_nonmissing(x) for x in v.values())
    return bool(v)


def _is_placeholder_value(field_name: str, v: Any) -> bool:
    fk = normalize_text(field_name)
    if v is None:
        return True
    if isinstance(v, str):
        return _is_placeholder_string(v)
    if isinstance(v, bool):
        return v is False
    if isinstance(v, (int, float)):
        try:
            fv = float(v)
        except Exception:
            return True
        if not math.isfinite(fv):
            return True
        if fk in {"label", "class_id", "category_id", "num_classes", "total", "train", "val", "test"}:
            return False
        if fk in {"age_years", "length_chars", "confidence_global"} and fv in {0.0, -1.0}:
            return True
        return fv == -1.0
    if isinstance(v, list):
        if not v:
            return True
        return all(_is_placeholder_value(field_name, x) for x in v)
    if isinstance(v, dict):
        if not v:
            return True
        return all(_is_placeholder_value(str(k), x) for k, x in v.items())
    return not bool(v)


def is_effectively_empty(field_name: str, v: Any) -> bool:
    return _is_placeholder_value(field_name, v)


def find_image_path(images_root: Path, idx: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_root / "images" / f"{idx}{ext}"
        if p.is_file():
            return p
    return None


def validate_image_file(path: Path | None) -> bool:
    if path is None or not path.is_file() or path.stat().st_size <= 0:
        return False
    if Image is None:
        return True
    # 允许评估超大图（取消 PIL "decompression bomb" 防护，否则会抛 DecompressionBombError）
    try:
        Image.MAX_IMAGE_PIXELS = None  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im2:
            w, h = im2.size
        return bool(w and h)
    except (UnidentifiedImageError, OSError, ValueError, Exception):
        return False


def image_relpath_from_path(path: Path | None) -> str:
    if path is None:
        return ""
    return f"images/{path.name}"


def extract_source_from_notes(notes: Any) -> str | None:
    if not isinstance(notes, str):
        return None
    txt = notes.strip()
    if not txt:
        return None
    lower = txt.lower()
    if lower.startswith("converted from:"):
        return txt.split(":", 1)[1].strip()
    # v2 patch5: "Converted from <name>." 或 "Converted from <name>,"（无冒号，句号/逗号分隔）
    # 注意：文件名本身可能含 '.'（如 0.bmp），所以 stop 条件是 "句号+空格" 或 "逗号+空格" 或行尾。
    m = re.match(r"converted\s+from\s+(\S+?)(?:\.\s|,\s|$)", txt, flags=re.I)
    if m:
        return m.group(1).strip()
    # v2 patch5: "Original file: <name>, <other>: <val>" 或 "Source file: <name>, ..."
    #   注意 KimiK2.5_new 常用 "Original file:" 而不是 "original filename:"
    #   文件名后通常跟逗号 + "<Key>:" 形式的其他元数据，所以遇到第一个 ", " 后跟一个大写字母开头的词就停
    m = re.search(r"(?:original\s*(?:file|filename)|source\s*file)\s*:\s*([^,\n]+?)(?:\s*,\s+[A-Z][A-Za-z_ ]*\s*:|$)", txt, flags=re.I)
    if m:
        return m.group(1).strip()
    # 旧版兜底（兼容 "...original filename: xxx" 整行结尾）
    m = re.search(r"(?:original\s*filename|source\s*file)\s*:\s*(.+)$", txt, flags=re.I)
    if m:
        return m.group(1).strip()
    return None


def has_source_trace_evidence(extra: dict, ann: dict) -> bool:
    if isinstance(extra, dict):
        for k in SOURCE_TRACE_KEYS:
            if k in extra and _field_nonmissing(extra.get(k)):
                return True
        csv_label = extra.get("csv_label")
        if isinstance(csv_label, dict):
            if _field_nonmissing(csv_label.get("dcm")) or _field_nonmissing(csv_label.get("jpg")):
                return True
        labels = extra.get("labels")
        if isinstance(labels, dict) and _field_nonmissing(labels.get("slice_stem")):
            return True
    raw = ann.get("raw") if isinstance(ann.get("raw"), dict) else {}
    for k in RAW_SOURCE_KEYS:
        if k in raw and _field_nonmissing(raw.get(k)):
            return True
    # v2 patch4: 兼容模型把源/路径写到 record 块下的常见错误
    record = ann.get("record") if isinstance(ann.get("record"), dict) else {}
    for k in RAW_SOURCE_KEYS:
        if k in record and _field_nonmissing(record.get(k)):
            return True
    # 模型常把"原始文件名"放在 record.source_file / record.original_filename
    for k in ("source_file", "original_filename", "original_file", "source_filename",
              "source_path", "original_path", "source_relative", "original_relative_path",
              "source_image_path", "original_image_path"):
        if k in record and _field_nonmissing(record.get(k)):
            return True
    prov = ann.get("provenance") if isinstance(ann.get("provenance"), dict) else {}
    notes = prov.get("notes")
    if extract_source_from_notes(notes):
        return True
    return False


def branch_is_effectively_empty(obj: Any) -> bool:
    if obj is None:
        return True
    if isinstance(obj, dict):
        if not obj:
            return True
        return all(branch_is_effectively_empty(v) for v in obj.values())
    if isinstance(obj, list):
        return len(obj) == 0 or all(branch_is_effectively_empty(x) for x in obj)
    if isinstance(obj, str):
        return _is_placeholder_string(obj)
    if isinstance(obj, bool):
        return obj is False
    if isinstance(obj, (int, float)):
        try:
            fv = float(obj)
        except Exception:
            return True
        return (not math.isfinite(fv)) or fv == -1.0
    return not bool(obj)


def required_fields_present(obj: dict, fields: list[str], *, nonmissing: bool = True) -> bool:
    if not isinstance(obj, dict):
        return False
    for k in fields:
        if k not in obj:
            return False
        if nonmissing:
            if not _field_nonmissing(obj.get(k)) and not (normalize_text(k) == "label" and obj.get(k) == 0):
                return False
        else:
            if not _field_present(obj.get(k)):
                return False
    return True


def mean_score(vals: list[float]) -> float:
    vals = [float(v) for v in vals]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _safe_first_list_value(v: Any) -> Any:
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _is_meaningful(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return not _is_placeholder_string(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            return math.isfinite(float(v))
        except Exception:
            return False
    if isinstance(v, list):
        return any(_is_meaningful(x) for x in v)
    if isinstance(v, dict):
        return any(_is_meaningful(x) for x in v.values())
    return bool(v)


def _has_forbidden_desc_options(obj: Any) -> bool:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and (k.endswith("__desc") or k.endswith("__options")):
                return True
            if _has_forbidden_desc_options(v):
                return True
    elif isinstance(obj, list):
        return any(_has_forbidden_desc_options(x) for x in obj)
    return False


def exact_enum_match(actual: Any, expected: Any) -> bool:
    return normalize_text(actual) == normalize_text(expected)


def normalized_text_match(actual: Any, expected: Any) -> bool:
    return normalize_token_text(actual) == normalize_token_text(expected)


def numeric_match(actual: Any, expected: Any, tol: float = 0.0) -> bool:
    try:
        a = float(actual)
        e = float(expected)
        return abs(a - e) <= tol
    except Exception:
        return False


def path_stem_match(actual: Any, expected: Any) -> bool:
    a_alias = expand_source_aliases(actual)
    e_alias = expand_source_aliases(expected)
    return bool(a_alias and e_alias and (a_alias & e_alias))


def normalize_full_path(v: Any) -> str:
    """用于严格路径一致：规范化斜杠与大小写，避免仅 stem 相同时误判为满分。"""
    if v is None:
        return ""
    s = str(v).strip().replace("\\", "/")
    s = re.sub(r"/+", "/", s).strip("/")
    return s.lower()


def full_path_match(actual: Any, expected: Any) -> bool:
    return normalize_full_path(actual) == normalize_full_path(expected)


def set_overlap_score(actual: Any, expected: Any) -> float:
    if not isinstance(actual, list) or not isinstance(expected, list):
        return 0.0
    a = {normalize_token_text(x) for x in actual if normalize_token_text(x)}
    e = {normalize_token_text(x) for x in expected if normalize_token_text(x)}
    if not a and not e:
        return 1.0
    if not a or not e:
        return 0.0
    inter = len(a & e)
    union = len(a | e)
    return inter / union if union else 0.0


def infer_field_type(field_name: str) -> str:
    k = normalize_text(field_name)
    if k in {"task_type", "dimension", "schema_variant", "dataset_name", "modality_primary", "modality_secondary"}:
        return "enum"
    if (
        "path" in k
        or "source_file" in k
        or "filename" in k
        or "relative_path" in k
        or k.endswith("_relative")
        or k.endswith("_relpath")
    ):
        return "path"
    if k.startswith("num_") or k.endswith("_n") or k.endswith("_count") or k in {"total", "num_classes"}:
        return "number"
    return "text"


def field_match_score(
    field_name: str,
    actual: Any,
    expected: Any,
    *,
    path_mode: str = "stem",
    strict_text_keys: frozenset[str] | None = None,
) -> float:
    fk = normalize_text(field_name)

    if isinstance(expected, str) and normalize_text(expected) in NULL_EQ_TOKENS:
        return 1.0 if is_effectively_empty(field_name, actual) else 0.0

    if expected is None:
        if STRICT_CONTENT_NULL_IS_EMPTY:
            return 1.0 if is_effectively_empty(field_name, actual) else 0.0
        return -1.0

    st_keys = strict_text_keys or frozenset()
    strict_text = fk in st_keys

    ftype = infer_field_type(field_name)

    if isinstance(expected, list):
        if isinstance(actual, list):
            return round(set_overlap_score(actual, expected), 4)
        return 0.0

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return 1.0 if numeric_match(actual, expected, tol=0.0) else 0.0

    if isinstance(expected, bool):
        return 1.0 if actual is expected else 0.0

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0.0
        sub_scores: list[float] = []
        for k, ev in expected.items():
            s = field_match_score(
                str(k),
                actual.get(k),
                ev,
                path_mode=path_mode,
                strict_text_keys=st_keys,
            )
            if s >= 0:
                sub_scores.append(s)
        return round(sum(sub_scores) / len(sub_scores), 4) if sub_scores else 1.0

    if ftype == "enum":
        return 1.0 if exact_enum_match(actual, expected) else 0.0
    if ftype == "path":
        if path_mode == "full":
            return 1.0 if full_path_match(actual, expected) else 0.0
        if isinstance(expected, str) and ("/" in expected or "\\" in expected or re.search(r"\.[a-z0-9]{1,6}$", expected, flags=re.I)):
            return 1.0 if full_path_match(actual, expected) else 0.0
        return 1.0 if path_stem_match(actual, expected) else 0.0

    if isinstance(expected, str) and strict_text:
        return 1.0 if normalize_text(actual) == normalize_text(expected) else 0.0

    return 1.0 if normalized_text_match(actual, expected) else 0.0


def infer_expected_task_and_dimension(gt: dict) -> tuple[str, str]:
    samples = gt.get("samples") if isinstance(gt.get("samples"), dict) else {}
    gt_dim_hint = normalize_dimension(gt.get("dimension"))
    for s in samples.values():
        if not isinstance(s, dict):
            continue
        exp_tasks = s.get("expected_tasks")
        if isinstance(exp_tasks, dict):
            for tk in exp_tasks.keys():
                tkn = normalize_text(tk)
                if tkn in ALLOWED_TASKS:
                    task_obj = exp_tasks.get(tk) if isinstance(exp_tasks.get(tk), dict) else {}
                    sv = normalize_text(task_obj.get("schema_variant"))
                    if "3d" in sv:
                        return tkn, "3d"
                    if "video" in sv:
                        return tkn, "video"
                    if gt_dim_hint in ("2d", "3d", "video"):
                        return tkn, gt_dim_hint
                    return tkn, "2d"

    task_hint = normalize_text(gt.get("task_type"))
    if task_hint in ALLOWED_TASKS:
        return task_hint, normalize_dimension(gt.get("dimension")) or "2d"
    return "classification", "2d"


def get_extra(ann: dict) -> dict:
    sample = ((ann.get("context") or {}).get("sample") or {})
    extra = sample.get("extra") if isinstance(sample.get("extra"), dict) else {}
    out = dict(extra)
    sid = out.get("subject_id") or sample.get("subject_id")
    if isinstance(sid, str) and sid.strip():
        out["subject_id"] = sid.strip()
    return out


def derive_source_stem(ann: dict, extra: dict) -> str | None:
    raw = ann.get("raw") if isinstance(ann.get("raw"), dict) else {}
    csv_label = extra.get("csv_label") if isinstance(extra.get("csv_label"), dict) else {}
    labels = extra.get("labels") if isinstance(extra.get("labels"), dict) else {}
    record = ann.get("record") if isinstance(ann.get("record"), dict) else {}

    # microscopy-demo：很多标注把来源拆成 sequence + frame_index 或 path 片段。
    # 优先组装成 dynamicnuclearnet_test_x_frame_xxxx，与 GT source_stem 对齐。
    microscopy_path_candidates = [
        extra.get("source_image_relative"),
        extra.get("source_path"),
        extra.get("original_path"),
        extra.get("subpath"),
        extra.get("source_file"),
        extra.get("original_filename"),
        raw.get("source_path"),
        raw.get("source_file"),
        raw.get("original_path"),
        raw.get("image_path"),
    ]
    for pv in microscopy_path_candidates:
        if not isinstance(pv, str) or not pv.strip():
            continue
        mm = re.search(r"(dynamicnuclearnet)[/_](test_\d+)[/_](frame[_-]?\d+)", pv, flags=re.I)
        if mm:
            ds, seq, frame = mm.group(1), mm.group(2), mm.group(3).replace("-", "_")
            return normalize_source_stem(f"{ds}_{seq}_{frame}")

    seq = extra.get("sequence_id") or extra.get("sequence")
    if not _field_nonmissing(seq):
        sid = extra.get("subject_id")
        if isinstance(sid, str):
            m_sid = re.search(r"(test_\d+)", sid, flags=re.I)
            if m_sid:
                seq = m_sid.group(1)
    frame_token: str | None = None
    fi = extra.get("frame_index")
    if isinstance(fi, (int, float)) and float(fi).is_integer():
        frame_token = f"frame_{int(fi):04d}"
    if not frame_token:
        for fv in [extra.get("original_filename"), extra.get("source_file"), extra.get("source_path"), extra.get("subpath")]:
            if not isinstance(fv, str):
                continue
            m_f = re.search(r"(frame[_-]?\d+)", fv, flags=re.I)
            if m_f:
                frame_token = m_f.group(1).replace("-", "_").lower()
                break
    if _field_nonmissing(seq) and frame_token:
        seq_s = str(seq).strip().lower()
        ds_prefix = ""
        if normalize_text(record.get("dataset_name")) == "microscopy-demo":
            ds_prefix = "dynamicnuclearnet"
        for pv in microscopy_path_candidates:
            if isinstance(pv, str) and "dynamicnuclearnet" in pv.lower():
                ds_prefix = "dynamicnuclearnet"
                break
        if ds_prefix:
            return normalize_source_stem(f"{ds_prefix}_{seq_s}_{frame_token}")
        return normalize_source_stem(f"{seq_s}_{frame_token}")

    # mri-rd：GT 的 source_stem 会显式保留后缀语义（test_dcm / test_jpg）。
    if normalize_text(record.get("dataset_name")) == "mri-rd":
        mri_rd_paths = [
            extra.get("source_path"),
            extra.get("source_relative"),
            extra.get("source_file"),
            extra.get("original_filename"),
            extra.get("original_file"),
            raw.get("source_path"),
            raw.get("source_file"),
            raw.get("source_filename"),
            raw.get("source_relative"),
            raw.get("original_filename"),
            raw.get("original_file"),
        ]
        for pv in mri_rd_paths:
            if not isinstance(pv, str) or not pv.strip():
                continue
            base = pv.replace("\\", "/").split("/")[-1].strip()
            m_ext = re.match(r"(.+)\.(dcm|jpg|jpeg|png|bmp|tif|tiff)$", base, flags=re.I)
            if m_ext:
                stem = m_ext.group(1).strip()
                ext = m_ext.group(2).lower()
                if ext == "jpeg":
                    ext = "jpg"
                return normalize_source_stem(f"{stem}_{ext}")

    # medical-7-2020 等重建类数据：来源由 采样配置 + 参数 + 原始 anatomy 名称 共同确定
    src_name = (
        extra.get("source_label_name")
        or extra.get("anatomy_name")
        or extra.get("name_in_mat")
        or extra.get("mat_name_field")
    )
    if not _field_nonmissing(src_name):
        for k in ("original_filename", "original_file", "mat_basename", "mat_file"):
            vv = extra.get(k)
            if isinstance(vv, str) and vv.strip():
                src_name = Path(vv).stem
                break
    src_cfg = (
        extra.get("acceleration_dir")
        or extra.get("radial_config_dir")
        or extra.get("sampling_pattern")
        or extra.get("sampling_scheme")
        or extra.get("sampling_rate")  # v2 patch5: KimiK2.5_new uses 'sampling_rate'
    )
    src_param = (
        extra.get("parameter_setting")
        or extra.get("radial_undersampling_dir")
        or extra.get("radial_spokes")
        or extra.get("acceleration_factor")
        or extra.get("parameter")  # v2 patch5: KimiK2.5_new uses 'parameter'
    )
    composed_source: str | None = None
    if _field_nonmissing(src_name) and _field_nonmissing(src_cfg) and _field_nonmissing(src_param):
        composed_source = f"{str(src_cfg).strip()}_{str(src_param).strip()}_{str(src_name).strip()}"

    candidates: list[Any] = [
        composed_source,
        extra.get("source_file"),
        extra.get("source_relative"),
        extra.get("source_h5"),
        extra.get("original_h5"),
        extra.get("original_filename"),
        extra.get("original_file"),
        extra.get("source_filename"),
        extra.get("source_file_name"),
        extra.get("source_adc_mha_path"),
        extra.get("source_adc_path"),
        extra.get("source_image_tif"),
        extra.get("source_tif_path"),
        extra.get("experiment_id"),
        extra.get("source_swi_series_dir"),
        extra.get("source_dcm_relative"),
        extra.get("relative_dicom_path"),
        extra.get("dicom_relative_to_dataset_data"),
        extra.get("source_image_path"),
        extra.get("source_image_relative"),
        extra.get("raw_image_path"),
        extra.get("raw_mask_path"),
        extra.get("source_tiff_relative"),
        extra.get("source_relative_to_dataset_data"),
        extra.get("image_id"),
        extra.get("source_mat_relative"),
        extra.get("source_mat_path"),
        extra.get("source_npz"),
        extra.get("npz_filename"),
        extra.get("source_image_path_name"),
        extra.get("source_parquet"),
        extra.get("dcm_path_csv"),
        extra.get("slice_stem"),
        extra.get("original_stem"),
        extra.get("original_relative_path"),
        extra.get("source_relative_path"),
        extra.get("source_path"),
        extra.get("source_nifti_relative"),
        extra.get("source_nifti_path"),
        extra.get("bids_subject"),
        extra.get("original_path"),
        extra.get("source_rel_path"),
        extra.get("source_relpath"),
        extra.get("source_stem"),
        # === 新增同义 key：必须放在 subject_id 之前，因为更具体的完整路径优先级高于 ID ===
        extra.get("source_image_file"),
        extra.get("original_image_path"),
        extra.get("original_file_path"),
        extra.get("original_label_path"),
        extra.get("source_image"),
        extra.get("source_label"),
        extra.get("source_mask_file"),
        extra.get("source_label_file"),
        extra.get("segmentation_file"),
        extra.get("source_video"),
        extra.get("source_video_filename"),
        extra.get("source_flair_nifti"),
        extra.get("source_h5_file"),
        extra.get("source_dicom"),
        extra.get("source_image_member"),
        extra.get("source_label_member"),
        extra.get("raw_filename"),
        extra.get("raw_relative_path"),
        extra.get("raw_image_member_path"),
        extra.get("dicom_source_path"),
        extra.get("source_dataset_name"),
        extra.get("source_angle_path"),
        extra.get("source_dcm_path_csv"),
        extra.get("source_jpg_path_csv"),
        extra.get("filename"),
        # source_dir 是目录名（不含文件），放在 subject_id 后面作兜底
        # source_sample_dir 同理
        # =============================================================
        extra.get("subject_id"),
        extra.get("original_subject_id"),
        # v2 第三轮补丁：fallback candidates for stem derivation
        extra.get("original_relative_path"),
        extra.get("original_image_relative_path"),
        extra.get("original_source_relpath"),
        extra.get("original_absolute_path"),
        extra.get("original_image_filename"),
        extra.get("original_video_filename"),
        extra.get("original_video_path"),
        extra.get("source_file_relative"),
        extra.get("source_file_relpath"),
        extra.get("source_file_relative_to_dataset_data"),
        extra.get("source_csv_relative"),
        extra.get("source_npy_absolute"),
        extra.get("source_image_relative_path"),
        extra.get("source_image_relpath_from_data"),
        extra.get("source_image_relative_to_dataset"),
        extra.get("source_h5_relative_to_data_dir"),
        extra.get("source_dicom_absolute_path"),
        extra.get("drive_image_stem"),
        extra.get("image_reference_id"),
        extra.get("name"),
        extra.get("original_image"),
        # v2 patch5: KimiK2.5_new 新增字段（低优先级 fallback，仅当上面都没命中时使用）
        extra.get("mat_file"),
        extra.get("original_sample_name"),
        extra.get("StudyInstanceUID"),
        labels.get("slice_stem"),
        csv_label.get("dcm"),
        csv_label.get("jpg"),
        # source_subset 有时是“数据集来源目录名”，如果原始文件名存在，
        # 更应该优先用原始文件名/路径来匹配 GT 的 source_stem。
        extra.get("source_subset"),
        # 目录类兜底（只有目录、无文件名），放在 subject_id 之后
        extra.get("source_sample_dir"),
        extra.get("source_dir"),
        raw.get("source_image_path"),
        raw.get("source_dcm_relative"),
        raw.get("source_path"),
        raw.get("source_file"),
        raw.get("source_relative"),
        raw.get("source_filename"),
        raw.get("source_file_name"),
        raw.get("source_npz"),
        raw.get("npz_filename"),
        raw.get("source_nifti_relative"),
        raw.get("source_nifti_path"),
        raw.get("bids_subject"),
        raw.get("original_filename"),
        raw.get("original_file"),
        raw.get("original_path"),
        raw.get("raw_image_path"),
        raw.get("image_path"),
    ]

    # BONBID 类数据有时仅给 patient_id + visit，不给原始文件名。
    pid = extra.get("patient_id")
    vis = extra.get("visit")
    if isinstance(pid, str) and pid.strip() and isinstance(vis, str) and vis.strip():
        candidates.append(f"{pid.strip()}-{vis.strip()}-ADC_ss")
    elif isinstance(pid, str) and pid.strip():
        # BONBID 测试集常仅给 patient_id（visit 固定为 VISIT_01）
        candidates.append(f"{pid.strip()}-VISIT_01-ADC_ss")

    # brats20-training-data 等数据可能用 volume_id + slice_id 表示源，
    # GT 的 source_stem 往往是 volume_{id}_slice_{id}。
    vid = extra.get("volume_id")
    sid = extra.get("slice_id")
    if vid is not None and sid is not None:
        vid_s = str(vid).strip()
        sid_s = str(sid).strip()
        if vid_s and sid_s:
            candidates.insert(0, f"volume_{vid_s}_slice_{sid_s}")

    bisque_tags = extra.get("bisque_tags")
    if isinstance(bisque_tags, dict):
        candidates.append(bisque_tags.get("name"))

    for c in candidates:
        if _field_nonmissing(c):
            return normalize_source_stem(c)

    prov = ann.get("provenance") if isinstance(ann.get("provenance"), dict) else {}
    notes = prov.get("notes")
    source_from_notes = extract_source_from_notes(notes)
    if source_from_notes:
        return normalize_source_stem(source_from_notes)

    return None


# ============================================================
# v2 新增：返回"多候选 alias 集合"，不替代 derive_source_stem 的优先级。
# 在原 derive_source_stem 之上，另外扫描所有"看起来是源/路径/文件名"的
# extra/raw 字符串值，逐个 normalize_source_stem 并 expand_source_aliases，
# 与主候选合并成一个集合返回。caller 用集合 ∩ expected_alias 判断。
# ============================================================
_V2_PATH_HINT_RE = re.compile(
    r"[\\/]|\.(nii|nii\.gz|mha|mhd|nrrd|dcm|dicom|tif|tiff|png|jpg|jpeg|h5|hdf5|npz|npy|mat|json|tar|zip)$",
    re.I,
)
_V2_SCAN_KEY_RE = re.compile(
    r"(source|original|raw|label|mask|seg|file|path|nifti|nii|dicom|dcm|tif|tiff|"
    r"mhd|mha|nrrd|h5|npz|mat|name|stem|filename|image|volume|case|sample|"
    r"folder|dir|relative|relpath|cohort|split|series|study|sop|"
    # v2 patch5: KimiK2.5_new 字段
    r"subset|class|cell|participant|roi|condition|slice|extracted|"
    r"channel|isotope|tissue|antenna|frequency|sampling|parameter|description)",
    re.I,
)


def derive_source_stem_aliases_v2(ann: dict, extra: dict) -> set[str]:
    aliases: set[str] = set()

    # 1) 主候选：保留 derive_source_stem 的优先级（最重要）
    main = derive_source_stem(ann, extra)
    aliases |= expand_source_aliases(main)

    # 2) 兜底扫描：extra / raw 中所有看起来像源/路径/文件名的字符串值
    raw = ann.get("raw") if isinstance(ann.get("raw"), dict) else {}

    def _scan(d: dict) -> None:
        nonlocal aliases
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s:
                continue
            if not _V2_SCAN_KEY_RE.search(str(k)):
                continue
            # 仅在值像路径/带文件后缀，或本身是较短 ID 时才考虑
            if not (_V2_PATH_HINT_RE.search(s) or len(s) <= 96):
                continue
            ns = normalize_source_stem(s)
            if ns:
                aliases |= expand_source_aliases(ns)

    _scan(extra)
    _scan(raw)
    # v2 patch4: 兼容模型把源/路径写到 record 块下的常见错误（prompt 没明确要求位置）
    record = ann.get("record") if isinstance(ann.get("record"), dict) else {}
    _scan(record)
    return aliases


def parse_ground_truth(raw: dict) -> tuple[dict[str, dict], int]:
    samples = raw.get("samples") or raw.get("source_to_truth") or {}
    samples = {str(k): (v or {}) for k, v in samples.items() if v is not None}
    expected = raw.get("expected_count")
    if expected is None:
        expected = len(samples)
    return samples, int(expected)


def get_annotation_paths(model_root: Path) -> list[Path]:
    ann_dir = model_root / "agent1_data_organization" / "standardized_annotations"
    if not ann_dir.is_dir():
        return []
    return sorted(ann_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)


def _idx_sort_key(idx: str) -> int:
    return int(idx) if str(idx).isdigit() else -1


def infer_image_exists(images_root: Path, idx: str) -> bool:
    return find_image_path(images_root, idx) is not None


def extra_has_any_key(extra: dict, keys: list[str]) -> bool:
    if not isinstance(extra, dict):
        return False
    lowered = {str(k).lower() for k in extra.keys()}
    for rk in keys:
        rr = normalize_text(rk)
        for ek in lowered:
            if rr == ek:
                return True
    return False


def check_schema_validity(ann: dict, gt_format: dict | None = None) -> tuple[float, bool, dict[str, bool]]:
    checks: dict[str, bool] = {}

    fmt = gt_format if isinstance(gt_format, dict) else {}
    required_top = fmt.get("top_level_blocks") or ["record", "context", "media_geometry", "tasks"]
    checks["top_level_blocks"] = all(isinstance(ann.get(k), dict) for k in required_top)

    record = ann.get("record") if isinstance(ann.get("record"), dict) else {}
    context = ann.get("context") if isinstance(ann.get("context"), dict) else {}
    sample = context.get("sample") if isinstance(context.get("sample"), dict) else {}
    media_geometry = ann.get("media_geometry") if isinstance(ann.get("media_geometry"), dict) else {}
    media = media_geometry.get("media") if isinstance(media_geometry.get("media"), dict) else {}
    spatial = media_geometry.get("spatial") if isinstance(media_geometry.get("spatial"), dict) else {}
    tasks = ann.get("tasks") if isinstance(ann.get("tasks"), dict) else {}
    extra = sample.get("extra") if isinstance(sample.get("extra"), dict) else {}

    record_required = fmt.get("record_required") or ["schema_version", "dataset_name", "image_path"]
    checks["record_required"] = required_fields_present(record, list(record_required), nonmissing=True)
    checks["record_image_path_rel"] = isinstance(record.get("image_path"), str) and normalize_full_path(record.get("image_path")).startswith("images/")

    context_sample_required = fmt.get("context_sample_required") or ["extra"]
    sample_req_ok = True
    for k in context_sample_required:
        if k == "extra":
            sample_req_ok = sample_req_ok and isinstance(sample.get("extra"), dict)
        else:
            sample_req_ok = sample_req_ok and _field_present(sample.get(k))
    checks["context_sample_required"] = sample_req_ok

    sample_required_fields = fmt.get("sample_required_fields") or []
    checks["sample_required_fields"] = all(_field_nonmissing(sample.get(k)) for k in sample_required_fields) if sample_required_fields else True

    sample_non_missing_fields = fmt.get("sample_non_missing_fields") or []
    checks["sample_non_missing_fields"] = all(k in sample and sample.get(k) is not None for k in sample_non_missing_fields) if sample_non_missing_fields else True

    extra_required_any = fmt.get("extra_required_any") or []
    checks["extra_required_any"] = any(_field_nonmissing(extra.get(k)) for k in extra_required_any if isinstance(k, str)) if extra_required_any else True

    extra_non_missing_fields = fmt.get("extra_non_missing_fields") or []
    checks["extra_non_missing_fields"] = any(_field_nonmissing(extra.get(k)) for k in extra_non_missing_fields if isinstance(k, str)) if extra_non_missing_fields else True

    checks["media_required"] = bool(normalize_text(media.get("task_type"))) and bool(normalize_dimension(media.get("dimension")))
    checks["forbidden_desc_options"] = not _has_forbidden_desc_options(ann)

    dim = normalize_dimension(media.get("dimension"))
    common_branch = spatial.get("common")
    three_d_branch = spatial.get("three_d")
    video_branch = spatial.get("video")
    if dim == "2d":
        checks["spatial_branch_selected"] = isinstance(common_branch, dict) and not branch_is_effectively_empty(common_branch)
        checks["spatial_branch_exclusive"] = branch_is_effectively_empty(three_d_branch) and branch_is_effectively_empty(video_branch)
    elif dim == "3d":
        checks["spatial_branch_selected"] = isinstance(three_d_branch, dict) and not branch_is_effectively_empty(three_d_branch)
        checks["spatial_branch_exclusive"] = branch_is_effectively_empty(common_branch) and branch_is_effectively_empty(video_branch)
    elif dim == "video":
        checks["spatial_branch_selected"] = isinstance(video_branch, dict) and not branch_is_effectively_empty(video_branch)
        checks["spatial_branch_exclusive"] = branch_is_effectively_empty(common_branch) and branch_is_effectively_empty(three_d_branch)
    else:
        checks["spatial_branch_selected"] = False
        checks["spatial_branch_exclusive"] = False

    task_type = normalize_text(media.get("task_type"))
    task_req_map = fmt.get("task_required_fields") if isinstance(fmt.get("task_required_fields"), dict) else {}
    active_branch = tasks.get(task_type) if isinstance(tasks.get(task_type), dict) else {}
    req_fields = task_req_map.get(task_type) if isinstance(task_req_map.get(task_type), list) else ["schema_variant"]
    checks["task_branch_required"] = bool(active_branch) and required_fields_present(active_branch, list(req_fields), nonmissing=True)

    inactive_ok = True
    for other_task in ALLOWED_TASKS:
        if other_task == task_type:
            continue
        if not branch_is_effectively_empty(tasks.get(other_task)):
            inactive_ok = False
            break
    checks["task_branch_exclusive"] = inactive_ok

    score = mean_score([1.0 if ok else 0.0 for ok in checks.values()])
    return score, all(checks.values()), checks


def compute_semantic_score(
    ann: dict,
    sample_expected: dict,
    expected_task: str,
    expected_dim: str,
    content_spec: dict | None = None,
) -> float:
    sample = ((ann.get("context") or {}).get("sample") or {})
    media = (((ann.get("media_geometry") or {}).get("media")) or {})
    record = ann.get("record") if isinstance(ann.get("record"), dict) else {}
    extra = get_extra(ann)

    exp_modality = normalize_modality((sample_expected.get("expected_context_sample") or {}).get("modality_primary"))
    exp_dataset = normalize_text((sample_expected.get("expected_record") or {}).get("dataset_name"))
    exp_cls = normalize_token_text(sample_expected.get("class"))
    exp_map_type = normalize_token_text(sample_expected.get("map_type"))

    pred_modality = normalize_modality(sample.get("modality_primary"))
    pred_task = normalize_text(media.get("task_type"))
    pred_dim = normalize_dimension(media.get("dimension"))
    pred_dataset = normalize_text(record.get("dataset_name"))
    pred_map_type = normalize_token_text(
        extra.get("map_type")
        or media.get("annotation_type")
        or media.get("map_type")
    )

    tasks = ann.get("tasks") if isinstance(ann.get("tasks"), dict) else {}

    strict_dataset = bool((content_spec or {}).get("strict_dataset_name_match"))

    def _contains_match(a: str, b: str) -> bool:
        return a == b or (a and b and (a in b or b in a))

    pred_cls = ""
    if expected_task == "classification":
        cls = tasks.get("classification") if isinstance(tasks.get("classification"), dict) else {}
        pred_cls = normalize_token_text(
            cls.get("label_name") or extra.get("label_text") or extra.get("class_name")
        )
    elif expected_task in {"segmentation", "detection"}:
        branch = tasks.get(expected_task) if isinstance(tasks.get(expected_task), dict) else {}
        classes_obj = branch.get("classes") or branch.get("class_names")
        if isinstance(classes_obj, dict):
            vals: list[str] = []
            for _, v in classes_obj.items():
                if not _is_meaningful(v):
                    continue
                nt = normalize_token_text(v)
                if nt == "background":
                    continue
                vals.append(str(v))
            pred_cls = normalize_token_text("|".join(vals))
        if not pred_cls:
            pred_cls = normalize_token_text(extra.get("label_text") or extra.get("class_name") or extra.get("label_name") or "")

    items: list[tuple[float, bool]] = []
    items.append((0.25, pred_task == expected_task))
    items.append((0.15, pred_dim == expected_dim))

    if exp_modality:
        items.append((0.20, pred_modality == exp_modality))
    if exp_dataset:
        ds_ok = normalize_text(pred_dataset) == normalize_text(exp_dataset) if strict_dataset else _contains_match(pred_dataset, exp_dataset)
        items.append((0.15, ds_ok))
    if exp_cls:
        items.append((0.15, pred_cls == exp_cls))
    if exp_map_type:
        items.append((0.10, pred_map_type == exp_map_type))

    denom = sum(w for w, _ in items) or 1.0
    field_score = sum(w * (1.0 if ok else 0.0) for w, ok in items) / denom

    path_mode, strict_text_keys = _content_fidelity_match_opts(content_spec)
    payload_exact_scores: list[float] = []
    exp_tasks = sample_expected.get("expected_tasks")
    if isinstance(exp_tasks, dict) and exp_tasks:
        for task_name, task_expect in exp_tasks.items():
            if not isinstance(task_expect, dict):
                continue
            actual_task = tasks.get(task_name) if isinstance(tasks.get(task_name), dict) else {}
            s, _ = score_expected_block(
                actual_task,
                task_expect,
                path_mode=path_mode,
                strict_text_keys=strict_text_keys,
            )
            payload_exact_scores.append(s)

    payload_presence_checks: list[float] = []
    active_branch = tasks.get(expected_task) if isinstance(tasks.get(expected_task), dict) else {}
    if expected_task == "classification":
        payload_presence_checks.append(1.0 if "label" in active_branch else 0.0)
        payload_presence_checks.append(1.0 if _field_nonmissing(active_branch.get("label_name") or extra.get("label_text") or extra.get("class_name")) else 0.0)
    elif expected_task == "detection":
        payload_presence_checks.append(1.0 if _field_nonmissing(active_branch.get("schema_variant")) else 0.0)
        payload_presence_checks.append(1.0 if _field_nonmissing(active_branch.get("classes") or active_branch.get("class_names")) else 0.0)
        payload_presence_checks.append(1.0 if _field_nonmissing(active_branch.get("boxes2d") or active_branch.get("boxes3d") or active_branch.get("frame_boxes2d") or extra.get("bbox_file")) else 0.0)
    elif expected_task == "segmentation":
        payload_presence_checks.append(1.0 if _field_nonmissing(active_branch.get("schema_variant")) else 0.0)
        payload_presence_checks.append(1.0 if _field_nonmissing(active_branch.get("classes") or active_branch.get("class_names")) else 0.0)
        payload_presence_checks.append(1.0 if _field_nonmissing(active_branch.get("coco_2d") or active_branch.get("mask_3d") or active_branch.get("video_coco") or extra.get("label_file") or extra.get("mask_file") or extra.get("seg_file")) else 0.0)

    payload_presence = sum(payload_presence_checks) / len(payload_presence_checks) if payload_presence_checks else 0.0
    payload_exact = sum(payload_exact_scores) / len(payload_exact_scores) if payload_exact_scores else payload_presence

    score = 0.70 * field_score + 0.10 * payload_presence + 0.20 * payload_exact
    return round(score, 4)


def compute_information_quality(extra: dict, tasks_obj: dict, media_obj: dict, content_spec: dict | None) -> tuple[float, dict]:
    groups = (content_spec or {}).get("expected_extra_key_groups")
    mode_any = normalize_text((content_spec or {}).get("expected_extra_keys_mode")) == "any"

    completeness = 0.0
    validity = 0.0

    if not isinstance(groups, dict) or not groups:
        candidate_keys = [str(k) for k in extra.keys()]
        if candidate_keys:
            present = sum(1 for k in candidate_keys if _field_present(extra.get(k)))
            valid = sum(1 for k in candidate_keys if not _is_placeholder_value(k, extra.get(k)))
            completeness = present / len(candidate_keys)
            validity = valid / len(candidate_keys)
        else:
            completeness = 0.0
            validity = 0.0
    else:
        valid_groups = [(k, v) for k, v in groups.items() if isinstance(v, list) and v]
        n = max(len(valid_groups), 1)

        if mode_any:
            present_hit = False
            valid_hit = False
            for _, keys in valid_groups:
                present = any(k in extra for k in keys)
                valid = any((k in extra) and (not _is_placeholder_value(str(k), extra.get(k))) for k in keys)
                present_hit = present_hit or present
                valid_hit = valid_hit or valid
            completeness = 1.0 if present_hit else 0.0
            validity = 1.0 if valid_hit else 0.0
        else:
            present_count = 0
            valid_count = 0
            for _, keys in valid_groups:
                present = any(k in extra for k in keys)
                valid = any((k in extra) and (not _is_placeholder_value(str(k), extra.get(k))) for k in keys)
                present_count += 1 if present else 0
                valid_count += 1 if valid else 0
            completeness = present_count / n
            validity = valid_count / n

    task_type = normalize_text(media_obj.get("task_type"))
    branch = tasks_obj.get(task_type) if isinstance(tasks_obj.get(task_type), dict) else {}

    branch_signal = 0.0
    if task_type == "classification":
        branch_signal = mean_score([
            1.0 if _field_nonmissing(branch.get("schema_variant")) else 0.0,
            1.0 if "label" in branch else 0.0,
            1.0 if _field_nonmissing(branch.get("label_name") or extra.get("label_text") or extra.get("class_name")) else 0.0,
        ])
    elif task_type == "detection":
        branch_signal = mean_score([
            1.0 if _field_nonmissing(branch.get("schema_variant")) else 0.0,
            1.0 if _field_nonmissing(branch.get("classes") or branch.get("class_names")) else 0.0,
            1.0 if _field_nonmissing(branch.get("boxes2d") or branch.get("boxes3d") or branch.get("frame_boxes2d") or extra.get("bbox_file")) else 0.0,
        ])
    elif task_type == "segmentation":
        branch_signal = mean_score([
            1.0 if _field_nonmissing(branch.get("schema_variant")) else 0.0,
            1.0 if _field_nonmissing(branch.get("classes") or branch.get("class_names")) else 0.0,
            1.0 if _field_nonmissing(branch.get("coco_2d") or branch.get("mask_3d") or branch.get("video_coco") or extra.get("label_file") or extra.get("mask_file") or extra.get("seg_file")) else 0.0,
        ])

    strict_req = (content_spec or {}).get("strict_required_extra_keys")
    if isinstance(strict_req, list) and strict_req:
        pool = [str(k) for k in strict_req]
    elif isinstance(groups, dict) and groups:
        pool = []
        for _, keys in groups.items():
            if isinstance(keys, list):
                pool.extend([str(k) for k in keys])
        pool = sorted(set(pool))
    else:
        pool = [str(k) for k in extra.keys()]

    if pool:
        extra_signal = sum(1 for k in pool if (k in extra) and (not _is_placeholder_value(k, extra.get(k)))) / len(pool)
    else:
        extra_signal = 0.0

    usefulness = math.sqrt(max(branch_signal, 0.0) * max(extra_signal, 0.0)) if branch_signal > 0 and extra_signal > 0 else 0.0

    texts = [normalize_token_text(v) for v in extra.values() if isinstance(v, str) and not _is_placeholder_string(v) and normalize_token_text(v)]
    redundancy_ratio = (len(set(texts)) / len(texts)) if texts else 0.0
    signal_density = extra_signal
    non_redundancy = 0.5 * redundancy_ratio + 0.5 * signal_density

    score = 0.25 * completeness + 0.25 * validity + 0.30 * usefulness + 0.20 * non_redundancy
    return round(score, 4), {
        "completeness": round(completeness, 4),
        "validity": round(validity, 4),
        "usefulness": round(usefulness, 4),
        "non_redundancy": round(non_redundancy, 4),
        "branch_signal": round(branch_signal, 4),
        "extra_signal": round(extra_signal, 4),
    }


def score_expected_block(
    actual_obj: dict,
    expected_obj: dict,
    *,
    path_mode: str = "stem",
    strict_text_keys: frozenset[str] | None = None,
) -> tuple[float, dict]:
    if not isinstance(expected_obj, dict) or not expected_obj:
        return 1.0, {}

    per_field = {}
    scored = []

    for k, ev in expected_obj.items():
        s = field_match_score(
            k,
            actual_obj.get(k),
            ev,
            path_mode=path_mode,
            strict_text_keys=strict_text_keys,
        )
        if s < 0:
            continue
        per_field[k] = round(s, 4)
        scored.append(s)

    if not scored:
        return 1.0, per_field
    return round(sum(scored) / len(scored), 4), per_field


def _content_fidelity_match_opts(content_spec: dict | None) -> tuple[str, frozenset[str]]:
    """从 GT 的 content 块读取更严匹配选项，降低「仅 stem/模糊 token」带来的虚高。"""
    if not isinstance(content_spec, dict):
        return "stem", frozenset()
    path_mode = "full" if content_spec.get("strict_path_full_match") else "stem"
    raw = content_spec.get("strict_content_text_keys")
    if isinstance(raw, list):
        st = frozenset(normalize_text(str(x)) for x in raw if str(x).strip())
    else:
        st = frozenset()
    return path_mode, st


def compute_content_fidelity(
    ann: dict,
    sample_expected: dict,
    content_spec: dict | None = None,
    agg_mode: str = "arithmetic",
) -> tuple[float, dict]:
    if not isinstance(sample_expected, dict) or not sample_expected:
        return 0.0, {}

    path_mode, strict_text_keys = _content_fidelity_match_opts(content_spec)

    record = ann.get("record") if isinstance(ann.get("record"), dict) else {}
    context_sample = ((ann.get("context") or {}).get("sample") or {})
    extra = get_extra(ann)
    tasks = ann.get("tasks") if isinstance(ann.get("tasks"), dict) else {}
    media = (((ann.get("media_geometry") or {}).get("media")) or {})

    details: dict[str, Any] = {}
    block_scores: list[float] = []

    for expect_name, actual_obj in [
        ("expected_record", record),
        ("expected_context_sample", context_sample),
        ("expected_extra", extra),
    ]:
        exp = sample_expected.get(expect_name)
        if isinstance(exp, dict) and exp:
            s, d = score_expected_block(
                actual_obj,
                exp,
                path_mode=path_mode,
                strict_text_keys=strict_text_keys,
            )
            details[expect_name] = d
            block_scores.append(s)

    exp_keys = sample_expected.get("expected_extra_keys")
    if isinstance(exp_keys, list) and exp_keys:
        key_scores = {}
        vals = []
        for k in exp_keys:
            kk = str(k)
            ok = 1.0 if (kk in extra and not _is_placeholder_value(kk, extra.get(kk))) else 0.0
            key_scores[kk] = ok
            vals.append(ok)
        details["expected_extra_keys"] = key_scores
        block_scores.append(sum(vals) / len(vals))

    exp_tasks = sample_expected.get("expected_tasks")
    if isinstance(exp_tasks, dict) and exp_tasks:
        task_scores = []
        details["expected_tasks"] = {}
        for task_name, task_expect in exp_tasks.items():
            if not isinstance(task_expect, dict):
                continue
            actual_task = tasks.get(task_name) if isinstance(tasks.get(task_name), dict) else {}
            s, d = score_expected_block(
                actual_task,
                task_expect,
                path_mode=path_mode,
                strict_text_keys=strict_text_keys,
            )
            details["expected_tasks"][task_name] = d
            task_scores.append(s)
        if task_scores:
            block_scores.append(sum(task_scores) / len(task_scores))

    geb = (content_spec or {}).get("global_expected_blocks") if isinstance(content_spec, dict) else None
    if isinstance(geb, dict) and geb:
        global_scores: list[float] = []
        details["global_expected_blocks"] = {}
        if isinstance(geb.get("record"), dict):
            s, d = score_expected_block(record, geb["record"], path_mode=path_mode, strict_text_keys=strict_text_keys)
            details["global_expected_blocks"]["record"] = d
            global_scores.append(s)
        if isinstance(geb.get("context_sample"), dict):
            s, d = score_expected_block(context_sample, geb["context_sample"], path_mode=path_mode, strict_text_keys=strict_text_keys)
            details["global_expected_blocks"]["context_sample"] = d
            global_scores.append(s)
        if isinstance(geb.get("media"), dict):
            s, d = score_expected_block(media, geb["media"], path_mode=path_mode, strict_text_keys=strict_text_keys)
            details["global_expected_blocks"]["media"] = d
            global_scores.append(s)
        if isinstance(geb.get("tasks"), dict):
            details["global_expected_blocks"]["tasks"] = {}
            local_scores: list[float] = []
            for task_name, task_expect in geb["tasks"].items():
                if not isinstance(task_expect, dict):
                    continue
                actual_task = tasks.get(task_name) if isinstance(tasks.get(task_name), dict) else {}
                s, d = score_expected_block(actual_task, task_expect, path_mode=path_mode, strict_text_keys=strict_text_keys)
                details["global_expected_blocks"]["tasks"][task_name] = d
                local_scores.append(s)
            if local_scores:
                global_scores.append(sum(local_scores) / len(local_scores))
        if global_scores:
            block_scores.append(sum(global_scores) / len(global_scores))

    if not block_scores:
        return 0.0, details

    nonzero_blocks = [max(min(s, 1.0), 0.0) for s in block_scores]
    if normalize_text(agg_mode) == "geometric":
        score = _geometric_mean(nonzero_blocks)
    else:
        score = sum(nonzero_blocks) / len(nonzero_blocks)
    return round(score, 4), details


def get_meta_strict_value(meta: dict, dotted_key: str) -> Any:
    """读取 data_meta 中由 dotted key 指定的值，支持 task_type.0 / modality_primary.0 / data_volume.total。"""
    parts = dotted_key.split(".")
    cur: Any = meta
    for p in parts:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(p) if p in cur else None
        elif isinstance(cur, list):
            try:
                idx = int(p)
                cur = cur[idx] if 0 <= idx < len(cur) else None
            except ValueError:
                return None
        elif isinstance(cur, (str, int, float, bool)) and str(p).isdigit() and int(p) == 0:
            return cur
        else:
            return None
    return cur


def strict_meta_value_matches(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return bool(actual) == expected
    if isinstance(expected, (int, float)):
        try:
            return int(actual) == int(expected)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str) and normalize_text(expected) in {"na", "n/a", "none", "unknown", "tbd"}:
        # GT 用 NA 表示「不强制字面一致」，只要实际值有意义即算通过
        return _is_meaningful(actual)
    return normalize_text(actual) == normalize_text(expected)


def strict_meta_value_matches_data_volume_total(
    actual: Any, expected: Any, expected_n: int
) -> bool:
    """严格模式下默认只接受 GT 指定值；仅当显式允许时才回退到 expected_n。"""
    try:
        if int(actual) == int(expected):
            return True
    except (TypeError, ValueError):
        return False
    if ALLOW_META_TOTAL_MATCH_EXPECTED_N:
        try:
            return int(actual) == int(expected_n)
        except (TypeError, ValueError):
            return False
    return False


def _geometric_mean(vals: list[float]) -> float:
    """几何平均：任一维为 0 则整体为 0，比算术平均更惩罚短板。"""
    if not vals:
        return 0.0
    if any(v <= 0.0 for v in vals):
        return 0.0
    p = 1.0
    for v in vals:
        p *= float(v)
    return p ** (1.0 / len(vals))


# 元信息总分校准参数已在文件顶部定义

def evaluate_data_meta_quality(model_root: Path, expected_n: int, gt: dict, parsed_samples: list[dict]) -> dict:
    meta_path = model_root / "agent1_data_organization" / "data_meta.json"
    out: dict[str, Any] = {
        "present": False,
        "valid_json": False,
        "meta_schema_validity": 0.0,
        "meta_semantic_correctness": 0.0,
        "meta_content_fidelity": 0.0,
        "strict_equals_score": None,
        "score_raw_geometric": 0.0,
        "score": 0.0,
        "issues": [],
    }
    if not meta_path.is_file():
        out["issues"].append("data_meta.json 缺失")
        return out

    out["present"] = True
    meta = load_json(meta_path)
    if meta is None:
        out["issues"].append("data_meta.json 不是合法 JSON")
        return out

    if not isinstance(meta, dict):
        out["issues"].append("data_meta.json 顶层必须是对象")
        return out

    out["valid_json"] = True
    dm = gt.get("data_meta_expected") if isinstance(gt.get("data_meta_expected"), dict) else {}

    required_keys = dm.get("required_keys") or [
        "dataset_name",
        "modality_primary",
        "data_volume",
        "task_type",
    ]
    dv_required = dm.get("data_volume_required_fields") or ["total"]

    schema_checks: list[bool] = []
    for k in required_keys:
        v = meta.get(k)
        ok = v is not None
        if isinstance(v, str):
            ok = bool(v.strip()) and normalize_text(v) not in MISSING_TOKENS
        if isinstance(v, list):
            ok = len(v) > 0 and any(_field_nonmissing(x) for x in v)
        schema_checks.append(ok)

    dv = meta.get("data_volume") if isinstance(meta.get("data_volume"), dict) else {}
    schema_checks.append(isinstance(meta.get("data_volume"), dict))
    for sub in dv_required:
        vsub = dv.get(sub) if isinstance(dv, dict) else None
        schema_checks.append(vsub is not None and not _is_placeholder_value(str(sub), vsub))

    mm_task_obj = meta.get("task_type")
    mm_task_values = []
    if isinstance(mm_task_obj, list):
        mm_task_values = [normalize_text(x) for x in mm_task_obj if normalize_text(x)]
    elif normalize_text(mm_task_obj):
        mm_task_values = [normalize_text(mm_task_obj)]
    schema_checks.append(any(t in ALLOWED_TASKS for t in mm_task_values))

    opt_rec = dm.get("optional_but_recommended")
    if isinstance(opt_rec, list) and opt_rec:
        for ok in opt_rec:
            if not isinstance(ok, str) or not ok.strip():
                continue
            schema_checks.append(_field_nonmissing(meta.get(ok.strip())))

    out["meta_schema_validity"] = round(sum(schema_checks) / len(schema_checks), 4) if schema_checks else 0.0

    gt_dataset_name = normalize_text(gt.get("dataset_name"))
    gt_modality = normalize_modality(gt.get("modality_primary"))
    exp_task, _ = infer_expected_task_and_dimension(gt)

    mm_dataset = normalize_text(meta.get("dataset_name"))
    mm_modality = normalize_modality(_safe_first_list_value(meta.get("modality_primary")))

    sem_parts: list[float] = []
    if gt_dataset_name:
        sem_parts.append(1.0 if mm_dataset == gt_dataset_name else 0.0)
    if gt_modality:
        sem_parts.append(1.0 if mm_modality == gt_modality else 0.0)
    if exp_task:
        sem_parts.append(1.0 if exp_task in mm_task_values else 0.0)

    content_block = gt.get("content") if isinstance(gt.get("content"), dict) else {}
    exp_dn = content_block.get("expected_dataset_name")
    if isinstance(exp_dn, str) and exp_dn.strip():
        sem_parts.append(1.0 if mm_dataset == normalize_text(exp_dn) else 0.0)

    gt_classes = gt.get("classes")
    if isinstance(gt_classes, list) and gt_classes and exp_task:
        nc = ((meta.get("num_classes_per_task") or {}).get(exp_task) or {}).get("num_classes")
        sem_parts.append(1.0 if isinstance(nc, int) and nc == len(gt_classes) else 0.0)

    out["meta_semantic_correctness"] = round(sum(sem_parts) / len(sem_parts), 4) if sem_parts else 0.0

    strict_se = dm.get("strict_equals") if isinstance(dm.get("strict_equals"), dict) else {}
    strict_equals_matches: list[bool] = []
    strict_equals_detail: dict[str, bool] = {}
    for sk, ev in strict_se.items():
        if not isinstance(sk, str):
            continue
        act = get_meta_strict_value(meta, sk)
        if sk == "data_volume.total":
            ok = strict_meta_value_matches_data_volume_total(act, ev, expected_n)
        else:
            ok = strict_meta_value_matches(act, ev)
        strict_equals_matches.append(ok)
        strict_equals_detail[sk] = ok
    if strict_equals_matches:
        hit = sum(strict_equals_matches) / len(strict_equals_matches)
        out["strict_equals_score"] = round(hit, 4)
        out["strict_equals_detail"] = strict_equals_detail
        out["strict_equals_all_pass"] = all(strict_equals_matches)
    else:
        out["strict_equals_score"] = None

    n_images_present = sum(1 for x in parsed_samples if x.get("image_exists"))
    n_images_valid = sum(1 for x in parsed_samples if x.get("image_valid"))
    n_pairs_valid = sum(1 for x in parsed_samples if x.get("pair_valid"))
    content_checks: list[bool] = []
    total_meta = dv.get("total") if isinstance(dv, dict) else None
    content_checks.append(isinstance(total_meta, (int, float)))
    if isinstance(total_meta, (int, float)):
        raw_vol = strict_se.get("data_volume.total")
        if raw_vol is not None:
            try:
                content_checks.append(int(total_meta) == int(raw_vol))
            except (TypeError, ValueError):
                content_checks.append(False)
        else:
            content_checks.append(int(total_meta) == int(expected_n))

    vin_raw = meta.get("valid_image_n")
    vin_ok = isinstance(vin_raw, dict)
    vin = vin_raw if vin_ok else {}
    vin_total = vin.get("total") if vin_ok else None
    content_checks.append(vin_ok)
    content_checks.append(isinstance(vin_total, (int, float)))
    if isinstance(vin_total, (int, float)):
        content_checks.append(int(vin_total) == int(n_images_valid))
        content_checks.append(int(vin_total) <= int(n_images_present))
        content_checks.append(int(vin_total) <= int(n_pairs_valid))

    out["meta_content_fidelity"] = round(sum(content_checks) / len(content_checks), 4) if content_checks else 0.0

    parts = [
        float(out["meta_schema_validity"]),
        float(out["meta_semantic_correctness"]),
        float(out["meta_content_fidelity"]),
    ]
    if out["strict_equals_score"] is not None:
        parts.append(float(out["strict_equals_score"]))
    raw_geom = _geometric_mean(parts)
    out["score_raw_geometric"] = round(raw_geom, 4)
    out["score"] = round(
        (raw_geom ** META_QUALITY_SCORE_POWER) * META_QUALITY_SCORE_SCALE,
        4,
    )
    return out


def evaluate_model(model_name: str, model_root: Path, ground_truth: dict, content_agg: str = "arithmetic") -> dict:
    samples_gt, n_valid = parse_ground_truth(ground_truth)
    expected_task, expected_dim = infer_expected_task_and_dimension(ground_truth)
    content_spec = ground_truth.get("content") if isinstance(ground_truth.get("content"), dict) else None
    gt_format = ground_truth.get("format") if isinstance(ground_truth.get("format"), dict) else None

    images_root = model_root / "agent1_data_organization"
    ann_paths = get_annotation_paths(model_root)
    gt_indices = set(samples_gt.keys())
    ann_by_idx_all = {p.stem: p for p in ann_paths}
    ann_source_buckets: dict[str, list[str]] = {}
    for ann_idx, ann_path in ann_by_idx_all.items():
        ann_obj = load_json(ann_path)
        if not isinstance(ann_obj, dict):
            continue
        extra = get_extra(ann_obj)
        pred_source = derive_source_stem(ann_obj, extra)
        if not pred_source:
            continue
        ann_source_buckets.setdefault(pred_source, []).append(ann_idx)
    for k, v in ann_source_buckets.items():
        ann_source_buckets[k] = sorted(v, key=_idx_sort_key)
    used_ann_indices: set[str] = set()

    parsed_samples: list[dict] = []
    n_json_present = 0
    n_valid_json = 0
    source_scored_n = 0

    for idx in sorted(gt_indices, key=_idx_sort_key):
        expected = samples_gt.get(idx) if isinstance(samples_gt.get(idx), dict) else {}

        expected_sources: list[str] = []
        if isinstance(expected.get("expected_sources"), list):
            expected_sources = [x for x in expected.get("expected_sources") if isinstance(x, str) and x.strip()]
        if not expected_sources and isinstance(expected.get("source_stem"), str) and expected.get("source_stem").strip():
            expected_sources = [expected.get("source_stem")]
        # 某些数据集将 Study/Subject UID 作为唯一可追溯来源标识（如 chest-xray-unidatapro 的 Gemini 输出），
        # 因此将 GT 的 expected_extra.subject_id 也视为可接受来源候选。
        expected_extra_obj = expected.get("expected_extra") if isinstance(expected.get("expected_extra"), dict) else {}
        expected_subject_id = expected_extra_obj.get("subject_id")
        if _field_nonmissing(expected_subject_id):
            expected_sources.append(str(expected_subject_id))
        expected_norm = {normalize_source_stem(x) for x in expected_sources if normalize_source_stem(x)}

        matched_ann_idx: str | None = None
        if expected_norm:
            source_candidates: list[str] = []
            for s in expected_norm:
                source_candidates.extend(ann_source_buckets.get(s, []))
            source_candidates = sorted(set(source_candidates), key=_idx_sort_key)
            for cand in source_candidates:
                if cand not in used_ann_indices:
                    matched_ann_idx = cand
                    break
        if matched_ann_idx is None and idx in ann_by_idx_all and idx not in used_ann_indices:
            matched_ann_idx = idx
        if matched_ann_idx is not None:
            used_ann_indices.add(matched_ann_idx)

        ann_path = ann_by_idx_all.get(matched_ann_idx) if matched_ann_idx else None
        image_idx = matched_ann_idx if matched_ann_idx is not None else idx
        actual_image_path = find_image_path(images_root, image_idx)
        image_exists = actual_image_path is not None
        image_valid = validate_image_file(actual_image_path)
        actual_relpath = image_relpath_from_path(actual_image_path)

        json_present = ann_path is not None
        json_valid = False
        path_consistent = False
        pair_valid = False
        source_score = 0.0
        schema_score = 0.0
        schema_ok = False
        schema_checks: dict[str, bool] = {}
        semantic_score = 0.0
        content_score = 0.0
        content_details: dict[str, Any] = {}
        iq_score = 0.0
        iq_details: dict[str, Any] = {}
        record_dataset_name = None
        sample_modality_primary = None
        media_task_type = None

        if json_present:
            n_json_present += 1
            ann = load_json(ann_path)
            if ann is not None:
                json_valid = True
                n_valid_json += 1

                extra = get_extra(ann)
                record = ann.get("record") if isinstance(ann.get("record"), dict) else {}
                sample = ((ann.get("context") or {}).get("sample") or {})
                tasks_obj = ann.get("tasks") if isinstance(ann.get("tasks"), dict) else {}
                media_obj = ((ann.get("media_geometry") or {}).get("media")) or {}

                record_dataset_name = record.get("dataset_name")
                sample_modality_primary = sample.get("modality_primary")
                media_task_type = media_obj.get("task_type")

                path_consistent = image_valid and bool(actual_relpath) and normalize_full_path(record.get("image_path")) == normalize_full_path(actual_relpath)
                pair_valid = image_valid and json_valid and path_consistent

                if expected_norm:
                    source_scored_n += 1
                    pred_source = derive_source_stem(ann, extra)
                    trace_ok = has_source_trace_evidence(extra, ann)
                    # v2: 在原派生 stem 之上，再扫描 extra/raw 中所有疑似源/路径字符串
                    # 拼成 alias 集合，与 expected 求交集，避免命名差异造成的误判 0 分。
                    pred_alias = derive_source_stem_aliases_v2(ann, extra)
                    expected_alias: set[str] = set()
                    for es in expected_norm:
                        expected_alias |= expand_source_aliases(es)
                    source_score = 1.0 if (pred_alias and expected_alias and bool(pred_alias & expected_alias) and trace_ok) else 0.0

                schema_score, schema_ok, schema_checks = check_schema_validity(ann, gt_format)
                semantic_score = compute_semantic_score(ann, expected, expected_task, expected_dim, content_spec)
                content_score, content_details = compute_content_fidelity(
                    ann,
                    expected,
                    content_spec,
                    agg_mode=content_agg,
                )
                iq_score, iq_details = compute_information_quality(extra, tasks_obj, media_obj, content_spec)

        if (not json_valid) and expected_norm:
            source_scored_n += 1

        parsed_samples.append(
            {
                "index": idx,
                "matched_annotation_index": matched_ann_idx,
                "image_exists": image_exists,
                "image_valid": image_valid,
                "json_present": json_present,
                "json_valid": json_valid,
                "path_consistent": path_consistent,
                "pair_valid": pair_valid,
                "source_score": source_score,
                "schema_valid": schema_ok,
                "schema_valid_score": schema_score,
                "schema_checks": schema_checks,
                "semantic_score": semantic_score,
                "information_quality_score": iq_score,
                "information_quality_details": iq_details,
                "content_fidelity_score": content_score,
                "content_fidelity_details": content_details,
                "record_dataset_name": record_dataset_name,
                "sample_modality_primary": sample_modality_primary,
                "media_task_type": media_task_type,
            }
        )

    if n_valid == 0:
        return {
            "model": model_name,
            "error": "GT 中无可评估样本",
            "dimensions": {},
            "weaknesses": ["GT 中无可评估样本"],
        }

    source_selection_accuracy = round((sum(x.get("source_score", 0.0) for x in parsed_samples) / source_scored_n), 4) if source_scored_n else 0.0

    image_coverage = sum(1 for x in parsed_samples if x.get("image_exists")) / n_valid
    image_valid_coverage = sum(1 for x in parsed_samples if x.get("image_valid")) / n_valid
    json_presence = n_json_present / n_valid
    json_valid_coverage = n_valid_json / n_valid
    pair_valid_coverage = sum(1 for x in parsed_samples if x.get("pair_valid")) / n_valid
    coverage = round(min(image_coverage, image_valid_coverage, json_presence, json_valid_coverage, pair_valid_coverage), 4)

    schema_validity = round(sum(x.get("schema_valid_score", 0.0) for x in parsed_samples) / n_valid, 4)
    source_pass_samples = [x for x in parsed_samples if float(x.get("source_score", 0.0)) > 0.0]
    n_source_pass = len(source_pass_samples)
    if n_source_pass > 0:
        semantic = round(sum(x.get("semantic_score", 0.0) for x in source_pass_samples) / n_source_pass, 4)
        content_fidelity = round(sum(x.get("content_fidelity_score", 0.0) for x in source_pass_samples) / n_source_pass, 4)
        information_quality = round(sum(x.get("information_quality_score", 0.0) for x in source_pass_samples) / n_source_pass, 4)
        # v2 patch6: 方案 A - 暴露 Information Quality 的 4 个独立子项
        def _avg_iq_sub(key: str) -> float:
            vals = [float(((x.get("information_quality_details") or {}).get(key) or 0.0)) for x in source_pass_samples]
            return round(sum(vals) / len(vals), 4) if vals else 0.0
        information_completeness = _avg_iq_sub("completeness")
        information_validity = _avg_iq_sub("validity")
        information_usefulness = _avg_iq_sub("usefulness")
        information_non_redundancy = _avg_iq_sub("non_redundancy")
    else:
        semantic = 0.0
        content_fidelity = 0.0
        information_quality = 0.0
        information_completeness = 0.0
        information_validity = 0.0
        information_usefulness = 0.0
        information_non_redundancy = 0.0

    data_meta_eval = evaluate_data_meta_quality(model_root, n_valid, ground_truth, parsed_samples)
    dataset_metadata_quality = round(float(data_meta_eval.get("score", 0.0)), 4)
    # v2 patch6: 方案 A - 暴露 Metadata Quality 的 3 个独立子项
    meta_schema_validity = round(float(data_meta_eval.get("meta_schema_validity", 0.0)), 4)
    meta_semantic_correctness = round(float(data_meta_eval.get("meta_semantic_correctness", 0.0)), 4)
    meta_content_fidelity = round(float(data_meta_eval.get("meta_content_fidelity", 0.0)), 4)

    # === 方案 A 严格 11 项 ===
    # 样本级 8 项（4-6/7 是 Information Quality 拆出来的 3 个独立子项；原拆出的
    #   "information_usefulness" 由于与 schema_validity.task_branch_required 和
    #   information_validity 存在显著重叠，按 ISO 25012 / DAMA-DMBOK 的最小冗余原则移除）：
    #   1. source_selection_accuracy
    #   2. coverage
    #   3. schema_validity
    #   4. semantic
    #   5. information_completeness    ← 原 IQ 拆 1/3
    #   6. information_validity         ← 原 IQ 拆 2/3
    #   7. information_non_redundancy   ← 原 IQ 拆 3/3
    #   8. content_fidelity
    # 数据集级 3 项（9-11 是 Dataset Metadata Quality 拆出来的 3 个独立子项）：
    #   9.  meta_schema_validity        ← 原 Meta 拆 1/3
    #   10. meta_semantic_correctness   ← 原 Meta 拆 2/3
    #   11. meta_content_fidelity       ← 原 Meta 拆 3/3
    dimensions = {
        "source_selection_accuracy": source_selection_accuracy,
        "coverage": coverage,
        "schema_validity": schema_validity,
        "semantic": semantic,
        "information_completeness": information_completeness,
        "information_validity": information_validity,
        "information_non_redundancy": information_non_redundancy,
        "content_fidelity": content_fidelity,
        "meta_schema_validity": meta_schema_validity,
        "meta_semantic_correctness": meta_semantic_correctness,
        "meta_content_fidelity": meta_content_fidelity,
    }
    # legacy composites + 已移除的 usefulness 子项（仅保留在 breakdown 里以便回溯/诊断）
    legacy_information_quality = information_quality
    legacy_information_usefulness = information_usefulness
    legacy_dataset_metadata_quality = dataset_metadata_quality

    weaknesses: list[str] = []
    # 方案 A 严格 12 项的 weakness 报告
    if source_selection_accuracy < 1.0:
        weaknesses.append(f"信息源选择准确性不足({source_selection_accuracy:.0%})")
    if coverage < 1.0:
        weaknesses.append(f"产出完整性不足({coverage:.0%})")
    if schema_validity < 1.0:
        weaknesses.append(f"结构合法性不足({schema_validity:.0%})")
    if semantic < 1.0:
        weaknesses.append(f"语义正确性不足({semantic:.0%})")
    if information_completeness < 1.0:
        weaknesses.append(f"信息完整性不足({information_completeness:.0%})")
    if information_validity < 1.0:
        weaknesses.append(f"信息有效性不足({information_validity:.0%})")
    if information_non_redundancy < 1.0:
        weaknesses.append(f"信息非冗余性不足({information_non_redundancy:.0%})")
    if content_fidelity < 1.0:
        weaknesses.append(f"内容一致性不足({content_fidelity:.0%})")
    if meta_schema_validity < 1.0:
        weaknesses.append(f"元信息结构合法性不足({meta_schema_validity:.0%})")
    if meta_semantic_correctness < 1.0:
        weaknesses.append(f"元信息语义正确性不足({meta_semantic_correctness:.0%})")
    if meta_content_fidelity < 1.0:
        weaknesses.append(f"元信息内容一致性不足({meta_content_fidelity:.0%})")
    if not weaknesses:
        weaknesses.append("无")

    return {
        "model": model_name,
        "n_expected": n_valid,
        "n_samples": n_valid_json,
        "per_sample": parsed_samples,
        "dimensions": dimensions,
        "breakdown": {
            "n_expected": n_valid,
            "n_img": sum(1 for x in parsed_samples if x.get("image_exists")),
            "n_img_valid": sum(1 for x in parsed_samples if x.get("image_valid")),
            "n_json_present": n_json_present,
            "n_json_valid": n_valid_json,
            "n_pair_valid": sum(1 for x in parsed_samples if x.get("pair_valid")),
            "source_scored": source_scored_n,
            "source_pass_for_content_metrics": n_source_pass,
            "source_correct_equivalent": round(sum(x.get("source_score", 0.0) for x in parsed_samples), 4),
            "expected_task_type": expected_task,
            "expected_dimension": expected_dim,
            "data_meta": data_meta_eval,
            # legacy（不在 dimensions 中，仅保留用于回溯/审稿复现）
            "legacy_information_quality": legacy_information_quality,
            "legacy_information_usefulness": legacy_information_usefulness,
            "legacy_dataset_metadata_quality": legacy_dataset_metadata_quality,
        },
        "weaknesses": weaknesses,
    }


def print_report(results: dict) -> None:
    models = results.get("models", {})
    valid = [(name, m) for name, m in models.items() if "error" not in m]
    errored = [(name, m) for name, m in models.items() if "error" in m]

    # 方案 A 严格 11 项
    dim_keys = [
        "source_selection_accuracy",
        "coverage",
        "schema_validity",
        "semantic",
        "information_completeness",
        "information_validity",
        "information_non_redundancy",
        "content_fidelity",
        "meta_schema_validity",
        "meta_semantic_correctness",
        "meta_content_fidelity",
    ]
    dim_names = [
        "信息源选择",
        "产出完整性",
        "结构合法性",
        "语义正确性",
        "信息完整性",
        "信息有效性",
        "信息非冗余",
        "内容一致性",
        "Meta结构",
        "Meta语义",
        "Meta内容",
    ]
    col_w = 8
    name_w = max(12, max((len(n) for n in models.keys()), default=12))
    total_w = name_w + len(dim_keys) * (col_w + 1) + 2

    print("\n" + "=" * total_w)
    print("  通用 VLM 评估（方案 A 严格 11 项）")
    print("=" * total_w)
    print(f"\n{'模型':<{name_w}}", end="")
    for h in dim_names:
        print(f" {h:<{col_w}}", end="")
    print()
    print("-" * total_w)

    for name, m in valid:
        dim = m.get("dimensions") or {}
        print(f"{name:<{name_w}}", end="")
        for k in dim_keys:
            print(f" {f'{dim.get(k, 0):.0%}':<{col_w}}", end="")
        print()

    for name, m in errored:
        print(f"{name:<{name_w}}  错误: {m.get('error', '')}")


def resolve_vlm_dir(dataset_dir: Path) -> Path:
    if dataset_dir.name == "VLM":
        return dataset_dir
    candidate = dataset_dir / "VLM"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"未找到 VLM 目录: {candidate}")


def infer_ground_truth(vlm_dir: Path) -> Path:
    cands = sorted(vlm_dir.glob("ground_truth*.json"))
    if not cands:
        raise FileNotFoundError(f"未找到 ground_truth*.json: {vlm_dir}")
    return cands[0]


def infer_models(vlm_dir: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(vlm_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if (p / "agent1_data_organization").is_dir():
            out.append(p.name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="third_dataset 通用评估（7 指标版）")
    parser.add_argument("--dataset-dir", type=str, required=True, help="数据集目录（可传 dataset 根目录或其 VLM 目录）")
    parser.add_argument("--ground-truth", type=str, default=None, help="GT JSON，默认自动推断")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="模型目录名列表，默认自动发现")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径，默认写到 VLM/eval_results_v2.json")
    parser.add_argument(
        "--content-agg",
        type=str,
        choices=["arithmetic", "geometric"],
        default="arithmetic",
        help="内容一致性聚合方式：arithmetic(默认，非一票否决) 或 geometric(严格，一项为0则整体归0)",
    )
    parser.add_argument("--no-print", action="store_true", help="不打印报告")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    vlm_dir = resolve_vlm_dir(dataset_dir)
    gt_path = Path(args.ground_truth) if args.ground_truth else infer_ground_truth(vlm_dir)
    if not gt_path.is_file():
        print(f"Ground truth 文件不存在: {gt_path}")
        return 1

    ground_truth = load_json(gt_path)
    if not ground_truth:
        print("Ground truth 无法解析")
        return 1

    models = args.models or infer_models(vlm_dir)
    if not models:
        print(f"未发现可评估模型目录: {vlm_dir}")
        return 1

    results: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "vlm_dir": str(vlm_dir),
        "ground_truth_file": str(gt_path),
        "content_agg": args.content_agg,
        "models": {},
    }

    for model_name in models:
        model_root = vlm_dir / model_name
        if not model_root.is_dir():
            results["models"][model_name] = {"error": f"目录不存在: {model_root}"}
            continue
        results["models"][model_name] = evaluate_model(
            model_name,
            model_root,
            ground_truth,
            content_agg=args.content_agg,
        )

    out_path = Path(args.output) if args.output else (vlm_dir / "eval_results_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已写入: {out_path}")

    if not args.no_print:
        print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())