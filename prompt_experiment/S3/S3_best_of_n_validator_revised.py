#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3 Best-of-n + Validator for medical data standardization.

Purpose
-------
Use several independently generated S0-style candidate outputs and select the
best candidate JSON for each sample without reading ground_truth*.json or the
final evaluation results. The validator relies only on:
  1) the output schema implied by the data-organization prompt,
  2) the produced files under each candidate root,
  3) the original random.txt/source-list when available.

Expected candidate layouts
--------------------------
Each candidate may be either an agent1_data_organization directory itself or a
parent directory that contains agent1_data_organization:

  run_0/agent1_data_organization/images/0.png
  run_0/agent1_data_organization/standardized_annotations/0.json
  run_0/agent1_data_organization/data_meta.json

Default input root:
  {dataset_root}/prompt_experiment/candidates

Default output root:
  {dataset_root}/prompt_experiment/S3/agent1_data_organization

Example
-------
python3 s3_best_of_n_validator.py \
  --dataset-root /path/to/dataset \
  --runs-root /path/to/dataset/prompt_experiment/candidates
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image, UnidentifiedImageError
except Exception:  # pragma: no cover
    Image = None
    UnidentifiedImageError = Exception

ALLOWED_TASKS = {"classification", "segmentation", "detection"}
ALLOWED_DIMS = {"2d", "3d", "video"}
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
MISSING_TOKENS = {"", "na", "n/a", "null", "none", "unknown", "unk", "nan", "tbd", "not available", "not_applicable"}
SOURCE_KEY_RE = re.compile(
    r"(source|original|raw|file|path|filename|subject|patient|case|sample|"
    r"study|series|sop|dicom|dcm|nifti|nii|mask|label|seg|volume|"
    r"scan|slice|frame|cohort|split|folder|dir|stem|uid|mat|npz|npy|h5)",
    re.I,
)
DESC_OPTIONS_RE = re.compile(r"(__desc|__options)$")


def norm_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def norm_token(v: Any) -> str:
    s = norm_text(v).replace("\\", "/")
    s = re.sub(r"[\s\-_/]+", "_", s)
    s = re.sub(r"[^a-z0-9_\u4e00-\u9fff]+", "", s)
    return s.strip("_")


def norm_dim(v: Any) -> str:
    s = norm_text(v).replace("-", "").replace(" ", "")
    if s.startswith("2d"):
        return "2d"
    if s.startswith("3d") or s == "threed":
        return "3d"
    if "video" in s:
        return "video"
    return s


def norm_task(v: Any) -> str:
    s = norm_text(v)
    return s if s in ALLOWED_TASKS else s


def norm_path(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().replace("\\", "/")
    s = re.sub(r"/+", "/", s).strip("/")
    return s.lower()


def is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return norm_text(v) in MISSING_TOKENS or bool(re.fullmatch(r"<[^<>]+>", v.strip()))
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        try:
            return not math.isfinite(float(v))
        except Exception:
            return True
    if isinstance(v, (list, tuple, set)):
        return len(v) == 0 or all(is_missing(x) for x in v)
    if isinstance(v, dict):
        return len(v) == 0 or all(is_missing(x) for x in v.values())
    return False


def nonmissing(v: Any) -> bool:
    return not is_missing(v)


def branch_empty(v: Any) -> bool:
    return is_missing(v)


def load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def valid_image(path: Path | None) -> bool:
    if path is None or not path.is_file() or path.stat().st_size <= 0:
        return False
    if Image is None:
        return True
    try:
        Image.MAX_IMAGE_PIXELS = None  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im2:
            w, h = im2.size
        return w > 0 and h > 0
    except (UnidentifiedImageError, OSError, ValueError, Exception):
        return False


def image_by_index(org_root: Path, idx: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = org_root / "images" / f"{idx}{ext}"
        if p.is_file():
            return p
    return None


def image_from_record(org_root: Path, record_image_path: Any) -> Path | None:
    rel = norm_path(record_image_path)
    if not rel:
        return None
    p = org_root / rel
    return p if p.is_file() else None


def has_desc_options(obj: Any) -> bool:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and DESC_OPTIONS_RE.search(k):
                return True
            if has_desc_options(v):
                return True
    elif isinstance(obj, list):
        return any(has_desc_options(x) for x in obj)
    return False


def deep_get(obj: dict, path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def deep_set(obj: dict, path: str, value: Any) -> None:
    cur = obj
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def normalize_source_stem(v: Any) -> str | None:
    """Normalize a source identifier/path into the same coarse stem used by the validator.

    This intentionally mirrors the main strict evaluator for common medical-data
    naming patterns, but it remains non-oracle: it only normalizes fields already
    present in candidates or random.txt.
    """
    if v is None:
        return None
    raw = str(v).strip().strip("'\"").replace("\\", "/")
    if not raw:
        return None

    # medical-7-2020 style: radial_128_2/10/3DMR_Chest.mat
    m = re.search(r"(radial_\d+_\d+)/(\d+)/([^/]+?)(?:\.[a-z0-9.]+)?$", raw, flags=re.I)
    if m:
        cfg, spokes, name = m.group(1), m.group(2), m.group(3)
        return norm_token(f"{cfg}_{spokes}_{name}") or None

    # medical-7-2020 style: 3DMR_Chest_radial_128_2_spokes10
    m = re.match(r"(.+?)_(radial_\d+_\d+)_(?:spokes)?(\d+)$", raw, flags=re.I)
    if m:
        name, cfg, spokes = m.group(1), m.group(2), m.group(3)
        return norm_token(f"{cfg}_{spokes}_{name}") or None

    # microscopy-demo style: dynamicnuclearnet/test_1/frame_0000.tif
    m = re.search(r"(dynamicnuclearnet)[/_](test_\d+)[/_](frame[_-]?\d+)", raw, flags=re.I)
    if m:
        ds, seq, frame = m.group(1), m.group(2), m.group(3).replace("-", "_")
        return norm_token(f"{ds}_{seq}_{frame}") or None

    base = raw.rstrip("/").split("/")[-1]

    # microUS_test_01.nii and test_01 denote the same source unless a frame is present.
    m = re.search(r"(test_\d+)", base, flags=re.I)
    if m and not re.search(r"frame[_-]?\d+", base, flags=re.I):
        return m.group(1).lower()

    for ext in [
        ".ome.tiff", ".ome.tif", ".nii.gz", ".nii", ".mha", ".mhd", ".nrrd",
        ".dcm", ".dicom", ".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp",
        ".gif", ".h5", ".hdf5", ".npz", ".npy", ".mat", ".json", ".txt",
        ".csv", ".avi", ".mp4", ".mov", ".webm", ".webp", ".pdf",
    ]:
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break

    stem = norm_token(base)
    m = re.match(r"(brain_p\d+)_0000$", stem)
    if m:
        stem = m.group(1)
    for suf in ["_flair", "_t1ce", "_t1", "_t2", "_seg", "_mask"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem or None


def extract_source_from_notes(notes: Any) -> str | None:
    if not isinstance(notes, str):
        return None
    txt = notes.strip()
    if not txt:
        return None
    if txt.lower().startswith("converted from:"):
        return txt.split(":", 1)[1].strip()
    m = re.match(r"converted\s+from\s+(\S+?)(?:\.\s|,\s|$)", txt, flags=re.I)
    if m:
        return m.group(1).strip()
    m = re.search(
        r"(?:original\s*(?:file|filename)|source\s*file)\s*:\s*([^,\n]+?)(?:\s*,\s+[A-Z][A-Za-z_ ]*\s*:|$)",
        txt,
        flags=re.I,
    )
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:original\s*filename|source\s*file)\s*:\s*(.+)$", txt, flags=re.I)
    if m:
        return m.group(1).strip()
    return None


def source_aliases(v: Any) -> set[str]:
    stem = normalize_source_stem(v)
    if not stem:
        return set()
    out = {stem}
    # Common BIDS/patient spelling variants.
    m = re.fullmatch(r"sub_?([a-z]+)0*(\d+)(?:_dti)?", stem)
    if m:
        pref, n = m.group(1), int(m.group(2))
        out |= {f"{pref}{n}", f"{pref}{n:02d}", f"sub_{pref}{n}", f"sub_{pref}{n:02d}", f"sub_{pref}{n}_dti", f"sub_{pref}{n:02d}_dti"}
    m = re.fullmatch(r"([a-z]+)0*(\d+)", stem)
    if m:
        pref, n = m.group(1), int(m.group(2))
        out |= {f"{pref}{n}", f"{pref}{n:02d}", f"sub_{pref}{n}", f"sub_{pref}{n:02d}"}
    return out


def iter_source_like_values(obj: Any, parent_key: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{parent_key}.{k}" if parent_key else str(k)
            if isinstance(v, (dict, list)):
                yield from iter_source_like_values(v, key)
            elif isinstance(v, str):
                note_source = extract_source_from_notes(v) if key.endswith(".notes") or key == "notes" else None
                if note_source:
                    yield key, note_source
                if SOURCE_KEY_RE.search(key) and v.strip():
                    yield key, v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_source_like_values(v, f"{parent_key}.{i}")


def read_random_txt(dataset_root: Path, explicit: str | None = None) -> dict[str, str]:
    path = Path(explicit) if explicit else dataset_root / "VLM" / "random.txt"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\d+)\s+(.+)$", line)
        if m:
            idx, p = m.group(1), m.group(2).strip()
        else:
            idx, p = str(line_no), line
        p = p.replace("{dataset_root}", str(dataset_root))
        out[str(idx)] = p
    return out


def resolve_org_root(path: Path) -> Path:
    p = Path(path)
    if (p / "agent1_data_organization").is_dir():
        return p / "agent1_data_organization"
    return p


def is_candidate_org_root(path: Path) -> bool:
    p = resolve_org_root(path)
    return (p / "standardized_annotations").is_dir() or (p / "images").is_dir() or (p / "data_meta.json").is_file()


def discover_candidate_roots(runs_root: Path, explicit: list[str] | None = None) -> list[Path]:
    roots: list[Path] = []
    if explicit:
        for x in explicit:
            p = resolve_org_root(Path(x))
            if is_candidate_org_root(p):
                roots.append(p)
        return roots
    rr = Path(runs_root)
    if is_candidate_org_root(rr):
        return [resolve_org_root(rr)]
    if not rr.is_dir():
        return []
    for child in sorted(rr.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        p = resolve_org_root(child)
        if is_candidate_org_root(p):
            roots.append(p)
    return roots


def annotation_indices(roots: list[Path], source_map: dict[str, str], include_extra_candidates: bool = False) -> list[str]:
    # When random.txt/source-map exists, it is the sample universe. Extra candidate
    # JSON files are usually stale or hallucinated outputs and should not inflate
    # data_meta.total. They can be included explicitly for debugging.
    ids = set(source_map.keys())
    if include_extra_candidates or not source_map:
        for root in roots:
            ann_dir = root / "standardized_annotations"
            if ann_dir.is_dir():
                for p in ann_dir.glob("*.json"):
                    ids.add(p.stem)
    def key(s: str):
        return (0, int(s)) if str(s).isdigit() else (1, str(s))
    return sorted(ids, key=key)


def default_runs_root(dataset_root: Path, strategy: str = "S3") -> Path:
    candidates = [
        dataset_root / "prompt_experiment" / strategy / "candidates",
        dataset_root / "prompt_experiment" / "candidates",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


def class_id_set(classes: Any) -> set[str]:
    if isinstance(classes, dict):
        return {str(k) for k in classes.keys()}
    if isinstance(classes, list):
        return {str(i) for i in range(len(classes))}
    return set()


def classes_nonempty(classes: Any) -> bool:
    if isinstance(classes, dict):
        return any(nonmissing(k) and nonmissing(v) for k, v in classes.items())
    if isinstance(classes, list):
        return any(nonmissing(x) for x in classes)
    return False


def label_in_classes(label: Any, classes: Any) -> bool:
    ids = class_id_set(classes)
    if not ids or label is None:
        return False
    if str(label) in ids:
        return True
    try:
        return str(int(label)) in ids
    except Exception:
        return False


def collect_category_ids(obj: Any) -> list[Any]:
    out: list[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = norm_text(k)
            if lk in {"category_id", "class_id", "label", "label_id"} and not isinstance(v, (dict, list)):
                out.append(v)
            else:
                out.extend(collect_category_ids(v))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(collect_category_ids(x))
    return out


def task_payload_score(task_type: str, tasks: dict) -> tuple[float, dict[str, Any]]:
    branch = tasks.get(task_type) if isinstance(tasks.get(task_type), dict) else {}
    detail: dict[str, Any] = {"active_branch_present": bool(branch)}
    if not branch:
        return 0.0, detail
    parts: list[float] = []
    parts.append(1.0 if nonmissing(branch.get("schema_variant")) else 0.0)
    if task_type == "classification":
        classes = branch.get("class_names") if branch.get("class_names") is not None else branch.get("classes")
        label = branch.get("label")
        cls_ok = classes_nonempty(classes)
        label_ok = nonmissing(label)
        in_map = label_in_classes(label, classes)
        label_name_present = nonmissing(branch.get("label_name"))
        name_ok = label_name_present or in_map
        parts.extend([1.0 if cls_ok else 0.0, 1.0 if label_ok else 0.0, 1.0 if in_map else 0.0,
                       1.0 if name_ok else 0.0, 1.0 if label_name_present else 0.0])
        detail.update({"classes_nonempty": cls_ok, "label_present": label_ok, "label_in_classes": in_map,
                        "label_name_present": label_name_present})
    elif task_type == "detection":
        classes = branch.get("class_names") if branch.get("class_names") is not None else branch.get("classes")
        cls_ok = classes_nonempty(classes)
        boxes = branch.get("boxes2d") or branch.get("boxes3d") or branch.get("frame_boxes2d")
        box_ok = nonmissing(boxes)
        ids = collect_category_ids(boxes)
        id_ok = True if not ids else all(label_in_classes(x, classes) for x in ids)
        parts.extend([1.0 if cls_ok else 0.0, 1.0 if box_ok else 0.0, 1.0 if id_ok else 0.0])
        detail.update({"classes_nonempty": cls_ok, "boxes_present": box_ok, "category_ids_valid": id_ok})
    elif task_type == "segmentation":
        classes = branch.get("classes") if branch.get("classes") is not None else branch.get("class_names")
        cls_ok = classes_nonempty(classes)
        payload = branch.get("coco_2d") or branch.get("mask_3d") or branch.get("video_coco")
        payload_ok = nonmissing(payload)
        ids = collect_category_ids(payload)
        id_ok = True if not ids else all(label_in_classes(x, classes) for x in ids)
        parts.extend([1.0 if cls_ok else 0.0, 1.0 if payload_ok else 0.0, 1.0 if id_ok else 0.0])
        detail.update({"classes_nonempty": cls_ok, "mask_or_coco_present": payload_ok, "category_ids_valid": id_ok})
    else:
        return 0.0, detail
    inactive_ok = True
    for t in ALLOWED_TASKS:
        if t != task_type and not branch_empty(tasks.get(t)):
            inactive_ok = False
            break
    parts.append(1.0 if inactive_ok else 0.0)
    detail["inactive_task_branches_empty"] = inactive_ok
    return round(sum(parts) / len(parts), 4), detail


def spatial_score(dim: str, spatial: dict) -> tuple[float, dict[str, Any]]:
    branch_name = {"2d": "common", "3d": "three_d", "video": "video"}.get(dim, "")
    detail = {"selected_branch": branch_name}
    if not branch_name:
        detail["selected_nonempty"] = False
        detail["inactive_empty"] = False
        return 0.0, detail
    selected_ok = isinstance(spatial.get(branch_name), dict) and not branch_empty(spatial.get(branch_name))
    inactive_ok = True
    for b in ["common", "three_d", "video"]:
        if b != branch_name and not branch_empty(spatial.get(b)):
            inactive_ok = False
    detail["selected_nonempty"] = selected_ok
    detail["inactive_empty"] = inactive_ok
    return round((float(selected_ok) + float(inactive_ok)) / 2.0, 4), detail


def source_trace_score(ann: dict, idx: str, source_map: dict[str, str]) -> tuple[float, dict[str, Any]]:
    expected = source_map.get(str(idx))
    values = list(iter_source_like_values(ann))
    has_trace = any(nonmissing(v) for _, v in values)
    detail = {"has_source_like_field": has_trace, "matched_random_txt": None, "matched_field": None}
    if not expected:
        return (0.7 if has_trace else 0.0), detail
    exp_alias = source_aliases(expected)
    for key, value in values:
        val_alias = source_aliases(value)
        if val_alias & exp_alias:
            detail["matched_random_txt"] = True
            detail["matched_field"] = key
            return 1.0, detail
        # Also allow literal containment after coarse normalization.
        if norm_token(expected) and norm_token(value) and (norm_token(expected) in norm_token(value) or norm_token(value) in norm_token(expected)):
            detail["matched_random_txt"] = True
            detail["matched_field"] = key
            return 1.0, detail
    detail["matched_random_txt"] = False
    return (0.35 if has_trace else 0.0), detail


def score_annotation_object(ann: Any, org_root: Path, idx: str, source_map: dict[str, str]) -> tuple[float, dict[str, Any], Path | None]:
    checks: dict[str, Any] = {}
    if not isinstance(ann, dict):
        return 0.0, {"valid_json_object": False}, None
    checks["valid_json_object"] = True

    record = ann.get("record") if isinstance(ann.get("record"), dict) else {}
    context = ann.get("context") if isinstance(ann.get("context"), dict) else {}
    sample = context.get("sample") if isinstance(context.get("sample"), dict) else {}
    media_geometry = ann.get("media_geometry") if isinstance(ann.get("media_geometry"), dict) else {}
    media = media_geometry.get("media") if isinstance(media_geometry.get("media"), dict) else {}
    spatial = media_geometry.get("spatial") if isinstance(media_geometry.get("spatial"), dict) else {}
    tasks = ann.get("tasks") if isinstance(ann.get("tasks"), dict) else {}

    top_blocks = all(isinstance(ann.get(k), dict) for k in ["record", "context", "media_geometry", "tasks"])
    checks["top_level_blocks"] = top_blocks

    rec_parts = [
        nonmissing(record.get("schema_version")),
        nonmissing(record.get("dataset_name")),
        isinstance(record.get("image_path"), str) and norm_path(record.get("image_path")).startswith("images/"),
    ]
    record_score = sum(map(float, rec_parts)) / len(rec_parts)
    checks["record"] = {"score": round(record_score, 4), "parts": rec_parts}

    record_img = image_from_record(org_root, record.get("image_path"))
    index_img = image_by_index(org_root, idx)
    chosen_img = record_img if valid_image(record_img) else index_img
    record_path_expected = f"images/{Path(record.get('image_path', '')).name}" if record.get("image_path") else ""
    path_consistent = bool(record_img and index_img and record_img.resolve() == index_img.resolve()) or norm_path(record.get("image_path")) == norm_path(f"images/{idx}{Path(record.get('image_path', '')).suffix}")
    img_valid = valid_image(record_img) or valid_image(index_img)
    path_score = 0.0
    if valid_image(record_img) and path_consistent:
        path_score = 1.0
    elif valid_image(record_img):
        path_score = 0.8
    elif valid_image(index_img):
        path_score = 0.55
    checks["image_path"] = {"score": path_score, "record_image_valid": valid_image(record_img), "index_image_valid": valid_image(index_img), "path_consistent": path_consistent, "record_path_expected_hint": record_path_expected}

    task = norm_task(media.get("task_type"))
    dim = norm_dim(media.get("dimension"))
    conf = media.get("confidence_global")
    conf_ok = True
    if conf is not None:
        try:
            conf_ok = 0.0 <= float(conf) <= 1.0
        except Exception:
            conf_ok = False
    media_parts = [task in ALLOWED_TASKS, dim in ALLOWED_DIMS, conf_ok]
    media_score = sum(map(float, media_parts)) / len(media_parts)
    checks["media"] = {"score": round(media_score, 4), "task_type": task, "dimension": dim, "confidence_ok": conf_ok}

    sp_score, sp_detail = spatial_score(dim, spatial)
    checks["spatial"] = {"score": sp_score, **sp_detail}

    tp_score, tp_detail = task_payload_score(task, tasks)
    checks["task_payload"] = {"score": tp_score, **tp_detail}

    context_parts = [
        isinstance(sample.get("extra"), dict),
        nonmissing(sample.get("modality_primary")),
        nonmissing(sample.get("anatomical_structure")),
    ]
    context_score = sum(map(float, context_parts)) / len(context_parts)
    checks["context_sample"] = {"score": round(context_score, 4), "parts": context_parts}

    src_score, src_detail = source_trace_score(ann, idx, source_map)
    checks["source_trace"] = {"score": src_score, **src_detail}

    no_desc = not has_desc_options(ann)
    checks["no_desc_options"] = no_desc

    # extra completeness: source_file, label_text, class_name, original_filename
    extra = sample.get("extra") if isinstance(sample.get("extra"), dict) else {}
    extra_expected = ["source_file", "original_filename", "label_text", "class_name"]
    extra_hits = sum(1.0 for k in extra_expected if nonmissing(extra.get(k)))
    extra_completeness = extra_hits / len(extra_expected)
    checks["extra_completeness"] = {"score": round(extra_completeness, 4),
                                     "present": [k for k in extra_expected if nonmissing(extra.get(k))],
                                     "missing": [k for k in extra_expected if not nonmissing(extra.get(k))]}

    # annotation_type consistency with schema_variant
    active_branch = tasks.get(task) if isinstance(tasks.get(task), dict) else {}
    schema_variant = active_branch.get("schema_variant")
    ann_type = media.get("annotation_type")
    ann_type_ok = (nonmissing(ann_type) and nonmissing(schema_variant)
                   and norm_text(ann_type) == norm_text(schema_variant))
    checks["annotation_type_consistency"] = ann_type_ok

    critical_values = [
        record.get("dataset_name"), sample.get("modality_primary"), sample.get("anatomical_structure"),
        media.get("task_type"), media.get("dimension"), tasks.get(task) if isinstance(tasks, dict) else None,
    ]
    nonempty_ratio = sum(1 for x in critical_values if nonmissing(x)) / len(critical_values)
    checks["critical_nonempty_ratio"] = round(nonempty_ratio, 4)

    weights = {
        "top_level_blocks": 0.06,
        "record": 0.08,
        "image_path": 0.10,
        "media": 0.08,
        "spatial": 0.06,
        "task_payload": 0.16,
        "context_sample": 0.06,
        "source_trace": 0.08,
        "no_desc_options": 0.02,
        "critical_nonempty_ratio": 0.02,
        "extra_completeness": 0.18,
        "annotation_type_consistency": 0.10,
    }
    score = (
        weights["top_level_blocks"] * float(top_blocks)
        + weights["record"] * record_score
        + weights["image_path"] * path_score
        + weights["media"] * media_score
        + weights["spatial"] * sp_score
        + weights["task_payload"] * tp_score
        + weights["context_sample"] * context_score
        + weights["source_trace"] * src_score
        + weights["no_desc_options"] * float(no_desc)
        + weights["critical_nonempty_ratio"] * nonempty_ratio
        + weights["extra_completeness"] * extra_completeness
        + weights["annotation_type_consistency"] * float(ann_type_ok)
    )
    return round(score, 6), checks, chosen_img if valid_image(chosen_img) else None


@dataclass
class CandidateSample:
    candidate_name: str
    org_root: Path
    idx: str
    ann_path: Path | None
    ann: dict | None
    score: float
    checks: dict[str, Any]
    image_path: Path | None
    json_error: str | None = None


def score_candidate_sample(org_root: Path, idx: str, source_map: dict[str, str]) -> CandidateSample:
    ann_path = org_root / "standardized_annotations" / f"{idx}.json"
    if not ann_path.is_file():
        return CandidateSample(org_root.parent.name if org_root.name == "agent1_data_organization" else org_root.name, org_root, idx, None, None, 0.0, {"missing_annotation": True}, image_by_index(org_root, idx), None)
    obj, err = load_json(ann_path)
    if err or not isinstance(obj, dict):
        return CandidateSample(org_root.parent.name if org_root.name == "agent1_data_organization" else org_root.name, org_root, idx, ann_path, None, 0.0, {"json_error": err or "top-level is not object"}, image_by_index(org_root, idx), err)
    score, checks, img = score_annotation_object(obj, org_root, idx, source_map)
    return CandidateSample(org_root.parent.name if org_root.name == "agent1_data_organization" else org_root.name, org_root, idx, ann_path, obj, score, checks, img, None)


def choose_best(cands: list[CandidateSample]) -> CandidateSample | None:
    valid = [c for c in cands if c.ann is not None]
    if not valid:
        return None
    # Stable tie-break: score, valid image, valid path score, candidate name.
    def key(c: CandidateSample):
        path_score = 0.0
        try:
            path_score = float(c.checks.get("image_path", {}).get("score", 0.0))
        except Exception:
            pass
        return (c.score, 1.0 if valid_image(c.image_path) else 0.0, path_score, -len(c.candidate_name))
    return sorted(valid, key=key, reverse=True)[0]


def ensure_record_image_path(ann: dict, idx: str, image_path: Path | None) -> None:
    if not isinstance(ann.get("record"), dict):
        ann["record"] = {}
    ext = image_path.suffix.lower() if image_path is not None and image_path.suffix else ".png"
    ann["record"]["image_path"] = f"images/{idx}{ext}"


def copy_image_to_output(src: Path | None, out_img_dir: Path, idx: str) -> Path | None:
    if src is None or not src.is_file():
        return None
    out_img_dir.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTS else ".png"
    dst = out_img_dir / f"{idx}{ext}"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def list_valid_images_for_idx(cands: list[CandidateSample]) -> list[Path]:
    out = []
    seen = set()
    for c in cands:
        for p in [c.image_path, image_by_index(c.org_root, c.idx)]:
            if p and valid_image(p) and str(p.resolve()) not in seen:
                out.append(p)
                seen.add(str(p.resolve()))
    return out


def sample_summary(ann: dict) -> dict[str, Any]:
    record = ann.get("record") if isinstance(ann.get("record"), dict) else {}
    sample = deep_get(ann, "context.sample", {}) or {}
    media = deep_get(ann, "media_geometry.media", {}) or {}
    tasks = ann.get("tasks") if isinstance(ann.get("tasks"), dict) else {}
    task = norm_task(media.get("task_type"))
    branch = tasks.get(task) if isinstance(tasks.get(task), dict) else {}
    classes = branch.get("class_names") if branch.get("class_names") is not None else branch.get("classes")
    return {
        "dataset_name": record.get("dataset_name"),
        "modality_primary": sample.get("modality_primary"),
        "modality_secondary": sample.get("modality_secondary"),
        "anatomical_structure": sample.get("anatomical_structure"),
        "task_type": task,
        "classes": classes,
    }


def majority_nonmissing(values: list[Any]) -> Any:
    vals = [v for v in values if nonmissing(v)]
    if not vals:
        return None
    keys = [json.dumps(v, ensure_ascii=False, sort_keys=True) for v in vals]
    best_key = Counter(keys).most_common(1)[0][0]
    return json.loads(best_key)


def count_classes(classes: Any) -> int:
    if isinstance(classes, dict):
        return len([k for k, v in classes.items() if nonmissing(k) and nonmissing(v)])
    if isinstance(classes, list):
        return len([x for x in classes if nonmissing(x)])
    return 0


def build_minimal_meta(selected_annotations: list[dict], valid_image_total: int) -> dict[str, Any]:
    summaries = [sample_summary(a) for a in selected_annotations]
    tasks = sorted({s["task_type"] for s in summaries if s.get("task_type") in ALLOWED_TASKS})
    dataset_name = majority_nonmissing([s.get("dataset_name") for s in summaries]) or "unknown"
    mod_primary = majority_nonmissing([s.get("modality_primary") for s in summaries])
    mod_secondary = majority_nonmissing([s.get("modality_secondary") for s in summaries])
    anatomy = majority_nonmissing([s.get("anatomical_structure") for s in summaries])
    meta = {
        "schema_version": "1.0.2",
        "dataset_name": dataset_name,
        "release_date": "NA",
        "homepage_url": "NA",
        "organization": ["NA"],
        "challenge_series": "NA",
        "license": "NA",
        "dataset_description": "NA",
        "modality_primary": mod_primary if isinstance(mod_primary, list) else ([mod_primary] if nonmissing(mod_primary) else ["NA"]),
        "modality_secondary": mod_secondary if isinstance(mod_secondary, list) else ([mod_secondary] if nonmissing(mod_secondary) else ["NA"]),
        "anatomical_structure": anatomy if isinstance(anatomy, list) else ([anatomy] if nonmissing(anatomy) else ["NA"]),
        "disease": ["NA"],
        "domain": "medical",
        "data_volume": {"total": len(selected_annotations), "train": 0, "val": 0, "test": 0, "label": len(selected_annotations)},
        "valid_image_n": {"total": valid_image_total},
        "label_presence": "labeled",
        "task_type": tasks,
        "num_classes_per_task": {},
    }
    for t in tasks:
        class_counts = [count_classes(s.get("classes")) for s in summaries if s.get("task_type") == t]
        n_cls = max(class_counts) if class_counts else 0
        if t == "classification":
            meta["num_classes_per_task"][t] = {"num_classes": n_cls}
        elif t == "detection":
            meta["num_classes_per_task"][t] = {"type": "single-class" if n_cls <= 1 else "multi-class", "num_classes": n_cls}
        elif t == "segmentation":
            meta["num_classes_per_task"][t] = {"num_classes": n_cls, "include_background": False}
    return meta


def meta_score(meta: Any, selected_annotations: list[dict], valid_image_total: int) -> float:
    if not isinstance(meta, dict):
        return 0.0
    summaries = [sample_summary(a) for a in selected_annotations]
    expected_total = len(selected_annotations)
    tasks = {s.get("task_type") for s in summaries if s.get("task_type") in ALLOWED_TASKS}
    parts = []
    parts.append(1.0 if nonmissing(meta.get("dataset_name")) else 0.0)
    parts.append(1.0 if nonmissing(meta.get("modality_primary")) else 0.0)
    dv = meta.get("data_volume") if isinstance(meta.get("data_volume"), dict) else {}
    parts.append(1.0 if int_like(dv.get("total")) == expected_total else 0.0)
    vin = meta.get("valid_image_n") if isinstance(meta.get("valid_image_n"), dict) else {}
    parts.append(1.0 if int_like(vin.get("total")) == valid_image_total else 0.0)
    mt = meta.get("task_type")
    mt_set = {norm_task(x) for x in mt} if isinstance(mt, list) else ({norm_task(mt)} if mt is not None else set())
    parts.append(1.0 if tasks and tasks.issubset(mt_set) else 0.0)
    parts.append(1.0 if isinstance(meta.get("num_classes_per_task"), dict) else 0.0)
    return sum(parts) / len(parts)


def int_like(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def choose_or_build_meta(candidate_roots: list[Path], selected_annotations: list[dict], valid_image_total: int, mode: str) -> tuple[dict, dict[str, Any]]:
    if mode == "rebuild":
        return build_minimal_meta(selected_annotations, valid_image_total), {"mode": "rebuild"}
    scored = []
    for root in candidate_roots:
        obj, err = load_json(root / "data_meta.json")
        if err or not isinstance(obj, dict):
            continue
        scored.append((meta_score(obj, selected_annotations, valid_image_total), root, obj))
    if not scored:
        return build_minimal_meta(selected_annotations, valid_image_total), {"mode": "rebuild_no_valid_candidate_meta"}
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_root, meta = scored[0]
    meta = copy.deepcopy(meta)
    report = {"mode": "best_candidate" if mode == "best" else "best_then_patch", "selected_meta_from": str(best_root), "pre_patch_score": round(best_score, 4)}
    if mode == "best_then_patch":
        if not isinstance(meta.get("data_volume"), dict):
            meta["data_volume"] = {}
        meta["data_volume"]["total"] = len(selected_annotations)
        if "label" not in meta["data_volume"] or not isinstance(meta["data_volume"].get("label"), int):
            meta["data_volume"]["label"] = len(selected_annotations)
        if not isinstance(meta.get("valid_image_n"), dict):
            meta["valid_image_n"] = {}
        meta["valid_image_n"]["total"] = valid_image_total
        tasks = sorted({sample_summary(a).get("task_type") for a in selected_annotations if sample_summary(a).get("task_type") in ALLOWED_TASKS})
        if tasks:
            meta["task_type"] = tasks
    return meta, report


def prepare_output_root(path: Path, overwrite: bool) -> None:
    # Avoid stale images/JSON from previous runs contaminating coverage or metadata.
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            for sub in ["images", "standardized_annotations"]:
                sub_path = path / sub
                if sub_path.exists():
                    shutil.rmtree(sub_path)
            for fname in ["data_meta.json", "s3_validator_report.json", "s4_voting_report.json"]:
                f = path / fname
                if f.is_file():
                    f.unlink()
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "standardized_annotations").mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="S3 Best-of-n + automatic validator selector")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing data/ and usually VLM/random.txt")
    parser.add_argument("--runs-root", default=None, help="Directory containing candidate runs. Default: {dataset_root}/prompt_experiment/candidates")
    parser.add_argument("--candidate-roots", nargs="*", default=None, help="Explicit candidate directories; each may be agent1_data_organization or its parent")
    parser.add_argument("--random-file", default=None, help="Optional random.txt/source list path")
    parser.add_argument("--output", default=None, help="Output agent1_data_organization directory. Default: {dataset_root}/prompt_experiment/S3/agent1_data_organization")
    parser.add_argument("--meta-mode", choices=["best", "best_then_patch", "rebuild"], default="best_then_patch")
    parser.add_argument("--overwrite", action="store_true", help="Remove the whole output directory before writing. By default, managed output subdirectories are cleaned.")
    parser.add_argument("--include-extra-candidate-indices", action="store_true", help="Include candidate JSON indices not listed in random.txt/source list")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    runs_root = Path(args.runs_root).resolve() if args.runs_root else default_runs_root(dataset_root, "S3")
    out_root = Path(args.output).resolve() if args.output else dataset_root / "prompt_experiment" / "S3" / "agent1_data_organization"

    roots = discover_candidate_roots(runs_root, args.candidate_roots)
    if not roots:
        raise SystemExit(f"No candidate roots found under {runs_root}")
    source_map = read_random_txt(dataset_root, args.random_file)
    indices = annotation_indices(roots, source_map, include_extra_candidates=args.include_extra_candidate_indices)
    if not indices:
        raise SystemExit("No annotation indices found in candidates or random.txt")

    prepare_output_root(out_root, args.overwrite)

    report: dict[str, Any] = {
        "strategy": "S3_best_of_n_validator",
        "dataset_root": str(dataset_root),
        "candidate_roots": [str(r) for r in roots],
        "output_root": str(out_root),
        "uses_ground_truth": False,
        "source_list": args.random_file or str(dataset_root / "VLM" / "random.txt"),
        "samples": {},
    }

    selected_annotations: list[dict] = []
    valid_image_total = 0
    for idx in indices:
        cands = [score_candidate_sample(root, idx, source_map) for root in roots]
        best = choose_best(cands)
        sample_report = {
            "candidate_scores": [
                {
                    "candidate": c.candidate_name,
                    "root": str(c.org_root),
                    "annotation": str(c.ann_path) if c.ann_path else None,
                    "score": round(c.score, 6),
                    "json_error": c.json_error,
                    "checks": c.checks,
                }
                for c in cands
            ]
        }
        if best is None or best.ann is None:
            sample_report["selected"] = None
            sample_report["status"] = "no_valid_json_candidate"
            report["samples"][idx] = sample_report
            continue
        chosen_img = best.image_path
        if not valid_image(chosen_img):
            imgs = list_valid_images_for_idx(cands)
            chosen_img = imgs[0] if imgs else None
        out_img = copy_image_to_output(chosen_img, out_root / "images", idx)
        ann_out = copy.deepcopy(best.ann)
        ensure_record_image_path(ann_out, idx, out_img)

        dump_json(out_root / "standardized_annotations" / f"{idx}.json", ann_out)
        selected_annotations.append(ann_out)
        if valid_image(out_img):
            valid_image_total += 1
        sample_report["selected"] = best.candidate_name
        sample_report["selected_score"] = round(best.score, 6)
        sample_report["selected_image"] = str(chosen_img) if chosen_img else None
        sample_report["output_json"] = str(out_root / "standardized_annotations" / f"{idx}.json")
        sample_report["output_image"] = str(out_img) if out_img else None
        sample_report["status"] = "selected"
        report["samples"][idx] = sample_report

    meta, meta_report = choose_or_build_meta(roots, selected_annotations, valid_image_total, args.meta_mode)
    dump_json(out_root / "data_meta.json", meta)
    report["meta"] = meta_report
    report["summary"] = {
        "n_indices": len(indices),
        "n_selected_json": len(selected_annotations),
        "n_valid_images": valid_image_total,
        "output_data_meta": str(out_root / "data_meta.json"),
    }
    dump_json(out_root / "s3_validator_report.json", report)
    print(f"Wrote S3 output: {out_root}")
    print(f"Wrote report: {out_root / 's3_validator_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
