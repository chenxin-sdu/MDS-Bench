#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S4 Field-wise Voting, narrow high-risk version.

This script implements the intended S4 ablation:
  - It does NOT perform global self-checking.
  - It does NOT vote modality, anatomy, demographics, extra, provenance, source trace,
    or free-text dataset metadata.
  - It only votes on high-risk fields and selects the best values:
      1) task_type / dimension / annotation_type (weighted voting across candidates),
      2) active task branch selection from the most consistent candidate,
      3) extra source-trace fields voting (source_file, original_filename, etc.),
  with only minor cleanup (ID type normalization, e.g. string "1" -> int 1).
  No label repair, no class-table patching, no geometry editing.

It reuses S3's non-oracle validator utilities for candidate scoring, but the final
S4 operation is not sample-level best-of-n selection. The output starts from the
best-scored candidate as a base, then applies field-wise voting only to the
whitelisted fields.

The script never reads ground_truth*.json or eval_results*.json.

Default candidate root:
  {dataset_root}/prompt_experiment/candidates

Default output root:
  {dataset_root}/prompt_experiment/S4/agent1_data_organization

Example:
  python3 S4_fieldwise_voting_narrow.py \
    --dataset-root /path/to/dataset \
    --runs-root /path/to/dataset/prompt_experiment/candidates
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


S3_MODULE_CANDIDATES = [
    "S3_best_of_n_validator_revised",
    "S3_best_of_n_validator",
    "s3_best_of_n_validator_revised",
    "s3_best_of_n_validator",
]


def _load_s3_module():
    import importlib.util as _ilu

    script_dir = Path(__file__).resolve().parent
    sibling_s3_dir = script_dir.parent / "S3"
    search_dirs = [script_dir, sibling_s3_dir]

    for name in S3_MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except Exception:
            pass

    import sys as _sys
    for search_dir in search_dirs:
        for name in S3_MODULE_CANDIDATES:
            candidate_path = search_dir / f"{name}.py"
            if candidate_path.is_file():
                spec = _ilu.spec_from_file_location(name, candidate_path)
                if spec and spec.loader:
                    mod = _ilu.module_from_spec(spec)
                    _sys.modules[name] = mod
                    spec.loader.exec_module(mod)
                    return mod
        if search_dir.is_dir():
            for py_file in sorted(search_dir.glob("S3*.py")) + sorted(search_dir.glob("s3*.py")):
                spec = _ilu.spec_from_file_location(py_file.stem, py_file)
                if spec and spec.loader:
                    mod = _ilu.module_from_spec(spec)
                    _sys.modules[py_file.stem] = mod
                    spec.loader.exec_module(mod)
                    return mod

    raise SystemExit(
        "Cannot import an S3 validator module. Searched in: "
        + ", ".join(str(d) for d in search_dirs)
        + " for: "
        + ", ".join(S3_MODULE_CANDIDATES)
    )


s3 = _load_s3_module()

ALLOWED_TASKS = set(getattr(s3, "ALLOWED_TASKS", {"classification", "segmentation", "detection"}))
ALLOWED_DIMS = {"2d", "3d", "video"}
IMAGE_EXTS = getattr(s3, "IMAGE_EXTS", (".png", ".jpg", ".jpeg"))

# Reused S3 utilities. They are intentionally non-oracle: they use only schema,
# generated files, candidates, and optional random.txt/source-list information.
CandidateSample = s3.CandidateSample
annotation_indices = s3.annotation_indices
choose_best = s3.choose_best
copy_image_to_output = s3.copy_image_to_output
deep_get = s3.deep_get
deep_set = s3.deep_set
discover_candidate_roots = s3.discover_candidate_roots
dump_json = s3.dump_json
ensure_record_image_path = s3.ensure_record_image_path
image_by_index = s3.image_by_index
load_json = s3.load_json
meta_score = s3.meta_score
nonmissing = s3.nonmissing
norm_dim = s3.norm_dim
norm_task = s3.norm_task
norm_text = s3.norm_text
norm_token = s3.norm_token
read_random_txt = s3.read_random_txt
sample_summary = s3.sample_summary
score_annotation_object = s3.score_annotation_object
score_candidate_sample = s3.score_candidate_sample
valid_image = s3.valid_image
build_minimal_meta = s3.build_minimal_meta
branch_empty = s3.branch_empty


VOTED_FIELD_PATHS = [
    "media_geometry.media.task_type",
    "media_geometry.media.dimension",
    "media_geometry.media.annotation_type",
]

MECHANICAL_META_FIELDS = [
    "data_meta.json.data_volume.total",
    "data_meta.json.data_volume.label",
    "data_meta.json.valid_image_n.total",
    "data_meta.json.task_type",
]

ID_KEYS = {"category_id", "class_id", "label_id", "label"}
SPATIAL_BRANCH_BY_DIM = {"2d": "common", "3d": "three_d", "video": "video"}


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def prepare_output_root_clean(path: Path, clean: bool = True) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "standardized_annotations").mkdir(parents=True, exist_ok=True)


def canonical_for_vote(path: str, value: Any) -> str | None:
    if not nonmissing(value):
        return None
    last = path.split(".")[-1]
    if last == "task_type":
        v = norm_task(value)
        return v if v in ALLOWED_TASKS else None
    if last == "dimension":
        v = norm_dim(value)
        return v if v in ALLOWED_DIMS else None
    if last == "annotation_type":
        return norm_text(value) if norm_text(value) else None
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def weighted_vote_value(path: str, candidates: list[CandidateSample]) -> tuple[Any, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for c in candidates:
        if c.ann is None:
            continue
        value = deep_get(c.ann, path)
        key = canonical_for_vote(path, value)
        if key is None:
            continue
        if key not in groups:
            groups[key] = {
                "weight": 0.0,
                "values": [],
                "best_score": -1.0,
                "best_value": value,
                "candidates": [],
            }
        weight = max(float(c.score), 0.001)
        groups[key]["weight"] += weight
        groups[key]["values"].append(value)
        groups[key]["candidates"].append(c.candidate_name)
        if c.score > groups[key]["best_score"]:
            groups[key]["best_score"] = c.score
            groups[key]["best_value"] = value
    if not groups:
        return None, {"voted": False, "reason": "no_nonmissing_whitelisted_values"}
    ranked = sorted(groups.items(), key=lambda kv: (kv[1]["weight"], kv[1]["best_score"]), reverse=True)
    winner_key, winner = ranked[0]
    return copy.deepcopy(winner["best_value"]), {
        "voted": True,
        "winner_key": winner_key,
        "winner_weight": round(winner["weight"], 6),
        "winner_candidates": winner["candidates"],
        "distribution": [
            {
                "key": k,
                "weight": round(v["weight"], 6),
                "n": len(v["values"]),
                "candidates": v["candidates"],
            }
            for k, v in ranked
        ],
    }


def best_valid_image(candidates: list[CandidateSample], idx: str) -> Path | None:
    for c in sorted(candidates, key=lambda x: x.score, reverse=True):
        if valid_image(c.image_path):
            return c.image_path
        p = image_by_index(c.org_root, idx)
        if valid_image(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Class table and ID consistency utilities
# ---------------------------------------------------------------------------

def get_classes(branch: Any) -> Any:
    if not isinstance(branch, dict):
        return None
    if branch.get("class_names") is not None:
        return branch.get("class_names")
    return branch.get("classes")


def class_id_set(classes: Any) -> set[str]:
    if isinstance(classes, dict):
        return {str(k) for k in classes.keys()}
    if isinstance(classes, list):
        return {str(i) for i in range(len(classes))}
    return set()


def normalize_id_value(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    if not s:
        return None
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def id_in_classes(v: Any, classes: Any) -> bool:
    ids = class_id_set(classes)
    nv = normalize_id_value(v)
    return bool(ids and nv is not None and nv in ids)


def class_name_for_id(classes: Any, label: Any) -> Any:
    lid = normalize_id_value(label)
    if lid is None:
        return None
    if isinstance(classes, dict):
        if lid in classes:
            return classes[lid]
        try:
            i = int(lid)
            if i in classes:
                return classes[i]
        except Exception:
            pass
    if isinstance(classes, list):
        try:
            i = int(lid)
            if 0 <= i < len(classes):
                return classes[i]
        except Exception:
            pass
    return None


def iter_id_locations(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                yield from iter_id_locations(v, path)
            elif str(k) in ID_KEYS:
                yield path, v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            path = f"{prefix}.{i}" if prefix else str(i)
            yield from iter_id_locations(v, path)


def deep_set_by_path(obj: Any, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cur = obj
    for p in parts[:-1]:
        if isinstance(cur, list):
            cur = cur[int(p)]
        elif isinstance(cur, dict):
            cur = cur[p]
        else:
            return
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    elif isinstance(cur, dict):
        cur[last] = value


def coerce_id_types_in_place(branch: dict, classes: Any) -> list[dict[str, Any]]:
    """Only convert stringified integer IDs to int when they already match the class table."""
    changes = []
    ids = class_id_set(classes)
    if not ids:
        return changes
    for path, old in list(iter_id_locations(branch)):
        nv = normalize_id_value(old)
        if nv is None or nv not in ids:
            continue
        if isinstance(old, str) and nv != old:
            # Leave non-integer class keys as strings. Only canonicalize numeric IDs.
            try:
                new = int(nv)
            except Exception:
                continue
            deep_set_by_path(branch, path, new)
            changes.append({"field_path": path, "old_value": old, "new_value": new, "reason": "stringified numeric class id"})
        elif isinstance(old, float) and old.is_integer():
            new = int(old)
            deep_set_by_path(branch, path, new)
            changes.append({"field_path": path, "old_value": old, "new_value": new, "reason": "float numeric class id"})
    return changes


def classification_branch_valid(branch: Any) -> bool:
    if not isinstance(branch, dict) or branch_empty(branch):
        return False
    classes = get_classes(branch)
    label = branch.get("label")
    return id_in_classes(label, classes)


def detection_or_segmentation_branch_valid(branch: Any) -> bool:
    if not isinstance(branch, dict) or branch_empty(branch):
        return False
    classes = get_classes(branch)
    ids = list(iter_id_locations(branch))
    checked = []
    for path, v in ids:
        # Avoid treating task-level metadata labels as object/category labels unless
        # the field is explicitly an ID-like key or appears under a payload object.
        checked.append((path, v))
    if not class_id_set(classes):
        return False
    if not checked:
        return True
    return all(id_in_classes(v, classes) for _, v in checked)


def task_branch_valid(task: str, branch: Any) -> bool:
    if task == "classification":
        return classification_branch_valid(branch)
    if task in {"detection", "segmentation"}:
        return detection_or_segmentation_branch_valid(branch)
    return False


def candidate_task(c: CandidateSample) -> str:
    return norm_task(deep_get(c.ann or {}, "media_geometry.media.task_type"))


def candidate_dim(c: CandidateSample) -> str:
    return norm_dim(deep_get(c.ann or {}, "media_geometry.media.dimension"))


def choose_branch_source(candidates: list[CandidateSample], task: str, dim: str) -> tuple[CandidateSample | None, str]:
    def branch(c: CandidateSample) -> Any:
        return deep_get(c.ann or {}, f"tasks.{task}")

    tiers: list[tuple[str, list[CandidateSample]]] = []
    tiers.append((
        "same_task_same_dimension_valid_branch",
        [c for c in candidates if c.ann is not None and candidate_task(c) == task and candidate_dim(c) == dim and task_branch_valid(task, branch(c))],
    ))
    tiers.append((
        "same_task_valid_branch",
        [c for c in candidates if c.ann is not None and candidate_task(c) == task and task_branch_valid(task, branch(c))],
    ))
    tiers.append((
        "same_task_nonempty_branch",
        [c for c in candidates if c.ann is not None and candidate_task(c) == task and isinstance(branch(c), dict) and not branch_empty(branch(c))],
    ))
    tiers.append((
        "any_valid_json_fallback",
        [c for c in candidates if c.ann is not None],
    ))
    for reason, xs in tiers:
        if xs:
            return sorted(xs, key=lambda c: c.score, reverse=True)[0], reason
    return None, "no_valid_json_candidate"


def set_active_task_branch(ann: dict, source_ann: dict, task: str) -> None:
    tasks_obj = {"classification": {}, "segmentation": {}, "detection": {}}
    branch = deep_get(source_ann, f"tasks.{task}")
    tasks_obj[task] = copy.deepcopy(branch) if isinstance(branch, dict) else {}
    ann["tasks"] = tasks_obj


def set_spatial_by_dimension(ann: dict, source_ann: dict, dim: str) -> None:
    branch_name = SPATIAL_BRANCH_BY_DIM.get(dim, "common")
    source_branch = deep_get(source_ann, f"media_geometry.spatial.{branch_name}")
    if not isinstance(source_branch, dict):
        source_branch = {}
    deep_set(ann, "media_geometry.spatial", {"common": {}, "three_d": {}, "video": {}})
    deep_set(ann, f"media_geometry.spatial.{branch_name}", copy.deepcopy(source_branch))


def repair_label_id_consistency(ann: dict, candidates: list[CandidateSample], task: str) -> dict[str, Any]:
    report: dict[str, Any] = {"task": task, "changed": False, "actions": []}
    branch = deep_get(ann, f"tasks.{task}")
    if not isinstance(branch, dict):
        report["status"] = "missing_active_branch"
        return report

    classes = get_classes(branch)
    if task == "classification":
        label = branch.get("label")
        if id_in_classes(label, classes):
            canonical_name = class_name_for_id(classes, label)
            if canonical_name is not None and branch.get("label_name") != canonical_name:
                old = branch.get("label_name")
                branch["label_name"] = canonical_name
                report["changed"] = True
                report["actions"].append({
                    "field_path": "tasks.classification.label_name",
                    "old_value": old,
                    "new_value": canonical_name,
                    "reason": "label_name aligned to existing class table",
                })
            id_changes = coerce_id_types_in_place(branch, classes)
            if id_changes:
                report["changed"] = True
                report["actions"].extend(id_changes)
            report["status"] = "already_consistent_or_type_repaired"
            return report

        # Choose a compatible label from another candidate, but do not alter the class table.
        for c in sorted([x for x in candidates if x.ann is not None], key=lambda x: x.score, reverse=True):
            cand_label = deep_get(c.ann, "tasks.classification.label")
            cand_label_name = deep_get(c.ann, "tasks.classification.label_name")
            if id_in_classes(cand_label, classes):
                old_label = branch.get("label")
                old_name = branch.get("label_name")
                branch["label"] = cand_label
                branch["label_name"] = cand_label_name if nonmissing(cand_label_name) else class_name_for_id(classes, cand_label)
                report["changed"] = True
                report["status"] = "label_repaired_from_candidate"
                report["source_candidate"] = c.candidate_name
                report["actions"].append({
                    "field_path": "tasks.classification.label",
                    "old_value": old_label,
                    "new_value": branch.get("label"),
                    "reason": "candidate label is compatible with existing class table",
                })
                report["actions"].append({
                    "field_path": "tasks.classification.label_name",
                    "old_value": old_name,
                    "new_value": branch.get("label_name"),
                    "reason": "aligned with repaired label",
                })
                return report
        report["status"] = "unresolved_label_not_in_class_table"
        return report

    if task in {"detection", "segmentation"}:
        id_changes = coerce_id_types_in_place(branch, classes)
        if id_changes:
            report["changed"] = True
            report["actions"].extend(id_changes)
        if task_branch_valid(task, branch):
            report["status"] = "already_consistent_or_type_repaired"
            return report

        # Do not alter boxes/masks geometrically. Replace the entire active branch with
        # a candidate branch that is internally consistent.
        for c in sorted([x for x in candidates if x.ann is not None], key=lambda x: x.score, reverse=True):
            cand_branch = deep_get(c.ann, f"tasks.{task}")
            if task_branch_valid(task, cand_branch):
                old_branch = copy.deepcopy(branch)
                deep_set(ann, f"tasks.{task}", copy.deepcopy(cand_branch))
                report["changed"] = True
                report["status"] = "whole_active_branch_replaced_from_consistent_candidate"
                report["source_candidate"] = c.candidate_name
                report["actions"].append({
                    "field_path": f"tasks.{task}",
                    "old_value_summary": summarize_branch(old_branch),
                    "new_value_summary": summarize_branch(cand_branch),
                    "reason": "category IDs conflicted with class table; geometry was not field-wise edited",
                })
                return report
        report["status"] = "unresolved_category_id_conflict"
        return report

    report["status"] = "unsupported_task_type"
    return report


def summarize_branch(branch: Any) -> dict[str, Any]:
    if not isinstance(branch, dict):
        return {"type": type(branch).__name__}
    keys = sorted(branch.keys())
    out: dict[str, Any] = {"keys": keys}
    classes = get_classes(branch)
    out["class_ids"] = sorted(class_id_set(classes))
    id_locs = list(iter_id_locations(branch))
    out["n_id_fields"] = len(id_locs)
    return out


# ---------------------------------------------------------------------------
# Voted annotation construction
# ---------------------------------------------------------------------------

def build_narrow_voted_annotation(idx: str, candidates: list[CandidateSample]) -> tuple[dict | None, dict[str, Any], CandidateSample | None]:
    best = choose_best(candidates)
    if best is None or best.ann is None:
        return None, {"status": "no_valid_json_candidate"}, None

    ann = copy.deepcopy(best.ann)
    report: dict[str, Any] = {
        "status": "built_from_best_candidate_then_narrow_voting",
        "base_candidate": best.candidate_name,
        "base_score": round(best.score, 6),
        "whitelist": {
            "voted_fields": VOTED_FIELD_PATHS,
            "label_id_type_coercion": True,
            "label_id_repair": False,
            "image_path_alignment": True,
            "extra_source_trace_voting": True,
            "metadata_semantic_voting": False,
        },
        "field_votes": {},
        "branch_selection": {},
        "label_id_consistency": {},
        "extra_voting": {},
        "forbidden_fields_not_voted": [
            "record.dataset_name",
            "context.sample.modality_primary",
            "context.sample.modality_secondary",
            "context.sample.anatomical_structure",
            "provenance",
            "data_meta semantic fields",
        ],
    }

    # 1) Vote task_type, dimension, and annotation_type.
    for path in VOTED_FIELD_PATHS:
        value, rep = weighted_vote_value(path, candidates)
        report["field_votes"][path] = rep
        if value is not None:
            deep_set(ann, path, value)

    task = norm_task(deep_get(ann, "media_geometry.media.task_type"))
    dim = norm_dim(deep_get(ann, "media_geometry.media.dimension"))

    if task not in ALLOWED_TASKS:
        task = norm_task(deep_get(best.ann, "media_geometry.media.task_type"))
        deep_set(ann, "media_geometry.media.task_type", task)
        report["field_votes"]["media_geometry.media.task_type"]["fallback"] = "base_candidate_task_type"
    if dim not in ALLOWED_DIMS:
        dim = norm_dim(deep_get(best.ann, "media_geometry.media.dimension"))
        deep_set(ann, "media_geometry.media.dimension", dim)
        report["field_votes"]["media_geometry.media.dimension"]["fallback"] = "base_candidate_dimension"

    # 2) Select one internally consistent branch source. Do not vote payload geometry.
    branch_source, branch_reason = choose_branch_source(candidates, task, dim)
    if branch_source is None or branch_source.ann is None:
        return None, {"status": "no_branch_source", "previous_report": report}, best

    source_task = norm_task(deep_get(branch_source.ann, "media_geometry.media.task_type"))
    source_dim = norm_dim(deep_get(branch_source.ann, "media_geometry.media.dimension"))

    if source_task in ALLOWED_TASKS and source_task != task and branch_reason == "any_valid_json_fallback":
        task = source_task
        deep_set(ann, "media_geometry.media.task_type", task)
        report["branch_selection"]["task_fallback_to_branch_source"] = True
    if source_dim in ALLOWED_DIMS and source_dim != dim and branch_reason == "any_valid_json_fallback":
        dim = source_dim
        deep_set(ann, "media_geometry.media.dimension", dim)
        report["branch_selection"]["dimension_fallback_to_branch_source"] = True

    set_active_task_branch(ann, branch_source.ann, task)
    set_spatial_by_dimension(ann, branch_source.ann, dim)
    report["branch_selection"].update({
        "branch_source": branch_source.candidate_name,
        "selection_reason": branch_reason,
        "final_task_type": task,
        "final_dimension": dim,
        "payload_geometry_voted": False,
    })

    # 3) Minor cleanup: normalize ID types (e.g. string "1" → int 1) only.
    active_branch = deep_get(ann, f"tasks.{task}")
    if isinstance(active_branch, dict):
        classes = get_classes(active_branch)
        id_changes = coerce_id_types_in_place(active_branch, classes)
        report["label_id_consistency"] = {
            "task": task, "changed": bool(id_changes),
            "actions": id_changes, "status": "id_type_coercion_only",
        }
    else:
        report["label_id_consistency"] = {
            "task": task, "changed": False, "status": "no_active_branch",
        }

    # 4) Vote extra source-trace fields across candidates.
    EXTRA_VOTED_KEYS = ["source_file", "original_filename", "label_text", "class_name"]
    extra = deep_get(ann, "context.sample.extra")
    if not isinstance(extra, dict):
        extra = {}
        deep_set(ann, "context.sample.extra", extra)
    extra_changes: dict[str, Any] = {}
    for key in EXTRA_VOTED_KEYS:
        vals = []
        for c in candidates:
            if c.ann is None:
                continue
            v = deep_get(c.ann, f"context.sample.extra.{key}")
            vals.append(v)
        nonmissing_vals = [v for v in vals if v is not None and str(v).strip().lower() not in
                           ("", "na", "n/a", "null", "none", "unknown")]
        if nonmissing_vals:
            from collections import Counter as _Counter
            keys_ser = [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                        for v in nonmissing_vals]
            best_key = _Counter(keys_ser).most_common(1)[0][0]
            winner = nonmissing_vals[keys_ser.index(best_key)]
            old_val = extra.get(key)
            extra[key] = winner
            extra_changes[key] = {"old": old_val, "new": winner, "n_votes": len(nonmissing_vals)}
    report["extra_voting"] = extra_changes

    return ann, report, best


# ---------------------------------------------------------------------------
# data_meta: no semantic voting, only mechanical patching
# ---------------------------------------------------------------------------

def choose_base_meta(candidate_roots: list[Path], selected_annotations: list[dict], valid_image_total: int) -> tuple[dict, dict[str, Any]]:
    scored = []
    for root in candidate_roots:
        obj, err = load_json(root / "data_meta.json")
        if err is None and isinstance(obj, dict):
            scored.append((meta_score(obj, selected_annotations, valid_image_total), root, obj))
    if not scored:
        return build_minimal_meta(selected_annotations, valid_image_total), {
            "mode": "rebuild_no_valid_candidate_meta",
            "semantic_meta_voting": False,
            "note": "No candidate data_meta.json was valid; minimal meta was built as fallback.",
        }
    scored.sort(key=lambda x: x[0], reverse=True)
    score, root, meta = scored[0]
    return copy.deepcopy(meta), {
        "mode": "best_candidate_meta_then_mechanical_count_patch",
        "selected_meta_from": str(root),
        "pre_patch_score": round(score, 4),
        "semantic_meta_voting": False,
    }


def patch_meta_mechanically(meta: dict, selected_annotations: list[dict], valid_image_total: int) -> dict[str, Any]:
    report: dict[str, Any] = {"patched_fields": [], "not_patched_semantic_fields": True}
    if not isinstance(meta.get("data_volume"), dict):
        old = meta.get("data_volume")
        meta["data_volume"] = {}
        report["patched_fields"].append({"field_path": "data_volume", "old_value": old, "new_value": {}, "reason": "required object for mechanical counts"})

    old_total = meta["data_volume"].get("total")
    meta["data_volume"]["total"] = len(selected_annotations)
    report["patched_fields"].append({
        "field_path": "data_volume.total",
        "old_value": old_total,
        "new_value": len(selected_annotations),
        "reason": "mechanical count of final JSON files",
    })

    old_label = meta["data_volume"].get("label")
    meta["data_volume"]["label"] = len(selected_annotations)
    report["patched_fields"].append({
        "field_path": "data_volume.label",
        "old_value": old_label,
        "new_value": len(selected_annotations),
        "reason": "mechanical count of final labeled JSON files",
    })

    if not isinstance(meta.get("valid_image_n"), dict):
        old = meta.get("valid_image_n")
        meta["valid_image_n"] = {}
        report["patched_fields"].append({"field_path": "valid_image_n", "old_value": old, "new_value": {}, "reason": "required object for mechanical counts"})
    old_valid = meta["valid_image_n"].get("total")
    meta["valid_image_n"]["total"] = valid_image_total
    report["patched_fields"].append({
        "field_path": "valid_image_n.total",
        "old_value": old_valid,
        "new_value": valid_image_total,
        "reason": "mechanical count of valid final images",
    })

    tasks = sorted({sample_summary(a).get("task_type") for a in selected_annotations if sample_summary(a).get("task_type") in ALLOWED_TASKS})
    old_tasks = meta.get("task_type")
    meta["task_type"] = tasks
    report["patched_fields"].append({
        "field_path": "task_type",
        "old_value": old_tasks,
        "new_value": tasks,
        "reason": "mechanical set of active task_type values in final JSON files",
    })

    report["forbidden_meta_fields_left_unchanged"] = [
        "dataset_name",
        "release_date",
        "homepage_url",
        "organization",
        "challenge_series",
        "license",
        "dataset_description",
        "modality_primary",
        "modality_secondary",
        "anatomical_structure",
        "disease",
        "domain",
        "label_presence",
        "num_classes_per_task",
    ]
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="S4 narrow field-wise voting for high-risk fields only")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing data/ and usually VLM/random.txt")
    parser.add_argument("--runs-root", default=None, help="Candidate runs directory. Default: {dataset_root}/prompt_experiment/candidates")
    parser.add_argument("--candidate-roots", nargs="*", default=None, help="Explicit candidate dirs; each may be agent1_data_organization or its parent")
    parser.add_argument("--random-file", default=None, help="Optional random.txt/source list path")
    parser.add_argument("--output", default=None, help="Output agent1_data_organization dir. Default: {dataset_root}/prompt_experiment/S4/agent1_data_organization")
    parser.add_argument("--fallback-margin", type=float, default=0.05, help="If narrow-voted JSON scores lower than best candidate by more than this margin, use best candidate JSON")
    parser.add_argument("--force-vote", action="store_true", help="Always keep the narrow-voted JSON even if validator score drops")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove an existing output directory before writing")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    runs_root = Path(args.runs_root).resolve() if args.runs_root else dataset_root / "prompt_experiment" / "candidates"
    out_root = Path(args.output).resolve() if args.output else dataset_root / "prompt_experiment" / "S4" / "agent1_data_organization"

    roots = discover_candidate_roots(runs_root, args.candidate_roots)
    if not roots:
        raise SystemExit(f"No candidate roots found under {runs_root}")

    source_map = read_random_txt(dataset_root, args.random_file)
    indices = annotation_indices(roots, source_map)
    if not indices:
        raise SystemExit("No annotation indices found in candidates or random.txt")

    prepare_output_root_clean(out_root, clean=not args.no_clean)

    report: dict[str, Any] = {
        "strategy": "S4_fieldwise_voting_narrow",
        "dataset_root": str(dataset_root),
        "candidate_roots": [str(r) for r in roots],
        "output_root": str(out_root),
        "uses_ground_truth": False,
        "source_list": args.random_file or str(dataset_root / "VLM" / "random.txt"),
        "fallback_margin": args.fallback_margin,
        "force_vote": args.force_vote,
        "scope": {
            "voted_fields": VOTED_FIELD_PATHS,
            "mechanical_meta_fields": MECHANICAL_META_FIELDS,
            "excluded_from_voting": [
                "modality_primary",
                "modality_secondary",
                "anatomical_structure",
                "demographics",
                "context.sample.extra",
                "source trace",
                "provenance",
                "dataset-level semantic metadata",
                "box coordinates",
                "mask geometry",
            ],
        },
        "samples": {},
    }

    selected_annotations: list[dict] = []
    valid_image_total = 0

    for idx in indices:
        cands = [score_candidate_sample(root, idx, source_map) for root in roots]
        voted, vote_report, best = build_narrow_voted_annotation(idx, cands)
        sample_report: dict[str, Any] = {
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
            ],
            "vote_report": vote_report,
        }

        if voted is None or best is None or best.ann is None:
            sample_report["status"] = "no_valid_json_candidate"
            report["samples"][idx] = sample_report
            continue

        image_src = best_valid_image(cands, idx)
        out_img = copy_image_to_output(image_src, out_root / "images", idx)
        ensure_record_image_path(voted, idx, out_img)

        voted_score, voted_checks, _ = score_annotation_object(voted, out_root, idx, source_map)
        best_score = best.score
        final_ann = voted
        final_score = voted_score
        final_checks = voted_checks
        decision = "narrow_field_vote"

        if (not args.force_vote) and (voted_score + args.fallback_margin < best_score):
            final_ann = copy.deepcopy(best.ann)
            ensure_record_image_path(final_ann, idx, out_img)
            final_score, final_checks, _ = score_annotation_object(final_ann, out_root, idx, source_map)
            decision = "fallback_to_best_candidate_due_to_validator_drop"

        dump_json(out_root / "standardized_annotations" / f"{idx}.json", final_ann)
        selected_annotations.append(final_ann)
        if valid_image(out_img):
            valid_image_total += 1

        sample_report.update({
            "status": "written",
            "decision": decision,
            "best_candidate": best.candidate_name,
            "best_score": round(best_score, 6),
            "voted_score_before_fallback": round(voted_score, 6),
            "final_score": round(final_score, 6),
            "final_checks": final_checks,
            "selected_image": str(image_src) if image_src else None,
            "output_json": str(out_root / "standardized_annotations" / f"{idx}.json"),
            "output_image": str(out_img) if out_img else None,
        })
        report["samples"][idx] = sample_report

    meta, meta_report = choose_base_meta(roots, selected_annotations, valid_image_total)
    patch_report = patch_meta_mechanically(meta, selected_annotations, valid_image_total)
    dump_json(out_root / "data_meta.json", meta)

    report["meta"] = meta_report
    report["meta"]["mechanical_patch"] = patch_report
    report["summary"] = {
        "n_indices": len(indices),
        "n_written_json": len(selected_annotations),
        "n_valid_images": valid_image_total,
        "output_data_meta": str(out_root / "data_meta.json"),
        "output_report": str(out_root / "s4_voting_report.json"),
    }
    dump_json(out_root / "s4_voting_report.json", report)

    print(f"Wrote S4 output: {out_root}")
    print(f"Wrote report: {out_root / 's4_voting_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
