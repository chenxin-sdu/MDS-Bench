# S2: Risk-Adaptive Verification (Five-Category Whitelist Patch Version)

You are a **critical-field patcher**. Your task is not to reorganize the data, nor to perform a global self-check. Instead, after copying the S0 output, you only inspect and correct a small number of high-risk fields. Except for the whitelist fields explicitly listed in this file, all other fields must remain exactly as in S0.

## Do Not Read
Do not read or use the following files:
{dataset_root}/VLM/ground_truth*.json
{dataset_root}/VLM/eval_results*.json
Any final evaluation result files

## Input
Directory under review: {dataset_root}/prompt_experiment/S0/agent1_data_organization
Raw data: {dataset_root}/data
Sample manifest: {dataset_root}/VLM/random.txt (if present, used only to verify indices and source-file order)

## Output
First fully copy the S0 output directory to: {dataset_root}/prompt_experiment/S2/agent1_data_organization
Then modify whitelist fields only within the copied S2 directory. Do not modify S0 in place.

Write the self-check report to: {dataset_root}/prompt_experiment/S2/agent1_data_organization/risk_adaptive_verification_report.json
(Each modification suggestion in the self-check report includes: which file was changed, which field, old value, new value, and a one-sentence reason.)

## Only Five Categories of Issues May Be Checked and Modified

### 1. Consistency of Task Type, Dimension, and Branches

Only the following fields may be checked and modified:

```text
media_geometry.media.task_type
media_geometry.media.dimension

media_geometry.spatial.common
media_geometry.spatial.three_d
media_geometry.spatial.video

tasks.classification
tasks.segmentation
tasks.detection

data_meta.json.task_type
```

Check rules:
task_type can only be classification / segmentation / detection
dimension can only be 2D / 3D / video
2D may only activate media_geometry.spatial.common
3D may only activate media_geometry.spatial.three_d
video may only activate media_geometry.spatial.video
In tasks, only the active branch corresponding to task_type may be retained; other task branches must be empty objects {}
Likewise, inactive spatial branches must also explicitly exist and be empty objects {}

Allowed operations:
Correct clearly illegal task_type / dimension spelling or capitalization
Clear inactive task branches according to task_type (create as `{}` if missing)
Clear inactive spatial branches according to dimension (create as `{}` if missing)
When data_meta.json.task_type is clearly inconsistent with the active task set of the sample JSONs, mechanically update it to the set of tasks that actually appear in the sample JSONs

### 2. Consistency of Class ID / label / label_name with the Existing Class Table

Only the following fields may be checked and modified:

```text
tasks.classification.label
tasks.classification.label_name

# All object-level class ID fields in detection
tasks.detection.*.category_id
tasks.detection.*.class_id
tasks.detection.*.label_id
tasks.detection.*.label

# All object-level class ID fields in segmentation
tasks.segmentation.*.category_id
tasks.segmentation.*.class_id
tasks.segmentation.*.label_id
tasks.segmentation.*.label
```

These fields may only be checked relative to the existing class table:

```text
tasks.classification.class_names
tasks.classification.classes
tasks.detection.class_names
tasks.detection.classes
tasks.segmentation.classes
tasks.segmentation.class_names
```

Allowed operations:

```text
Correct clear type errors into a form recognizable by the class table, e.g., "0" -> 0
When label/category_id conflicts with the existing class table in the same JSON, and the correct ID can be clearly determined from the same original annotation or within the same S0 sample, correct that ID
When label_name is missing, look up the corresponding name in class_names using the label value as the key, and fill in label_name
When label_name exists but contradicts class_names[str(label)], correct label_name using class_names as the authority
```

### 3. Image Path, Index, and File Alignment

Only the following fields or file relationships may be checked and modified:

```text
record.image_path
images/<idx>.<png|jpg|jpeg>
standardized_annotations/<idx>.json

data_meta.json.data_volume.total
data_meta.json.data_volume.label
data_meta.json.valid_image_n.total
```

Check rules:

```text
standardized_annotations/<idx>.json must correspond to images/<idx>.<ext>
record.image_path must be a relative path in the form images/<idx>.<ext>
The image pointed to by record.image_path must actually exist and be openable
```

Allowed operations:

```text
Correct the filename, extension, or relative-path writing of record.image_path
When image and JSON indices are misaligned and can be clearly judged from filenames/manifest order, correct the index or path
Mechanically update data_meta.json.data_volume.total to the final number of JSON samples
Mechanically update data_meta.json.data_volume.label to the final number of labeled JSON samples
Mechanically update data_meta.json.valid_image_n.total to the final number of openable images
```

### 4. Consistency of annotation_type and schema_variant

Only the following fields may be checked and modified:

```text
media_geometry.media.annotation_type
```

Check rules:

```text
annotation_type should be semantically consistent with the schema_variant of the current active task branch.
For example: if tasks.classification.schema_variant = "multiclass", then annotation_type should also be "multiclass".
Common error: schema_variant is "multiclass" but annotation_type is written as "categorical".
```

Allowed operations:

```text
When annotation_type is inconsistent with the schema_variant of the active task branch, correct annotation_type to the value of schema_variant.
```

### 5. Completion of Extra Information Provenance Fields

Only the following fields may be checked and modified:

```text
context.sample.extra.source_file
context.sample.extra.label_text
context.sample.extra.class_name
```

Prerequisite for checking: `context.sample.extra` already exists and already contains an `original_filename` field.

Check rules:

```text
source_file: relative path of the original file. If missing, derive it from original_filename and the {dataset_root}/data directory structure.
    Actually ls the subdirectories under {dataset_root}/data to determine the full path, in a form such as data/<subdir>/<filename>.
label_text: text label of the class to which the sample belongs. If missing, derive it by removing the extension from original_filename.
    For example, "Bmode_artery.mat" → "Bmode_artery".
class_name: same as label_text, i.e., the canonical name of the class to which the sample belongs.
```

Allowed operations:

```text
When source_file is missing, assemble the full relative path from original_filename and the actual data directory structure and fill it in
When label_text is missing, fill it in after removing the extension from original_filename
When class_name is missing, fill it with the same value as label_text
Derivation is allowed only when original_filename already exists in extra; otherwise take no action
```

## Modification Principles
1. Change only what can be judged mechanically.
2. When evidence is insufficient, do not change; keep the S0 original value.
3. Make only minimal patches; do not perform a global rewrite.
4. Any change to a non-whitelist field must be marked as a violation in the report.
