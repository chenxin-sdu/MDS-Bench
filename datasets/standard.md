# Data Organization Agent

## Overview

The Data Organization Agent is responsible for automatically cleaning, extracting, and standardizing raw heterogeneous medical data, mapping all data types into a unified **image + structured JSON** format to ensure data consistency, field completeness, and cross-modal usability.

## Input

### Input Location
- **Input directory**: `{dataset_root}/data/` or a user-specified raw data directory
- **Sample manifest file**: `{dataset_root}/VLM/random.txt` (or a user-explicitly specified manifest file)
- **Reading rules**: The data scan scope remains the input directory, but conversion is performed only for samples listed in `random.txt` (processed in the order of manifest indices)
- **Input format restrictions**:
  - Only process volumetric data files in the manifest that can be converted to medical images (e.g., `.nii/.nii.gz/.dcm/.mha/.nrrd`)
  - Do not process chart labels, pure tabular labels, or other non-image primary data
  - Do not process `png/jpg/jpeg` files

### Example Input File Structure
```
0 {dataset_root}/data/<subdir>/<sample_0001>.nii.gz
1 {dataset_root}/data/<subdir>/<sample_0002>.dcm
...
N {dataset_root}/data/<subdir>/<sample_XXXX>.nii
```

## Output

### Output Location
- **Output root directory**: `{dataset_root}/agent1_data_organization/`
- **Output structure**:
```
agent1_data_organization/
├── images/                          # Standardized image files (PNG/JPG format)
│   ├── 0.png
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── standardized_annotations/        # Structured JSON files (numeric indices starting from 0)
│   ├── 0.json
│   ├── 1.json
│   ├── 2.json
│   └── ...
└── data_meta.json                   # Dataset-level metadata
```

> ✅ **Core principle**: Each image corresponds to one JSON file; filenames are numeric indices starting from 0
> 
> ✅ **Only fill in real data that matches the dataset; fill unknown information with `null`**

### Output File Formats

#### Image Files
- **Format**: PNG or JPG
- **Naming convention**: Numeric indices starting from 0, e.g., `0.png`, `1.png`, `2.png`
- **Requirements**: 
  - Unified format and size (if needed)
  - Preserve original image quality
  - Standardized color space (RGB)

#### JSON Annotation File Format

Each image corresponds to one JSON file, using a standardized medical imaging dataset format. Select the appropriate structure template based on task type (classification/segmentation/detection). **When generating, remove all `xxx__desc` and `xxx__options` fields; select the spatial scheme by dimension: 2D→common; 3D→three_D; video→video**.

**Core JSON file structure**:

```json
{
  "record": {
    "schema_version": "1.0.1",
    "dataset_name": "<DATASET_NAME>",
    "image_path": "images/<FILENAME_WITH_EXT>"
  },
  "context": {
    "sample": {
      "subject_id": "NA",
      "age_years": 0,
      "sex": "unknown",
      "race_ethnicity": "NA",
      "site_id": "NA",
      "modality_primary": "Endoscopy",
      "modality_secondary": "Colonoscopy",
      "anatomical_structure": ["Colon (Abdomen)"],
      "modality_special_sample": "e.g.: CT head coronal; Microscopy 40× HE",
      "extra": {}
    },
    "text": {
      "text_id": "NA",
      "text_path": "texts/<TEXT_FILE>",
      "text_type": "report",
      "language": "en",
      "length_chars": 0
    }
  },
  "media_geometry": {
    "media": {
      "task_type": "classification|segmentation|detection",
      "leaf_task": "...",
      "annotation_type": "...",
      "dimension": "2D|3D|video",
      "confidence_global": 1.0
    },
    "spatial": {
      "common": {
        "pixel_size_px": [1280, 720],
        "pixel_spacing_mm": [0.0, 0.0],
        "was_resampled": false
      },
      "three_d": { ... },
      "video": { ... }
    }
  },
  "tasks": {
    "classification": { ... },
    "segmentation": { ... },
    "detection": { ... }
  },
  "provenance": {
    "timestamp_utc": "2025-10-17T03:21:00Z",
    "notes": "Free-form notes area"
  }
}
```

**Key field descriptions**:

| Field Path | Meaning | Example |
|-----------|------|------|
| `record.dataset_name` | Dataset name | `"DiabeticRetinopathy_Detection"` |
| `record.image_path` | Relative image path | `"images/0.png"` |
| `context.sample.modality_primary` | Imaging modality | `"Fundus"` |
| `context.sample.anatomical_structure` | Anatomical structure | `["Retina (Head and Neck)"]` |
| `media_geometry.media.dimension` | Data dimension | `"2D"` |
| `media_geometry.spatial.common.pixel_size_px` | Image size | `[512, 496]` |
| `tasks.classification.classes` | Class mapping (classification task) | `{"0": "No DR", "1": "Mild DR"}` |
| `tasks.classification.label` | Current sample label (classification task) | `0` |
| `provenance.timestamp_utc` | Annotation generation time | `"2025-10-29T12:03:02Z"` |

**Select structure by task type**:

1. **Classification task (classification)**: Use `tasks.classification`, including fields such as `schema_variant` (multiclass/multilabel/exam_level), `class_names`, `label`, etc.
2. **Segmentation task (segmentation)**: Use `tasks.segmentation`, including fields such as `schema_variant` (2d_coco/3d_mask/video_coco), `classes`, `coco_2d`/`mask_3d`/`video_coco`, etc.
3. **Detection task (detection)**: Use `tasks.detection`, including fields such as `schema_variant` (2d/3d/video), `class_names`, `boxes2d`/`boxes3d`/`frame_boxes2d`, etc.

**Data filling principles**:
- ✅ **Only fill in real data**: Fill according to the actual dataset; do not fabricate information
- ✅ **Unknown data as null**: For unknown information, uniformly fill with `null` or use default values (e.g., `"NA"`, `0`, `"unknown"`)
- ✅ **Maintain data consistency**: The same type of information within the same dataset should remain consistent
- ✅ **Select spatial scheme by dimension**: Use `common` for 2D data, `three_d` for 3D data, and `video` for video data


#### Dataset Metadata File (data_meta.json)

Dataset-level metadata file, using Schema 1.0.2 format:

```json
{
  "schema_version": "1.0.2",
  "dataset_name": "dataset_name",
  "release_date": "YYYY-MM or YYYY-MM-DD",
  "homepage_url": "https://...",
  "organization": ["MICCAI"],
  "challenge_series": "MICCAI",
  "license": "CC BY 4.0",
  "dataset_description": "One to three sentences summarizing source/modality/scale/task and highlights.",
  "modality_primary": ["Endoscopy"],
  "modality_secondary": ["Colonoscopy"],
  "anatomical_structure": ["Colon (Abdomen)"],
  "disease": ["NA"],
  "domain": "medical",
  "data_volume": {
    "total": 0,
    "train": 0,
    "val": 0,
    "test": 0,
    "label": 0
  },
  "valid_image_n": {
    "total": 0
  },
  "label_presence": "labeled",
  "task_type": ["segmentation", "detection", "classification"],
  "num_classes_per_task": {
    "classification": {
      "num_classes": 0
    },
    "detection": {
      "type": "single-class",
      "num_classes": 0
    },
    "segmentation": {
      "num_classes": 0,
      "include_background": true
    }
  }
}
```

**Key field descriptions**:

| Field | Type | Description | Example |
|------|------|------|------|
| `schema_version` | string | Schema version number | `"1.0.2"` |
| `dataset_name` | string | Official dataset name or common abbreviation | `"DiabeticRetinopathy_Detection"` |
| `release_date` | string | First public release date | `"2015-01-01"` |
| `homepage_url` | string | Dataset homepage, paper, or public repository link | `"https://www.kaggle.com/..."` |
| `modality_primary` | array | Primary imaging modality | `["Fundus"]` |
| `modality_secondary` | array | Sub-modality | `["CFP"]` |
| `anatomical_structure` | array | Specific organ/site/lesion | `["Retina (Head and Neck)"]` |
| `data_volume` | object | Total volume and splits | `{"total": 1286, "train": 1000, "val": 200, "test": 86, "label": 1286}` |
| `task_type` | array | Research/evaluation tasks | `["classification"]` |
| `num_classes_per_task` | object | Number of classes/targets per task | See example above |

## Processing Pipeline

1. **Data scanning**: Scan the input directory and identify all processable files
2. **Image-convertible filtering**: 
   - Determine whether data can be converted to image format
   - Mark as "image-convertible / not image-convertible"
   - Record filtering results
3. **Data cleaning**:
   - Remove corrupted or unreadable files
   - Standardize file formats
   - Handle missing values
4. **Format conversion**:
   - Convert DICOM and other formats to PNG/JPG
   - Extract metadata information
   - Standardize image size and color space
5. **Structured annotation generation**:
   - Extract information from original annotation files
   - Map to standard JSON format
   - Fill in missing fields
6. **Quality check**:
   - Verify image file integrity
   - Verify JSON format correctness
   - Check field completeness
7. **Output generation**:
   - Save standardized images
   - Save structured JSON
   - Generate dataset metadata


## Usage Instructions

### Input Parameters
- `input_dir`: Path to the raw data directory
- `output_dir`: Path to the output directory (default: `{input_dir}/agent1_data_organization/`)
- `image_format`: Output image format (PNG or JPG, default PNG)
- `quality_check`: Whether to perform quality checks (default True)

### Output Validation
After processing is complete, check the following:
1. Whether all standardized images exist under the `images/` directory
2. Whether corresponding JSON files exist under the `standardized_annotations/` directory
3. Whether statistics in `data_meta.json` are correct
4. Whether JSON file formats comply with the specification

## Notes

1. **Data consistency**: Ensure image IDs correspond one-to-one with JSON filenames
2. **Field completeness**: JSON must include all required fields
3. **Cross-modal usability**: The output format should support the processing needs of subsequent agents
4. **Error handling**: For data that cannot be processed, error logs should be recorded without affecting the processing of other data
