# Usage notes for Python codes

The dataset contains hyperspectral images stored in both ENVI- and TIFF-formats.
The images in TIFF-format have been flat-field corrected with a white reference
sample and a dark-current sample, except for the images in the References.zip.
These are the white and dark-current references saved in TIFF-format.
The ENVI-format images are as they were saved by the hyperspectral camera.

Functions for reading these files are provided in Python files "read_envi.py"
and "spectral_tiffs.py". Additionally, the dataset contains manual segmentations
created with SpimLab software. The native vector graphics format segmentations
are in *.csv files and bitmap raster renders of these vector graphics are in
*masks.tif files. Methods for rendering the vector graphics segmentations into
bitmap rasters are provided in "annotations.py".

Note for MATLAB-users: We have included read_stiff and read_mtiff functions in
this package. These functions were cobbled together from MATLAB's inconsistent
TIFF support and are provided as examples of how the data loading might be done
in MATLAB.

## Prerequisites
The Python environment needs packages:
- numpy             (TIFF, ENVI)
- pillow            (annotations)
- pyyaml            (TIFF)
- tifffile          (TIFF)
- python-dateutil   (ENVI)

## Usage example 1
Let's load the spectral image P001. The spectral image data is in file "P001.tif"
and its segmentation masks "P001, masks.tif".

```python
from spectral_tiffs import read_stiff, read_mtiff

data_cube, center_wavelengths, preview_image, metadata = read_stiff("P001.tif")
masks = read_mtiff("P001, masks.tif")
```

This results in several variables in the environment:
- `data_cube` is an ndarray of shape (height, width, band) and contains the
   spectral image.
- `center_wavelengths` is a vector of spectral band center wavelengths.
- `preview_image` is an RGB render of the spectral image, ndarray of shape
  (height, width, 3).
- `metadata`: a dictionary of metadata written in the spectral image. This
  is likely to be an empty dictionary in this dataset.
- `masks`: dictionary of segmentation label - segmentation mask pairs.

## Usage example 2
Let's render the vector graphics format segmentation file for spectral image
P001. The segmentation elements can be found in file "P001.csv".

```python
from pathlib import Path
import annotations

# Read the vector graphics format file
vector_text = Path("P001.csv").read_text("utf-8")

# Parse the contents of the vector graphics format file.
# The file may contain arbitrary number of polygons and
# lines.
vector_elements = annotations.parse(vector_text)

# Render vector elements into bitmap rasters. The method
# needs to know the size of the bitmap - all spectral images
# in the dataset have spatial resolution of 1024x1024.
bitmap_elements = annotations.create_masks(vector_elements, 1024, 1024)

# Combine bitmap elements by their labels into mask images.
# The result is a dictionary of type dict[str: ndarray], i.e.,
# each segmentation label maps to one segmentation mask.
masks = annotations.flatten_masks(bitmap_elements)
```

The code snippet above creates the same dictionary as `read_mtiff("P001, masks.tif")` would create in the previous
example.

## Usage example 3
The bitmap raster segmentation mask files have a preview image that shows all
segmentation masks overlaid on an RGB render of the associated spectral image.
A method for creating these pretty images is also included in "annotations.py".
Let's recreate "P001, masks.tif":

```python
from pathlib import Path
import annotations
from spectral_tiffs import read_stiff, write_mtiff

# Repeat example 2
vector_text = Path("P001.csv").read_text("utf-8")
vector_elements = annotations.parse(vector_text)
bitmap_elements = annotations.create_masks(vector_elements, 1024, 1024)
masks = annotations.flatten_masks(bitmap_elements)

# Load a preview image of P001
_, _, preview_image, _ = read_stiff("P001.tif", rgb_only=True)

# Set label colors
label_colors = {x[0]: tuple(x[3]) for x in vector_elements}

# Create a masked preview image
mask_preview_image = annotations.overlay_rgb_mask(preview_image, masks, label_colors,
                                                  legend=True, legend_title="Placenta-DB")

# Save the masks using a preview image
write_mtiff("P001, masks v2.tif", masks, mask_preview_image)
```

## Usage example 4
The raw spectral images as saved by the spectral camera are provided in ENVI-format.
The function for reading ENVI-data is provided in "read_envi.py".

```python
from read_envi import read_envi

data_cube, center_wavelengths, header = read_envi("P001.hdr", normalize=False)
```

This results in several variables in the environment:
- `data_cube` is an ndarray of shape (height, width, band) and contains the
   spectral image.
- `center_wavelengths` is a vector of spectral band center wavelengths.
- `header` is a dictionary representation of the ENVI header file

## Usage example 5
The "read_envi.py" also contains a function that makes it unnecessary to
extract the "Raw measurement data.zip".

```python
from zipfile import ZipFile
from read_envi import read_zipped_envi

with ZipFile("Raw measurement data.zip", "r") as zip_file:
  data_cube, center_wavelengths, header = read_zipped_envi(zip_file, "P001.hdr", normalize=False)
```
