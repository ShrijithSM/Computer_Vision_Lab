# codes.docx → Implementation Map

This README documents how the contents of the provided document "codes.docx" are implemented in this repository, and how to run them locally.

## Covered topics
- Mean Filter (cv2.blur)
- Box Filtering (cv2.blur)
- Gaussian Filtering (custom 3×3 kernel + cv2.filter2D, plus cv2.GaussianBlur variants)
- Affine Image Warping (cv2.getAffineTransform, cv2.warpAffine)
- Image Morphing / Cross-dissolve (cv2.addWeighted loop)

## Related repository files
- Notebook: `image_filtering_comparison.ipynb` — Mean, Box, Gaussian filters with side-by-side comparisons and matrix prints.
- Script: `affine_warping.py` — 3-point affine transform with optional display and save.
- Script: `image_morphing.py` — cross-dissolve between two images with adjustable steps and optional GIF export.

## How to run
- Notebook: Open `image_filtering_comparison.ipynb` in Jupyter/VS Code and run cells sequentially. Update image path `first.jpg` as needed.
- Warping script:
  ```bash
  python affine_warping.py
  ```
  Edit `image_path` or pass points to `affine_warp()` inside the script if desired.
- Morphing script:
  ```bash
  python image_morphing.py
  ```
  Update `source_path` and `dest_path` to local image files.

## Notes
- The original document paths (e.g., `E:/JAIN/...`, `D:/Sugumar/...`) are preserved conceptually; in this repo, defaults are set to simpler filenames. Replace paths with your local files.
- Gaussian filtering is demonstrated both via a normalized 3×3 kernel (`cv2.filter2D`) and with `cv2.GaussianBlur` for practical usage.

## Dependencies
- Python 3.8+
- OpenCV (cv2), NumPy, Matplotlib
- (Optional) imageio for exporting morphing GIFs
