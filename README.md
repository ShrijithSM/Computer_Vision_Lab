
# Computer_Vision_Lab

Repository mapping the provided lab documents to runnable Python scripts and step-by-step notebooks.

## objectDetectioncodes.docx
Related files:
- `sift_feature_detection.py`
- `haar_cascade_face_detection.py`
- `hog_feature_descriptor.py`

### objectDetectioncodes.docx coverage details
- SIFT keypoint detection and visualization (detectAndCompute, drawKeypoints)
- Haar Cascade face detection with `detectMultiScale(scaleFactor, minNeighbors, minSize)` and parameter guidance
- HOG features with visualization and parameter comparison (orientations, pixels_per_cell, cells_per_block)

## codes.docx (Filtering, Warping, Morphing)
Related files:
- `image_filtering_comparison.ipynb` (Mean, Box, Gaussian, comparisons)
- `affine_warping.py`
- `image_morphing.py`

### codes.docx coverage details
- Mean and Box filtering via `cv2.blur`
- Gaussian filtering: normalized 3×3 kernel with `cv2.filter2D` and `cv2.GaussianBlur`
- Affine image warping using 3-point `cv2.getAffineTransform` + `cv2.warpAffine`
- Image morphing (cross‑dissolve) using `cv2.addWeighted` with adjustable steps and optional GIF export

## segmentationcodes.docx
Related files:
- `segmentation_demos.py` (binary thresholding, thresholding types, Otsu, Canny + Hough)
- `edge_relaxation.py` (iterative relaxation on Sobel edge magnitude)

### segmentationcodes.docx coverage details
- Binary thresholding and multiple threshold variants
- Otsu’s method (mask + optimal value)
- Canny edges with Hough line detection overlay
- Edge relaxation: Sobel magnitude → probability field → iterative neighborhood relaxation → thresholded edges

## image_enhancement.docx
Related files:
- `image_enhancement_suite.py`

### image_enhancement.docx coverage details
- Negative, Log, Power-law (Gamma), Histogram Equalization, Contrast Stretching
- Mean and Gaussian filtering, Laplacian edges, Sobel magnitude with normalization
- Script function: `image_enhancement_suite(image_path, resize_to, gamma, show, save_grid)`
  - Use `show=True` to view a 3×4 grid; set `save_grid=True` to export to `enhancement_grid.png`

---

### How to run
- Scripts: `python file_name.py` (edit image paths inside as needed)
- Notebooks: open in Jupyter/VS Code and run cells step-by-step

### Requirements
- Python 3.8+
- OpenCV (cv2), NumPy, Matplotlib, scikit-image (for HOG), imageio (optional for GIF in morphing)

### Notes
- Example image paths in the original docs (e.g., `E:/JAIN/...`, `D:/Sugumar/...`) are kept in comments or defaults; update them to local paths.
- Contributions welcome: feel free to open issues/PRs for additional lab tasks or advanced segmentation pipelines (boundary melting, watershed, quadtree split/merge).
