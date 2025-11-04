# Computer_Vision_Lab

Repository mapping the provided lab documents to runnable Python scripts and step-by-step notebooks.

## objectDetectioncodes.docx
Related files:
- `sift_feature_detection.py`
- `haar_cascade_face_detection.py`
- `hog_feature_descriptor.py`

## codes.docx (Filtering, Warping, Morphing)
Related files:
- `image_filtering_comparison.ipynb` (Mean, Box, Gaussian, comparisons)
- `affine_warping.py`
- `image_morphing.py`

## segmentationcodes.docx
Related files:
- `segmentation_demos.py` (binary thresholding, thresholding types, Otsu, Canny + Hough)
- `edge_relaxation.py` (iterative relaxation on Sobel edge magnitude)

## image_enhancement.docx
Related files:
- `image_enhancement_suite.py`

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
