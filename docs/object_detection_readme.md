# objectDetectioncodes.docx → Implementation Map

This README documents how the contents of the provided document "objectDetectioncodes.docx" are implemented in this repository, and how to run them locally.

## Covered topics
- SIFT Keypoint Detection and Visualization
- Haar Cascade Face Detection (with parameter notes for detectMultiScale)
- HOG (Histogram of Oriented Gradients) features with visualization

## Related repository files
- `sift_feature_detection.py` — Detects SIFT keypoints, draws rich keypoints, shows count and descriptor shape.
- `haar_cascade_face_detection.py` — Loads Haar cascade, converts to grayscale, runs `detectMultiScale(scaleFactor, minNeighbors, minSize)`, draws rectangles, parameter explainer included.
- `hog_feature_descriptor.py` — Uses `skimage.feature.hog` with orientations/pixels_per_cell/cells_per_block; includes visualization and parameter comparison helper.

## How to run
- SIFT:
  ```bash
  python sift_feature_detection.py
  ```
  Edit `image_path` inside the script (e.g., `chair.jpg`).

- Haar Cascade:
  ```bash
  python haar_cascade_face_detection.py
  ```
  Edit `image_path` (e.g., `teamindia.jpg`). Ensure OpenCV haarcascades are available via `cv2.data.haarcascades`.

- HOG:
  ```bash
  python hog_feature_descriptor.py
  ```
  By default uses `skimage.data.astronaut()`; switch to a custom image by setting `use_sample_image=False` and providing `image_path`.

## Notes
- The original document uses Windows absolute paths (e.g., `D:/Sugumar/...`). Scripts here default to simpler filenames; update paths as needed.
- For Haar, typical `scaleFactor` is 1.01–1.3. Higher `minNeighbors` reduces false positives at the cost of recall. `minSize` ignores very small detections.

## Dependencies
- Python 3.8+
- OpenCV (cv2), NumPy, Matplotlib, scikit-image (for HOG)
