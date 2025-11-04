import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_enhancement_suite(image_path='lena.jpg', resize_to=(256, 256), gamma=0.5, show=True, save_grid=False, grid_path='enhancement_grid.png'):
    """
    Apply common image enhancement operations and optionally show/save a comparison grid.

    Operations:
      1) Negative
      2) Log transform
      3) Power-law (Gamma)
      4) Histogram Equalization
      5) Contrast Stretching
      6) Mean Filter
      7) Gaussian Filter
      8) Laplacian (edges)
      9) Sobel magnitude (edges)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not load image from {image_path}")

    if resize_to is not None:
        img = cv2.resize(img, resize_to)

    # 1. Negative
    negative = 255 - img

    # 2. Log Transformation
    log_img = np.uint8(255 * (np.log1p(img) / np.log1p(255)))

    # 3. Power-Law (Gamma)
    power_law = np.uint8(255 * ((img / 255.0) ** gamma))

    # 4. Histogram Equalization
    hist_eq = cv2.equalizeHist(img)

    # 5. Contrast Stretching
    min_val, max_val = int(np.min(img)), int(np.max(img))
    contrast_stretch = np.uint8((img - min_val) * 255 / max(1, (max_val - min_val)))

    # 6. Mean Filtering
    mean_filter = cv2.blur(img, (5, 5))

    # 7. Gaussian Filtering
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    # 8. Laplacian Filtering
    lap = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(lap))

    # 9. Sobel magnitude
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8((mag / (np.max(mag) + 1e-8)) * 255)

    titles = [
        'Original', 'Negative', 'Log', f'Gamma (γ={gamma})',
        'Hist Eq', 'Contrast Stretch', 'Mean', 'Gaussian',
        'Laplacian', 'Sobel Mag'
    ]
    images = [
        img, negative, log_img, power_law,
        hist_eq, contrast_stretch, mean_filter, gaussian,
        laplacian, sobel
    ]

    if show or save_grid:
        plt.figure(figsize=(15, 10))
        for i, (im, title) in enumerate(zip(images, titles)):
            plt.subplot(3, 4, i + 1)
            plt.imshow(im, cmap='gray')
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        if save_grid:
            plt.savefig(grid_path, dpi=150)
            print(f"Saved grid to {grid_path}")
        if show:
            plt.show()
        else:
            plt.close()

    return {
        'titles': titles,
        'images': images
    }

if __name__ == '__main__':
    # Example run
    image_enhancement_suite('lena.jpg', gamma=0.5, show=True, save_grid=False)
