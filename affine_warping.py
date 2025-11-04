import cv2
import numpy as np
import matplotlib.pyplot as plt


def affine_warp(image_path, src_points=None, dst_points=None, show=True, save_output=False, output_path='warped_output.jpg'):
    """
    Apply affine transformation (warping) to an image using 3-point mapping.

    Args:
        image_path (str): Path to the input image.
        src_points (np.ndarray): 3x2 float32 array of source points.
        dst_points (np.ndarray): 3x2 float32 array of destination points.
        show (bool): Whether to display the result using matplotlib.
        save_output (bool): Whether to save the warped image to disk.
        output_path (str): Path to save the warped image.

    Returns:
        np.ndarray: Warped image.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Could not load image from {image_path}. Please check the file path.")

    rows, cols = img.shape[:2]

    # Default points (example) if not provided
    if src_points is None:
        src_points = np.float32([[50, 50], [200, 50], [50, 200]])
    if dst_points is None:
        dst_points = np.float32([[60, 70], [220, 50], [70, 250]])

    # Ensure correct dtype and shape
    src_points = np.array(src_points, dtype=np.float32).reshape(3, 2)
    dst_points = np.array(dst_points, dtype=np.float32).reshape(3, 2)

    # Get affine matrix
    M = cv2.getAffineTransform(src_points, dst_points)

    # Apply warping
    warped = cv2.warpAffine(img, M, (cols, rows))

    # Save if requested
    if save_output:
        cv2.imwrite(output_path, warped)
        print(f"Saved warped image to: {output_path}")

    # Show if requested
    if show:
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.title('Affine Warped')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return warped


if __name__ == '__main__':
    # Example usage
    try:
        affine_warp('photo.jpg', show=True, save_output=True)
    except Exception as e:
        print(f"Error: {e}")
