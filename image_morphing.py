import cv2
import numpy as np

def image_morphing(source_path, dest_path, steps=100, size=(300, 300), delay_ms=100, save_gif=False, gif_path="morph.gif"):
    """
    Cross-dissolve morph between two images using cv2.addWeighted.

    Args:
        source_path (str): Path to source image
        dest_path (str): Path to destination image
        steps (int): Number of intermediate blends
        size (tuple): Resize both images to this (w,h)
        delay_ms (int): Delay between frames if displaying
        save_gif (bool): If True, export frames as a GIF (requires imageio)
        gif_path (str): Output GIF path
    """
    src = cv2.imread(source_path)
    dst = cv2.imread(dest_path)

    if src is None:
        raise IOError(f"Could not load source image: {source_path}")
    if dst is None:
        raise IOError(f"Could not load destination image: {dest_path}")

    src = cv2.resize(src, size)
    dst = cv2.resize(dst, size)

    frames = []
    for i in range(steps + 1):
        alpha = i / float(steps)
        blended = cv2.addWeighted(src, 1 - alpha, dst, alpha, 0)
        cv2.imshow("Morphing", blended)
        cv2.waitKey(delay_ms)
        frames.append(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

    # Wait for key before closing
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_gif:
        try:
            import imageio
            imageio.mimsave(gif_path, frames, fps=max(1, 1000 // max(1, delay_ms)))
            print(f"Saved morph GIF to: {gif_path}")
        except Exception as e:
            print(f"Could not save GIF: {e}. Install imageio to enable GIF export.")

if __name__ == "__main__":
    # Example usage (update paths)
    image_morphing(
        source_path="image1.png",
        dest_path="image2.png",
        steps=100,
        size=(300, 300),
        delay_ms=50,
        save_gif=False,
    )
