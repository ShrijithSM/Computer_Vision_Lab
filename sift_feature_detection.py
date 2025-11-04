import cv2
import matplotlib.pyplot as plt

def sift_feature_detection(image_path):
    """
    Detect and visualize SIFT keypoints in an image
    
    Args:
        image_path (str): Path to the input image
    """
    # Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise IOError("Image not loaded. Please check the image path.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(f"SIFT Keypoints Detection - {len(keypoints)} keypoints found")
    plt.axis("off")
    plt.show()
    
    print(f"Number of keypoints detected: {len(keypoints)}")
    print(f"Descriptor shape: {descriptors.shape if descriptors is not None else 'None'}")
    
    return keypoints, descriptors

if __name__ == "__main__":
    # Example usage - replace with your image path
    image_path = "chair.jpg"  # Update this path as needed
    try:
        keypoints, descriptors = sift_feature_detection(image_path)
    except IOError as e:
        print(f"Error: {e}")
        print("Please update the image_path variable with a valid image file path.")