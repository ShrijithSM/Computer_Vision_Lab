import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import cv2
import numpy as np

def hog_feature_descriptor(image_path=None, use_sample_image=True, orientations=8, 
                          pixels_per_cell=(16, 16), cells_per_block=(1, 1)):
    """
    Extract and visualize HOG (Histogram of Oriented Gradients) features from an image
    
    Args:
        image_path (str): Path to input image (optional if using sample)
        use_sample_image (bool): Use built-in astronaut image from skimage
        orientations (int): Number of orientation bins
        pixels_per_cell (tuple): Size (in pixels) of a cell
        cells_per_block (tuple): Number of cells in each block
    
    Returns:
        tuple: (hog_features, hog_image, original_image)
    """
    
    # Load image
    if use_sample_image:
        image = data.astronaut()
        print("Using sample astronaut image from skimage.data")
    else:
        if image_path is None:
            raise ValueError("Please provide image_path when use_sample_image=False")
        
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB for proper display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Image shape: {image.shape}")
    
    # Extract HOG features and visualization
    hog_features, hog_image = hog(
        image,
        orientations=orientations,           # Number of orientation bins
        pixels_per_cell=pixels_per_cell,     # Size of a cell
        cells_per_block=cells_per_block,     # Number of cells in each block
        visualize=True,                      # Return HOG image for visualization
        channel_axis=-1,                     # Channel axis for color images
        feature_vector=True                  # Return features as 1D array
    )
    
    # Print HOG parameters and results
    print(f"\n=== HOG Parameters ===")
    print(f"Orientations: {orientations}")
    print(f"Pixels per cell: {pixels_per_cell}")
    print(f"Cells per block: {cells_per_block}")
    print(f"HOG feature vector length: {len(hog_features)}")
    print(f"HOG image shape: {hog_image.shape}")
    
    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    
    # Original image
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray if len(image.shape) == 2 else None)
    ax1.set_title('Original Image', fontsize=14)
    
    # HOG visualization (rescaled for better display)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    return hog_features, hog_image, image

def explain_hog_parameters():
    """
    Explain HOG parameters in detail
    """
    print("\n" + "="*60)
    print("HOG (Histogram of Oriented Gradients) Parameters Explained")
    print("="*60)
    print("1. orientations: Number of orientation bins (typically 8-9)")
    print("   - More bins = more detailed orientation info but higher dimensionality")
    print("\n2. pixels_per_cell: Size of each cell in pixels (e.g., (8,8) or (16,16))")
    print("   - Smaller cells = more detailed features but higher dimensionality")
    print("\n3. cells_per_block: Number of cells per block (e.g., (1,1), (2,2))")
    print("   - Used for local contrast normalization")
    print("   - Larger blocks = more robust to illumination changes")
    print("\n4. visualize: Whether to return HOG image for visualization")
    print("\n5. channel_axis: Which axis represents color channels (-1 for last)")
    print("="*60)

def compare_hog_parameters(image_path=None):
    """
    Compare HOG features with different parameters
    """
    # Load image
    if image_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = data.astronaut()
    
    # Different parameter combinations
    params = [
        {"orientations": 8, "pixels_per_cell": (8, 8), "cells_per_block": (1, 1)},
        {"orientations": 8, "pixels_per_cell": (16, 16), "cells_per_block": (1, 1)},
        {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)},
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Original image
    axes[0, 0].imshow(image, cmap=plt.cm.gray if len(image.shape) == 2 else None)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    for i, param in enumerate(params):
        _, hog_img = hog(image, visualize=True, channel_axis=-1, **param)
        hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
        
        row, col = (0, i+1) if i < 2 else (1, i-2)
        axes[row, col].imshow(hog_img_rescaled, cmap=plt.cm.gray)
        axes[row, col].set_title(f'Orientations: {param["orientations"]}\n'
                                f'Cell: {param["pixels_per_cell"]}\n'
                                f'Block: {param["cells_per_block"]}')
        axes[row, col].axis('off')
    
    # Empty the last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    print("HOG Feature Descriptor Demo")
    print("="*50)
    
    try:
        # Explain parameters first
        explain_hog_parameters()
        
        # Basic HOG extraction with sample image
        print("\n1. Basic HOG extraction with default parameters:")
        features, hog_vis, orig_img = hog_feature_descriptor()
        
        # Compare different parameters
        print("\n2. Comparing different HOG parameters:")
        compare_hog_parameters()
        
        # For custom image, uncomment and provide path:
        # features, hog_vis, orig_img = hog_feature_descriptor(
        #     image_path="your_image.jpg", 
        #     use_sample_image=False
        # )
        
    except Exception as e:
        print(f"Error: {e}")
