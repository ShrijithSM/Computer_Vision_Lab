import cv2
import sys

def haar_cascade_face_detection(image_path, save_output=False, output_path="face_detection_result.jpg"):
    """
    Detect faces in an image using Haar Cascade classifier
    
    Args:
        image_path (str): Path to the input image
        save_output (bool): Whether to save the result image
        output_path (str): Path to save the output image
    
    Returns:
        tuple: (number_of_faces, processed_image)
    """
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the input image
    img = cv2.imread(image_path)
    
    if img is None:
        raise IOError(f"Could not load image from {image_path}. Please check the file path.")
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,      # How much the image size is reduced at each scale
        minNeighbors=4,       # How many neighbors each candidate should have to retain it
        minSize=(20, 20)      # Minimum possible object size, smaller objects are ignored
    )
    
    # Draw rectangles around the detected faces
    result_img = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Add face number labels
        cv2.putText(result_img, f'Face {len([f for f in faces if f[1] <= y])}', 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Print detection results
    print(f"Number of faces detected: {len(faces)}")
    for i, (x, y, w, h) in enumerate(faces, 1):
        print(f"Face {i}: Position({x}, {y}), Size({w}x{h})")
    
    # Save output if requested
    if save_output:
        cv2.imwrite(output_path, result_img)
        print(f"Result saved to: {output_path}")
    
    # Display the result
    cv2.imshow('Haar Cascade Face Detection', result_img)
    print("Press any key to close the display window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return len(faces), result_img

def explain_parameters():
    """
    Explain the detectMultiScale() parameters
    """
    print("\n=== Haar Cascade detectMultiScale() Parameters ===")
    print("1. image: Input grayscale image for detection")
    print("2. scaleFactor: Image size reduction factor (1.01 to 1.3)")
    print("   - Lower values = more detection time but higher accuracy")
    print("3. minNeighbors: Required neighbor detections to retain candidate")
    print("   - Higher values = fewer detections but higher quality")
    print("4. minSize: Minimum object size (width, height)")
    print("   - Objects smaller than this are ignored")
    print("=" * 50)

if __name__ == "__main__":
    # Example usage
    image_path = "teamindia.jpg"  # Update this path as needed
    
    try:
        # Explain parameters first
        explain_parameters()
        
        # Perform face detection
        num_faces, processed_img = haar_cascade_face_detection(
            image_path, 
            save_output=True, 
            output_path="detected_faces.jpg"
        )
        
        print(f"\nDetection completed! Found {num_faces} face(s) in the image.")
        
    except IOError as e:
        print(f"Error: {e}")
        print("Please update the image_path variable with a valid image file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
