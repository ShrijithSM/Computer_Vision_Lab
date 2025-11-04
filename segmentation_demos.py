import cv2
import matplotlib.pyplot as plt
import numpy as np

def thresholding_demo(image_path='image.jpeg'):
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Original Image", image)
    cv2.imshow("Binary Threshold", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def thresholding_types_demo(image_path=r'E:/JAIN/CVIP/grayscale/cat.jpg'):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Image not found: {image_path}")
    _, th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    _, th2 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    _, th3 = cv2.threshold(img, 50, 255, cv2.THRESH_TRUNC)
    _, th4 = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)
    _, th5 = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO_INV)
    titles = ['Original', 'Binary', 'Binary_INV', 'Trunc', 'ToZero', 'ToZero_INV']
    images = [img, th1, th2, th3, th4, th5]
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def otsu_threshold_demo(image_path=r'E:/JAIN/CVIP/grayscale/cat.jpg'):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise IOError(f"Image not found: {image_path}")
    _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(otsu_thresh, cmap='gray')
    plt.title('Otsu Thresholding')
    plt.axis('off')
    plt.show()


def otsu_optimal_value_demo(image_path=r'E:/JAIN/CVIP/grayscale/cat.jpg'):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise IOError(f"Image not found: {image_path}")
    val, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Optimal Threshold value:", val)
    plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(1,2,2), plt.imshow(otsu, cmap='gray'), plt.title('Otsu Thresholded')
    plt.show()


def canny_hough_demo(image_path='image.jpeg'):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Image not found: {image_path}")
    blur = cv2.GaussianBlur(img, (5,5), 1.2)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for l in lines:
            rho,theta = l[0]
            a,b = np.cos(theta), np.sin(theta)
            x0,y0 = a*rho, b*rho
            pt1 = (int(x0+1000*(-b)), int(y0+1000*(a)))
            pt2 = (int(x0-1000*(-b)), int(y0-1000*(a)))
            cv2.line(color, pt1, pt2, (0,0,255), 2)
    plt.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    plt.title("Canny + Hough Line Detection")
    plt.axis('off'); plt.show()


if __name__ == '__main__':
    # Quick smoke tests (update paths as needed)
    # thresholding_demo('image.jpeg')
    # thresholding_types_demo(r'E:/JAIN/CVIP/grayscale/cat.jpg')
    # otsu_threshold_demo(r'E:/JAIN/CVIP/grayscale/cat.jpg')
    # otsu_optimal_value_demo(r'E:/JAIN/CVIP/grayscale/cat.jpg')
    # canny_hough_demo('image.jpeg')
    pass
