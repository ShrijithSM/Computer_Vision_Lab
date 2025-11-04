import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Edge relaxation and visualization

def edge_relaxation(image_path="D:/Sugumar/actor.jpeg", lam=0.5, iters=12, thresh=0.5):
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError("Could not load image. Please check the path!")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray, (5,5), 1.2)
    gx = cv.Sobel(blur, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(blur, cv.CV_32F, 0, 1, ksize=3)
    mag = cv.magnitude(gx, gy)
    hi = np.percentile(mag, 99)
    P0 = np.clip(mag / (hi + 1e-6), 0, 1).astype(np.float32)

    kernel = np.array([[0.5,1.0,0.5],
                       [1.0,0.0,1.0],
                       [0.5,1.0,0.5]], dtype=np.float32)
    kernel /= kernel.sum()

    def relax(P, lam=0.5, iters=10):
        Pk = P.copy()
        for _ in range(iters):
            mean_nb = cv.filter2D(Pk, -1, kernel, borderType=cv.BORDER_REPLICATE)
            Pk = np.clip(Pk + lam*(mean_nb - Pk), 0, 1)
        return Pk

    final_P = relax(P0, lam=lam, iters=iters)
    edge_relaxed = (final_P > thresh).astype(np.uint8)*255

    plt.figure(figsize=(12,8))
    titles = ['Input', 'Sobel magnitude', 'Relaxed probability', 'Thresholded relaxed edges']
    images = [cv.cvtColor(img, cv.COLOR_BGR2RGB),
              cv.normalize(mag,None,0,255,cv.NORM_MINMAX).astype(np.uint8),
              (final_P*255).astype(np.uint8),
              edge_relaxed]

    for i,(t,im) in enumerate(zip(titles,images)):
        plt.subplot(2,2,i+1)
        plt.imshow(im, cmap='gray')
        plt.title(t)
        plt.axis('off')
    plt.tight_layout(); plt.show()

    return final_P, edge_relaxed

if __name__ == "__main__":
    edge_relaxation()
