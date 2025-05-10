# Image Segmentation Using Thresholding Techniques

import cv2
import matplotlib.pyplot as plt

# Read the image in grayscale
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Global Thresholding (Otsu's method)
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 2. Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(image, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        11, 2)

# Show the results
titles = ['Original Image', "Otsu's Thresholding", 'Adaptive Thresholding']
images = [image, otsu_thresh, adaptive_thresh]

plt.figure(figsize=(10, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# python(3.10.0)
# pip install opencv-python matplotlib
