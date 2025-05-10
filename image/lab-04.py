# Edge Detection Using Sobel, Prewitt, and Canny Operators

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Sobel Edge Detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

# 2. Prewitt Edge Detection (manual convolution)
kernelx = np.array([[ -1, 0, 1], [ -1, 0, 1], [ -1, 0, 1]])
kernely = np.array([[ -1, -1, -1], [ 0, 0, 0], [ 1, 1, 1]])
prewitt_x = cv2.filter2D(image, -1, kernelx)
prewitt_y = cv2.filter2D(image, -1, kernely)
prewitt = cv2.magnitude(prewitt_x.astype(np.float32), prewitt_y.astype(np.float32))

# 3. Canny Edge Detection
canny = cv2.Canny(image, 100, 200)

# Plotting
titles = ['Original', 'Sobel', 'Prewitt', 'Canny']
images = [image, sobel, prewitt, canny]

plt.figure(figsize=(12, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# python(3.10.0)
# pip install opencv-python numpy matplotlib
