# Image Convolution Using a 3x3 Mask

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Define a 3x3 sharpening kernel
kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]])

# Apply convolution using filter2D
convolved_image = cv2.filter2D(image, -1, kernel)

# Display original and convolved images
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(convolved_image, cmap='gray')
plt.title('Convolved Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# python(3.10.0)
# pip install opencv-python numpy matplotlib

