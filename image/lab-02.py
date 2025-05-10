# Histogram Equalization for Contrast Enhancement

# # # .........code-01..........


# import cv2
# import matplotlib.pyplot as plt

# # Load low-contrast grayscale image
# img = cv2.imread('low_contrast_image.jpg', cv2.IMREAD_GRAYSCALE)

# # Histogram Equalization
# equalized = cv2.equalizeHist(img)

# # Plot original and equalized images
# plt.figure(figsize=(10, 6))

# plt.subplot(2, 2, 1)
# plt.imshow(img, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(2, 2, 2)
# plt.imshow(equalized, cmap='gray')
# plt.title('Equalized Image')
# plt.axis('off')

# # Plot histograms
# plt.subplot(2, 2, 3)
# plt.hist(img.ravel(), bins=256, range=[0, 256], color='gray')
# plt.title('Original Histogram')

# plt.subplot(2, 2, 4)
# plt.hist(equalized.ravel(), bins=256, range=[0, 256], color='gray')
# plt.title('Equalized Histogram')

# plt.tight_layout()
# plt.show()


# pip install opencv-python matplotlib




# # .......code-2........


# import matplotlib.pyplot as plt
# from skimage import io, exposure, img_as_ubyte
# from skimage.color import rgb2gray
# import numpy as np

# # Load and convert image to grayscale
# img = io.imread('example.jpg')  # Replace with your image path
# gray = rgb2gray(img)
# gray = img_as_ubyte(gray)

# # Global Histogram Equalization
# global_eq = exposure.equalize_hist(gray)
# global_eq_img = img_as_ubyte(global_eq)

# # CLAHE (Adaptive Histogram Equalization)
# clahe_eq = exposure.equalize_adapthist(gray, clip_limit=0.03)
# clahe_eq_img = img_as_ubyte(clahe_eq)

# # Plot all in subplots
# fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# # Top row: Images
# axs[0, 0].imshow(gray, cmap='gray')
# axs[0, 0].set_title("Original Image")
# axs[0, 0].axis('off')

# axs[0, 1].imshow(global_eq, cmap='gray')
# axs[0, 1].set_title("Global Histogram Equalization")
# axs[0, 1].axis('off')

# axs[0, 2].imshow(clahe_eq, cmap='gray')
# axs[0, 2].set_title("CLAHE")
# axs[0, 2].axis('off')

# # Bottom row: Histograms
# axs[1, 0].hist(gray.ravel(), bins=256, color='skyblue')
# axs[1, 0].set_title("Original Histogram")

# axs[1, 1].hist(global_eq_img.ravel(), bins=256, color='skyblue')
# axs[1, 1].set_title("Global HE Hist")

# axs[1, 2].hist(clahe_eq_img.ravel(), bins=256, color='skyblue')
# axs[1, 2].set_title("CLAHE Histogram")

# plt.tight_layout()
# plt.show()


# pip install matplotlib scikit-image numpy
# python(3.10.0)