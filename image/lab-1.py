import cv2, numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from skimage import img_as_float
from skimage.util import random_noise

# Load & normalize
img = img_as_float(cv2.imread('image_sample.png', 0))

# Create noisy versions
noises = [
    ('Gaussian Noise',      random_noise(img, mode='gaussian', var=0.01)),
    ('Salt & Pepper Noise', random_noise(img, mode='s&p',    amount=0.05))
]

# Define filters
filters = [
    ('Mean Filter',     lambda x: cv2.blur(x, (3,3))),
    ('Median Filter',   lambda x: median_filter(x, size=3)),
    ('Gaussian Filter', lambda x: cv2.GaussianBlur(x, (3,3), 0.5))
]

# Build list of (title, image)
results = [('Original Image', img)]
for title, noisy in noises:
    results.append((title, noisy))
for f_title, fn in filters:
    for n_title, noisy in noises:
        short = n_title.split()[0]
        results.append((f"{f_title} ({short})", fn(noisy)))

# Plot 3×3 grid with smaller figure & constrained layout
fig, axes = plt.subplots(3, 3, figsize=(8, 8), dpi=100, constrained_layout=True)
for ax, (title, im) in zip(axes.ravel(), results):
    ax.imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.show()


# python(3.10.0)
# pip install opencv-python numpy matplotlib scipy scikit-image
