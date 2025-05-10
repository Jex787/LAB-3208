# Character Segmentation from an Image

import cv2, numpy as np, matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

# --- Load & binarize ---
img = cv2.imread('sample_image.jpg')
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bw = ~(g > threshold_local(g, 35, method='gaussian', offset=10))
clean = remove_small_objects(bw, min_size=50)

# --- Find character regions sorted left-to-right ---
props = sorted(regionprops(label(clean)), key=lambda p: p.bbox[1])
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- 2×2 overview with bounding boxes ---
fig, axs = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
imgs = [rgb, ~bw, clean, rgb]
titles = ['Original Image', 'Binarized Image', 'Cleaned Image', 'Detected Characters']
for ax, im, title in zip(axs.flatten(), imgs, titles):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

# draw boxes on the last subplot
for p in props:
    y0, x0, y1, x1 = p.bbox
    axs[1,1].add_patch(plt.Rectangle((x0,y0), x1-x0, y1-y0,
                                     fill=False, edgecolor='r'))
plt.show()

# --- grid of each character crop ---
n = len(props)
cols = 5
rows = int(np.ceil(n/cols))
fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3), constrained_layout=True)
for i, p in enumerate(props):
    ax = axs.flatten()[i]
    ax.imshow(p.image, cmap='gray')
    ax.set_title(str(i+1))
    ax.axis('off')
for ax in axs.flatten()[n:]:
    ax.axis('off')
plt.show()

print(f'Total characters detected: {n}')


# python(3.10.0)
# pip install opencv-python numpy matplotlib scikit-image

