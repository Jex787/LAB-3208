# Object Counting and Measurement in ‘rice.tif’ Image

import cv2, numpy as np, matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

# Load & binarize+clean
img = cv2.imread('rice.tiff', 0).astype(float)/255
bw = ~(img > threshold_local(img, block_size=35, method='gaussian'))
bw = remove_small_objects(bw, min_size=50)

# Measure regions
props = regionprops(label(bw))
areas  = [p.area for p in props]
cent   = [p.centroid for p in props]
major  = [p.major_axis_length for p in props]
peri   = [p.perimeter for p in props]

# 1) Overlay labels on the original
plt.imshow(img, cmap='gray')
for i, (y, x) in enumerate(cent, 1):
    plt.text(x, y, str(i), color='r', fontsize=6)
plt.title('Rice Grains'); plt.axis('off')
plt.show()

# 2) Print table & stats
print(f"Total grains: {len(props)}\n")
print(f"{'No.':<5}{'Area':<8}{'MajorAxisLen':<15}{'Perimeter':<10}")
for i, (a, M, P) in enumerate(zip(areas, major, peri), 1):
    print(f"{i:<5}{a:<8.1f}{M:<15.1f}{P:<10.1f}")

print(f"\nMin: {min(areas):.1f}, Max: {max(areas):.1f}, Mean: {np.mean(areas):.1f}")
print(f"Grains in area [200–400]: {sum(200 <= a <= 400 for a in areas)}")

# python (3.10.0)
# pip install opencv-python numpy matplotlib scikit-image
