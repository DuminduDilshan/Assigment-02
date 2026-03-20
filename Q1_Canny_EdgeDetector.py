import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv.imread('Cropped segment.png', cv.IMREAD_GRAYSCALE)

# Check image loaded correctly
if img is None:
	raise FileNotFoundError("Image not found. Make sure 'Cropped segment.png' is in the correct folder.")

# Apply Canny edge detector
edges = cv.Canny(img, 300, 440)

# Show original and edge image
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Image')
plt.axis('off')

plt.tight_layout()
plt.show()