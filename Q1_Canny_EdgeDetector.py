import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img_path = os.path.join(os.path.dirname(__file__), 'Cropped segment.png')
img = cv2.imread(img_path, 0)

if img is None:
    print(f"Error: Could not load image from {img_path}")
else:
    # Apply Canny Edge Detector
    edges = cv2.Canny(img, 400, 500)
    # Extract x and y coordinates
    indices = np.where(edges != 0)
    x = indices[1]
    y = indices[0]
    # Plot original and edge image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Cropped Image')
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title('Canny Edges')
    plt.savefig('Q1_Canny_Edges.png', dpi=100, bbox_inches='tight')
    plt.show()
