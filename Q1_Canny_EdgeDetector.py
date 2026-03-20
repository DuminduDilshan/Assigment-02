import cv2 as cv, matplotlib.pyplot as plt
img = cv.imread('Cropped segment.png', 0)
edges = cv.Canny(img, 550, 690)
plt.subplot(121); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(122); plt.imshow(edges, cmap='gray'); plt.title('Canny'); plt.axis('off')
plt.tight_layout(); plt.show()