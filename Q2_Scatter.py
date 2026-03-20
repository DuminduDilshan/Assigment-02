import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv.imread(os.path.join(os.path.dirname(__file__), 'Cropped segment.png'), cv.IMREAD_GRAYSCALE)
img = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST)
edges = cv.Canny(img, 550, 690)

indices = np.where(edges != 0)
x, y = indices[1], indices[0]

plt.figure(figsize=(10, 7))
plt.scatter(x, y, s=5)
plt.gca().invert_yaxis()
plt.title('Q2: Scatter Plot of Edge Points')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
