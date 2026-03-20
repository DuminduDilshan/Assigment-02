import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv.imread(os.path.join(os.path.dirname(__file__), 'Cropped segment.png'), cv.IMREAD_GRAYSCALE)
img = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST)
edges = cv.Canny(img, 550, 690)

indices = np.where(edges != 0)
x, y = indices[1], indices[0]

A = np.vstack([x, np.ones(len(x))]).T
m, b = np.linalg.lstsq(A, y, rcond=None)[0]

x_line = np.array([x.min(), x.max()])
y_line = m * x_line + b

plt.figure(figsize=(10, 7))
plt.scatter(x, y, s=5, alpha=0.3)
plt.plot(x_line, y_line, 'r-', linewidth=3)
plt.gca().invert_yaxis()
plt.title('Q3: Least-Squares-Fit Line')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
