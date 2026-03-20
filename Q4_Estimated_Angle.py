import cv2 as cv
import numpy as np
import os

img = cv.imread(os.path.join(os.path.dirname(__file__), 'Cropped segment.png'), cv.IMREAD_GRAYSCALE)
img = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST)
edges = cv.Canny(img, 550, 690)

indices = np.where(edges != 0)
x, y = indices[1], indices[0]

A = np.vstack([x, np.ones(len(x))]).T
m, b = np.linalg.lstsq(A, y, rcond=None)[0]
theta = np.degrees(np.arctan(m))

print(f'Q4: Estimated crop field angle = {theta:.2f} degrees')
