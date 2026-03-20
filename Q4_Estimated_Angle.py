import cv2 as cv, numpy as np
e = cv.Canny(cv.imread('Cropped segment.png', 0), 550, 690)
y, x = np.where(e != 0)
m, b = np.polyfit(x, y, 1)
print(f'Q4 angle = {np.degrees(np.arctan(m)):.2f} deg')
