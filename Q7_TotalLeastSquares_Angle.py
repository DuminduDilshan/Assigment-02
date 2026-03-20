import cv2 as cv, numpy as np

img = cv.imread('Cropped segment.png', 0)
e = cv.Canny(img, 550, 690)
y, x = np.where(e != 0)

xm, ym = np.mean(x), np.mean(y)
_, _, vt = np.linalg.svd(np.column_stack([x - xm, y - ym]), full_matrices=False)
m = vt[0][1] / vt[0][0]

print(f'Q7 angle = {np.degrees(np.arctan(m)):.2f} deg')
