import cv2 as cv, numpy as np, matplotlib.pyplot as plt
e = cv.Canny(cv.imread('Cropped segment.png', 0), 550, 690)
y, x = np.where(e != 0)
plt.scatter(x, y, s=2); plt.gca().invert_yaxis(); plt.title('Q2'); plt.show()
