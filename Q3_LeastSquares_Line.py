import cv2 as cv, numpy as np, matplotlib.pyplot as plt
e = cv.Canny(cv.imread('Cropped segment.png', 0), 550, 690)
y, x = np.where(e != 0)
m, b = np.polyfit(x, y, 1); xf = np.array([x.min(), x.max()])
plt.scatter(x, y, s=1, c='b', marker='.')
plt.plot(xf, m * xf + b, 'r-', lw=2); plt.gca().invert_yaxis(); plt.title('Q3'); plt.show()