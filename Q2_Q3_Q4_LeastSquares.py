import cv2 as cv, numpy as np, matplotlib.pyplot as plt
e = cv.Canny(cv.imread('Cropped segment.png', 0), 550, 690)
y, x = np.where(e != 0)
plt.scatter(x, y, s=2); plt.gca().invert_yaxis(); plt.title('Q2'); plt.show()
m, b = np.polyfit(x, y, 1); xl = np.array([x.min(), x.max()])
plt.scatter(x, y, s=1, alpha=0.3); plt.plot(xl, m * xl + b, 'r-', lw=2)
plt.gca().invert_yaxis(); plt.title('Q3-Q4'); plt.show()
print(f'Q4 angle = {np.degrees(np.arctan(m)):.2f} deg')
