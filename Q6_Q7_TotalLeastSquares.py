import cv2 as cv, numpy as np, matplotlib.pyplot as plt
e = cv.Canny(cv.imread('Cropped segment.png', 0), 550, 690)
y, x = np.where(e != 0)
xm, ym = np.mean(x), np.mean(y)
_, _, vt = np.linalg.svd(np.column_stack([x - xm, y - ym]), full_matrices=False)
m = vt[0][1] / vt[0][0]; b = ym - m * xm; xl = np.array([x.min(), x.max()])
plt.scatter(x, y, s=2, alpha=0.3); plt.plot(xl, m * xl + b, 'b-', lw=2)
plt.gca().invert_yaxis(); plt.title('Q6-Q7 TLS'); plt.show()
print(f'Q7 angle = {np.degrees(np.arctan(m)):.2f} deg')
