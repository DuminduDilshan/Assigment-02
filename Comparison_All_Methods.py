import cv2 as cv, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
e = cv.Canny(cv.imread('Cropped segment.png', 0), 550, 690)
y, x = np.where(e != 0); xl = np.array([x.min(), x.max()])
m1, b1 = np.polyfit(x, y, 1)
xm, ym = np.mean(x), np.mean(y); _, _, vt = np.linalg.svd(np.column_stack([x - xm, y - ym]), full_matrices=False)
m2 = vt[0][1] / vt[0][0]; b2 = ym - m2 * xm
r = RANSACRegressor(random_state=42).fit(x.reshape(-1, 1), y); m3, b3 = r.estimator_.coef_[0], r.estimator_.intercept_
fig, a = plt.subplots(1, 3, figsize=(14, 4))
a[0].scatter(x, y, s=1, alpha=0.3); a[0].plot(xl, m1 * xl + b1, 'r-'); a[0].invert_yaxis(); a[0].set_title('LS')
a[1].scatter(x, y, s=1, alpha=0.3); a[1].plot(xl, m2 * xl + b2, 'b-'); a[1].invert_yaxis(); a[1].set_title('TLS')
inl = r.inlier_mask_; a[2].scatter(x[inl], y[inl], s=1); a[2].scatter(x[~inl], y[~inl], s=1, alpha=0.3)
a[2].plot(xl, m3 * xl + b3, 'g-'); a[2].invert_yaxis(); a[2].set_title('RANSAC')
plt.tight_layout(); plt.show()
