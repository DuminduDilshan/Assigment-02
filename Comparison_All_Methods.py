import cv2, numpy as np, matplotlib.pyplot as plt, os
from sklearn.linear_model import RANSACRegressor
img = cv2.imread(os.path.join(os.path.dirname(__file__), 'Cropped segment.png'), 0)
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
edges = cv2.Canny(img, 550, 690)
indices = np.where(edges != 0)
x, y = indices[1], indices[0]
A = np.vstack([x, np.ones(len(x))]).T
m1, b1 = np.linalg.lstsq(A, y, rcond=None)[0]; t1 = np.degrees(np.arctan(m1))
x_m, y_m = np.mean(x), np.mean(y)
pts = np.column_stack([x-x_m, y-y_m]); U, S, Vt = np.linalg.svd(pts, full_matrices=False)
m2 = Vt[0][1]/Vt[0][0]; t2 = np.degrees(np.arctan(m2))
ransac = RANSACRegressor(random_state=42); ransac.fit(x.reshape(-1,1), y)
m3 = ransac.estimator_.coef_[0]; t3 = np.degrees(np.arctan(m3))
fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=100); x_line = np.array([x.min(), x.max()])
axes[0].scatter(x, y, s=5, alpha=0.3); axes[0].plot(x_line, m1*x_line+b1, 'r-', linewidth=3)
axes[0].set_title(f'LS \u03b8={t1:.2f}\u00b0', fontsize=14); axes[0].invert_yaxis()
axes[1].scatter(x, y, s=5, alpha=0.3); axes[1].plot(x_line, m2*x_line+(y_m-m2*x_m), 'b-', linewidth=3)
axes[1].set_title(f'TLS \u03b8={t2:.2f}\u00b0', fontsize=14); axes[1].invert_yaxis()
inlier = ransac.inlier_mask_
axes[2].scatter(x[inlier], y[inlier], s=5, alpha=0.5); axes[2].scatter(x[~inlier], y[~inlier], s=5, alpha=0.3)
axes[2].plot(x_line, m3*x_line+ransac.estimator_.intercept_, 'g-', linewidth=3)
axes[2].set_title(f'RANSAC \u03b8={t3:.2f}\u00b0', fontsize=14); axes[2].invert_yaxis()
plt.tight_layout(); plt.savefig('Comparison.png', dpi=150, bbox_inches='tight'); plt.show()
print(f'LS:{t1:.2f}\u00b0 TLS:{t2:.2f}\u00b0 RANSAC:{t3:.2f}\u00b0')
