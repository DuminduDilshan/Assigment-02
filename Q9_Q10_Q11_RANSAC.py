import cv2, numpy as np, matplotlib.pyplot as plt, os
from sklearn.linear_model import RANSACRegressor
img = cv2.imread(os.path.join(os.path.dirname(__file__), 'Cropped segment.png'), 0)
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
edges = cv2.Canny(img, 550, 690)
indices = np.where(edges != 0)
x, y = indices[1], indices[0]
ransac = RANSACRegressor(random_state=42)
ransac.fit(x.reshape(-1,1), y)
m = ransac.estimator_.coef_[0]
theta = np.degrees(np.arctan(m))
plt.figure(figsize=(14, 10), dpi=100)
inlier = ransac.inlier_mask_
plt.scatter(x[inlier], y[inlier], s=5, alpha=0.5, label='Inliers')
plt.scatter(x[~inlier], y[~inlier], s=5, alpha=0.3, label='Outliers')
x_line = np.array([x.min(), x.max()])
plt.plot(x_line, m*x_line+ransac.estimator_.intercept_, 'g-', linewidth=3)
plt.gca().invert_yaxis(); plt.title(f'Q9-Q11: RANSAC θ={theta:.2f}°', fontsize=14)
plt.legend(); plt.savefig('Q9_Q10_Q11_RANSAC.png', dpi=150, bbox_inches='tight'); plt.show()
print(f'Slope:{m:.6f} θ:{theta:.2f}°')
