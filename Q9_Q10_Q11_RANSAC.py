import cv2 as cv, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
e = cv.Canny(cv.imread('Cropped segment.png', 0), 550, 690)
y, x = np.where(e != 0)
r = RANSACRegressor(random_state=42).fit(x.reshape(-1, 1), y)
m, b = r.estimator_.coef_[0], r.estimator_.intercept_; xl = np.array([x.min(), x.max()])
inl = r.inlier_mask_
plt.scatter(x[inl], y[inl], s=2, alpha=0.5); plt.scatter(x[~inl], y[~inl], s=2, alpha=0.3)
plt.plot(xl, m * xl + b, 'g-', lw=2); plt.gca().invert_yaxis(); plt.title('Q9-Q11 RANSAC')
ang = np.degrees(np.arctan(m))
txt = f"m={m:.4f}\nb={b:.2f}\nangle={ang:.2f} deg"
plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes, va='top',
         bbox=dict(boxstyle='round', fc='white', ec='black', alpha=0.8))
plt.show()
print(f'Q11 angle = {ang:.2f} deg')
