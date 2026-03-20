import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

img_path = os.path.join(os.path.dirname(__file__), 'Cropped segment.png')
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
if img is None:
	raise FileNotFoundError(f"Image not found: {img_path}")

edges = cv.Canny(img, 550, 690)
y, x = np.where(edges != 0)
if len(x) == 0:
	raise ValueError('No edge points detected.')

# Least Squares
m_ls, b_ls = np.polyfit(x, y, 1)
t_ls = np.degrees(np.arctan(m_ls))

# Total Least Squares
x_mean, y_mean = np.mean(x), np.mean(y)
pts = np.column_stack([x - x_mean, y - y_mean])
_, _, vt = np.linalg.svd(pts, full_matrices=False)
dx, dy = vt[0][0], vt[0][1]
m_tls = dy / dx if dx != 0 else float('inf')
b_tls = y_mean - m_tls * x_mean
t_tls = np.degrees(np.arctan(m_tls))

# RANSAC
model = RANSACRegressor(random_state=42)
model.fit(x.reshape(-1, 1), y)
m_ran = model.estimator_.coef_[0]
b_ran = model.estimator_.intercept_
t_ran = np.degrees(np.arctan(m_ran))

x_line = np.array([x.min(), x.max()])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].scatter(x, y, s=2, alpha=0.3)
axes[0].plot(x_line, m_ls * x_line + b_ls, 'r-', linewidth=2)
axes[0].set_title(f'Least Squares ({t_ls:.2f} deg)')
axes[0].invert_yaxis()

axes[1].scatter(x, y, s=2, alpha=0.3)
axes[1].plot(x_line, m_tls * x_line + b_tls, 'b-', linewidth=2)
axes[1].set_title(f'Total Least Squares ({t_tls:.2f} deg)')
axes[1].invert_yaxis()

inlier = model.inlier_mask_
axes[2].scatter(x[inlier], y[inlier], s=2, alpha=0.5)
axes[2].scatter(x[~inlier], y[~inlier], s=2, alpha=0.3)
axes[2].plot(x_line, m_ran * x_line + b_ran, 'g-', linewidth=2)
axes[2].set_title(f'RANSAC ({t_ran:.2f} deg)')
axes[2].invert_yaxis()

plt.tight_layout()
plt.show()

print(f'Least Squares angle: {t_ls:.2f} degrees')
print(f'Total Least Squares angle: {t_tls:.2f} degrees')
print(f'RANSAC angle: {t_ran:.2f} degrees')
