import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
import os

img_path = os.path.join(os.path.dirname(__file__), 'Cropped segment.png')
img = cv2.imread(img_path, 0)

if img is None:
    print(f"Error: Could not load image from {img_path}")
else:
    # Apply Canny Edge Detector
    edges = cv2.Canny(img, 550, 690)
    # Extract x and y coordinates
    indices = np.where(edges != 0)
    x = indices[1]
    y = indices[0]
    
    # Q9-Q11: RANSAC
    ransac = RANSACRegressor(random_state=42, min_samples=10, max_trials=1000)
    ransac.fit(x.reshape(-1,1), y)
    m = ransac.estimator_.coef_[0]
    b = ransac.estimator_.intercept_
    theta = np.degrees(np.arctan(m))
    
    plt.figure(figsize=(10, 8))
    inlier = ransac.inlier_mask_
    plt.scatter(x[inlier], y[inlier], s=2, alpha=0.5, label='Inliers')
    plt.scatter(x[~inlier], y[~inlier], s=2, alpha=0.3, label='Outliers')
    x_line = np.array([x.min(), x.max()])
    plt.plot(x_line, m*x_line + b, 'g-', linewidth=2)
    plt.gca().invert_yaxis()
    plt.title(f'Q9-Q11: RANSAC (θ={theta:.2f}°)')
    plt.legend()
    plt.savefig('Q9_Q10_Q11_RANSAC.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Slope: {m:.6f}, θ: {theta:.2f}°")
