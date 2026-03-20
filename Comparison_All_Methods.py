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
    
    # Method 1: Least Squares
    A = np.vstack([x, np.ones(len(x))]).T
    m1, b1 = np.linalg.lstsq(A, y, rcond=None)[0]
    t1 = np.degrees(np.arctan(m1))
    
    # Method 2: Total Least Squares
    x_m, y_m = np.mean(x), np.mean(y)
    points = np.column_stack([x-x_m, y-y_m])
    U, S, Vt = np.linalg.svd(points, full_matrices=False)
    m2 = Vt[0][1]/Vt[0][0]
    t2 = np.degrees(np.arctan(m2))
    
    # Method 3: RANSAC
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(x.reshape(-1,1), y)
    m3 = ransac.estimator_.coef_[0]
    t3 = np.degrees(np.arctan(m3))
    
    # Plot comparison
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    x_line = np.array([x.min(), x.max()])
    
    axes[0].scatter(x, y, s=1, alpha=0.3)
    axes[0].plot(x_line, m1*x_line + b1, 'r-', linewidth=2)
    axes[0].set_title(f'Least Squares\nθ={t1:.2f}°')
    axes[0].invert_yaxis()
    
    axes[1].scatter(x, y, s=1, alpha=0.3)
    axes[1].plot(x_line, m2*x_line + (y_m-m2*x_m), 'b-', linewidth=2)
    axes[1].set_title(f'Total Least Squares\nθ={t2:.2f}°')
    axes[1].invert_yaxis()
    
    inlier = ransac.inlier_mask_
    axes[2].scatter(x[inlier], y[inlier], s=2, alpha=0.5, label='Inliers')
    axes[2].scatter(x[~inlier], y[~inlier], s=2, alpha=0.3, label='Outliers')
    axes[2].plot(x_line, m3*x_line + ransac.estimator_.intercept_, 'g-', linewidth=2)
    axes[2].set_title(f'RANSAC\nθ={t3:.2f}°')
    axes[2].invert_yaxis()
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('Comparison_All_Methods.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"LS: {t1:.2f}° | TLS: {t2:.2f}° | RANSAC: {t3:.2f}°")
