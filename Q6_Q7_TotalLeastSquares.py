import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    
    # Q6-Q7: Total Least Squares (SVD)
    x_m, y_m = np.mean(x), np.mean(y)
    points = np.column_stack([x-x_m, y-y_m])
    U, S, Vt = np.linalg.svd(points, full_matrices=False)
    m = Vt[0][1]/Vt[0][0] if Vt[0][0]!=0 else float('inf')
    theta = np.degrees(np.arctan(m))
    b = y_m - m*x_m
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, s=1, alpha=0.3)
    x_line = np.array([x.min(), x.max()])
    plt.plot(x_line, m*x_line + b, 'b-', linewidth=2)
    plt.gca().invert_yaxis()
    plt.title(f'Q6-Q7: Total Least Squares (θ={theta:.2f}°)')
    plt.savefig('Q6_Q7_TotalLeastSquares.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Slope: {m:.6f}, θ: {theta:.2f}°")
