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
    
    # Q2: Scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, s=1)
    plt.gca().invert_yaxis()
    plt.title('Q2: Edge Points')
    plt.savefig('Q2_Scatter.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Q3-Q4: Least squares
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    theta = np.degrees(np.arctan(m))
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, s=1, alpha=0.3)
    x_line = np.array([x.min(), x.max()])
    plt.plot(x_line, m*x_line + b, 'r-', linewidth=2)
    plt.gca().invert_yaxis()
    plt.title(f'Q3-Q4: Least Squares (θ={theta:.2f}°)')
    plt.savefig('Q3_Q4_LeastSquares.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Slope: {m:.6f}, θ: {theta:.2f}°")
