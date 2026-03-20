import cv2, numpy as np, matplotlib.pyplot as plt, os
img = cv2.imread(os.path.join(os.path.dirname(__file__), 'Cropped segment.png'), 0)
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
edges = cv2.Canny(img, 550, 690)
indices = np.where(edges != 0)
x, y = indices[1], indices[0]
plt.figure(figsize=(14, 10), dpi=100)
plt.scatter(x, y, s=5); plt.gca().invert_yaxis(); plt.title('Q2: Edge Points', fontsize=14)
plt.savefig('Q2_Scatter.png', dpi=150, bbox_inches='tight'); plt.show()
A = np.vstack([x, np.ones(len(x))]).T
m, b = np.linalg.lstsq(A, y, rcond=None)[0]
theta = np.degrees(np.arctan(m))
plt.figure(figsize=(14, 10), dpi=100)
plt.scatter(x, y, s=5, alpha=0.3)
x_line = np.array([x.min(), x.max()])
plt.plot(x_line, m*x_line+b, 'r-', linewidth=3); plt.gca().invert_yaxis()
plt.title(f'Q3-Q4: Least Squares θ={theta:.2f}°', fontsize=14)
plt.savefig('Q3_Q4_LeastSquares.png', dpi=150, bbox_inches='tight'); plt.show()
print(f'Slope:{m:.6f} θ:{theta:.2f}°')
