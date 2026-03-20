import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv.imread(os.path.join(os.path.dirname(__file__), 'Cropped segment.png'), cv.IMREAD_GRAYSCALE)
img = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST)
edges = cv.Canny(img, 550, 690)

indices = np.where(edges != 0)
x, y = indices[1], indices[0]

# Calculate the least-squares-fit
m, b = np.polyfit(x, y, 1)

# Create the line for plotting
x_fit = np.array([np.min(x), np.max(x)])
y_fit = m * x_fit + b

# Graphical Representation
plt.figure(figsize=(6, 6))

# Plot the original edge points as a scatter plot
plt.scatter(x, y, s=1, c='blue', marker='.', label='Edge Points (Data)')

# Plot the least-squares-fit line
plt.plot(x_fit, y_fit, color='red', linewidth=2, label=f'LS Fit: y={m:.2f}x+{b:.2f}')

# Adjust plot settings
plt.gca().invert_yaxis()  # Match image coordinate system
plt.title('Least-Squares-Fit Line on Extracted Edges')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print(f"Estimated Slope (m): {m}")
print(f"Estimated Intercept (b): {b}")
