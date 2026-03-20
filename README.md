# Assignment 02 - Crop Field Slope Estimation
## Code Structure and Instructions

This assignment uses image processing and line fitting algorithms to estimate crop field slope.

---

## FILES CREATED

### 1. **Q1_Canny_EdgeDetector.py**
- **Question**: Q1 - Apply Canny edge detector to estimate crop field slope
- **What it does**: 
  - Loads the crop field image
  - Applies Canny edge detector (minVal=550, maxVal=800)
  - Displays original image vs. edge detected image
  - Outputs total number of edge pixels detected
- **Output**: `Q1_Canny_Edges.png`

---

### 2. **Q2_Q3_Q4_LeastSquares.py**
- **Questions**: Q2, Q3, Q4
- **What it does**:
  - Extracts edge coordinates (x, y) from Canny output
  - Q2: Plots scatter plot of all edge points
  - Q3-Q4: Fits least-squares line using numpy
  - Calculates slope and crop field angle θ
  - Displays fitted line graphically
- **Outputs**: 
  - `Q2_Scatter_Plot.png`
  - `Q3_Q4_LeastSquares_Fit.png`

---

### 3. **Q6_Q7_TotalLeastSquares.py**
- **Questions**: Q6, Q7
- **What it does**:
  - Uses PCA/SVD (Total-Least-Squares) method
  - Minimizes errors in both x and y directions
  - Fits line and calculates angle θ
  - More robust than ordinary least-squares
- **Output**: `Q6_Q7_TotalLeastSquares_Fit.png`

---

### 4. **Q9_Q10_Q11_RANSAC.py**
- **Questions**: Q9, Q10, Q11
- **What it does**:
  - **Proposed algorithm**: RANSAC (Random Sample Consensus)
  - Automatically identifies inliers (good points) and outliers (noise)
  - Fits line using only inlier points
  - Most robust method for real-world data with noise
  - Shows inliers (green) vs outliers (red)
- **Output**: `Q9_Q10_Q11_RANSAC_Fit.png`

---

### 5. **Comparison_All_Methods.py**
- **Purpose**: Compare all three methods side-by-side
- **What it does**:
  - Runs all three algorithms
  - Creates 3-panel comparison visualization
  - Prints summary table with slopes and angles
- **Output**: `Comparison_All_Methods.png`

---

### 6. **Analysis_Answers_Q5_Q8_Q12.txt**
- **Questions**: Q5, Q8, Q12 (Analysis/explanation questions)
- **Content**:
  - Q5: Why least-squares may not be entirely correct
  - Q8: Why total-least-squares is better
  - Q12: Why RANSAC is the best approach

---

## HOW TO RUN

### Requirements
```bash
pip install opencv-python numpy matplotlib scikit-learn
```

### Run individual scripts:
```bash
python Q1_Canny_EdgeDetector.py
python Q2_Q3_Q4_LeastSquares.py
python Q6_Q7_TotalLeastSquares.py
python Q9_Q10_Q11_RANSAC.py
python Comparison_All_Methods.py
```

### Run all at once (optional):
```bash
python Q1_Canny_EdgeDetector.py && python Q2_Q3_Q4_LeastSquares.py && python Q6_Q7_TotalLeastSquares.py && python Q9_Q10_Q11_RANSAC.py && python Comparison_All_Methods.py
```

---

## ALGORITHM SUMMARY

| Method | Equation | Best For | Sensitivity to Outliers |
|--------|----------|----------|------------------------|
| **Least-Squares** | Minimize Σ(y_i - ŷ)² | General regression | Very High |
| **Total-Least-Squares** | Minimize orthogonal distance | Geometric fitting | High |
| **RANSAC** | Maximize inliers consensus | Real-world noisy data | Very Low |

---

## KEY OUTPUTS

1. **Canny edges** - Shows what features are detected
2. **Scatter plots** - Visualizes all extracted points
3. **Fitted lines** - Shows how each method fits the data
4. **Angle θ** - The estimated crop field slope angle
5. **Comparison** - Side-by-side visualization of all three methods

---

## EXPECTED RESULTS

The RANSAC method should give a more reliable crop field angle estimate because it:
- Filters out noise from edge detection
- Focuses on the dominant crop row direction
- Is resistant to broken or fragmented edges
- Better matches real-world agricultural data patterns
