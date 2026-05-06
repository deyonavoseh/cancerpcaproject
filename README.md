# Anderson Cancer Center — PCA Analysis
### Identifying Essential Variables for Donor Funding Using Principal Component Analysis

---

## Project Overview

This project was developed for the Anderson Cancer Center to address the growing number of referrals by identifying the most essential variables from cancer patient data. Principal Component Analysis (PCA) is used to reduce dimensionality and extract key features, and Logistic Regression is implemented as a bonus to demonstrate predictive capability.

The analysis uses the **Breast Cancer Wisconsin Dataset** from `sklearn.datasets`, which contains data on 569 patients across 30 features.

---

## Dataset

| Property | Detail |
|---|---|
| Source | `sklearn.datasets.load_breast_cancer()` |
| Patients | 569 |
| Features | 30 (radius, texture, perimeter, area, smoothness, etc.) |
| Classes | Malignant (212 cases), Benign (357 cases) |

No external download is needed — the dataset loads automatically from the sklearn library.

---

## Project Structure

```
cancer_pca_project/
│
├── cancer_pca.py          # Main Python script
├── README.md              # This file
├── pca_plot.png           # Output: PCA scatter plot (auto-generated)
├── loadings_plot.png      # Output: Feature contributions chart (auto-generated)
└── confusion_matrix.png   # Output: Logistic regression results (auto-generated)
```

---

## Requirements

Python 3.7 or higher is required. Install all dependencies using the command below.

### Install Dependencies

Open your terminal and run:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Library Versions Used

| Library | Purpose |
|---|---|
| numpy | Numerical computations |
| pandas | Data manipulation and DataFrames |
| matplotlib | Plotting charts |
| seaborn | Enhanced visualizations |
| scikit-learn | PCA, Logistic Regression, dataset |

---

## How to Run

1. Make sure Python is installed on your machine
2. Install the required libraries (see above)
3. Open a terminal in the project folder
4. Run the script:

```bash
python cancer_pca.py
```

5. Three chart windows will appear one after the other — **close each one** to proceed to the next
6. All charts are automatically saved as PNG files in the same folder

---

## Tasks Completed

### Task 1 — PCA Implementation
PCA is applied to the breast cancer dataset to demonstrate how essential variables can be extracted from high-dimensional medical data. The dataset's 30 features are analyzed and reduced using PCA.

### Task 2 — Dimensionality Reduction
The dataset is reduced from **30 features down to 2 PCA components**:

| | Value |
|---|---|
| PC1 variance explained | 44.27% |
| PC2 variance explained | 18.97% |
| Total variance retained | **63.24%** |

This means 63.24% of the information from all 30 features is preserved in just 2 components.

### Task 3 — Bonus: Logistic Regression
Logistic Regression was trained on the 2 PCA components:

| Metric | Score |
|---|---|
| Accuracy | **99%** |
| Macro avg | 0.99 |
| Weighted avg | 0.99 |

The model correctly classified 113 out of 114 test cases.

---

## Output Charts

### Chart 1: PCA Scatter Plot (`pca_plot.png`)
Shows how the 2 PCA components separate malignant (red) and benign (blue) tumors. Clear cluster separation confirms PCA successfully captured meaningful structure in the data.

### Chart 2: Feature Contributions (`loadings_plot.png`)
Shows which of the 30 original features contribute most to PC1. The **top essential variables** identified are:

1. worst radius
2. worst perimeter
3. worst area
4. mean concave points
5. worst concave points

These are the features most relevant for funding presentations and clinical decision-making.

### Chart 3: Confusion Matrix (`confusion_matrix.png`)
Visual summary of the logistic regression predictions showing correct vs incorrect classifications across malignant and benign classes.

---

## Key Findings

- PCA successfully reduced 30 medical features to 2 components while retaining over 63% of the dataset's variance
- The two classes (malignant vs benign) are clearly separable in PCA space, confirming the dataset's predictive potential
- Features related to **tumor size and shape** (radius, perimeter, area, concave points) are the most essential variables
- Logistic Regression trained on just 2 PCA components achieved **99% accuracy**, demonstrating that dimensionality reduction did not significantly compromise predictive power

---

## Author

Submitted for the Anderson Cancer Center Data Analysis Assignment  
Tool: Python 3 | Libraries: scikit-learn, pandas, matplotlib, seaborn