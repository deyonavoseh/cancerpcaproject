# =============================================
# ANDERSON CANCER CENTER - PCA ANALYSIS
# =============================================

# STEP 1: Import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =============================================
# STEP 2: Load and Explore the Dataset
# =============================================
cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

print("=== DATASET INFO ===")
print("Shape:", df.shape)
print("Target classes:", cancer.target_names)
print("Class counts:\n", df['target'].value_counts())

# =============================================
# STEP 3: Standardize the Data
# =============================================
X = cancer.data
y = cancer.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n=== STANDARDIZATION DONE ===")
print("Mean (should be ~0):", round(X_scaled.mean(), 4))
print("Std  (should be ~1):", round(X_scaled.std(), 4))

# =============================================
# STEP 4: Apply PCA - Reduce to 2 Components
# =============================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n=== PCA RESULTS ===")
print("Original shape:", X_scaled.shape)
print("Reduced shape :", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained: {:.2f}%".format(
    sum(pca.explained_variance_ratio_) * 100))

# =============================================
# STEP 5: Visualize the 2 PCA Components
# =============================================
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y
pca_df['Diagnosis'] = pca_df['Target'].map({0: 'Malignant', 1: 'Benign'})

plt.figure(figsize=(9, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2',
                hue='Diagnosis',
                palette={'Malignant': 'red', 'Benign': 'steelblue'},
                alpha=0.7, s=80)
plt.title('PCA - Breast Cancer Dataset (2 Components)', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.legend(title='Diagnosis')
plt.tight_layout()
plt.savefig('pca_plot.png', dpi=150)
plt.show()

# =============================================
# STEP 6: Identify Essential Variables (Loadings)
# =============================================
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=cancer.feature_names
)

print("\n=== TOP ESSENTIAL VARIABLES (PC1) ===")
print(loadings['PC1'].abs().sort_values(ascending=False).head(10))

loadings['PC1'].abs().sort_values(ascending=False).plot(
    kind='bar', figsize=(12, 5), color='steelblue',
    title='Feature Contributions to PC1 (Essential Variables)'
)
plt.ylabel('Absolute Loading')
plt.tight_layout()
plt.savefig('loadings_plot.png', dpi=150)
plt.show()

# =============================================
# STEP 7: BONUS - Logistic Regression
# =============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\n=== LOGISTIC REGRESSION RESULTS ===")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("\n=== DONE! Check your folder for the saved plot images ===")