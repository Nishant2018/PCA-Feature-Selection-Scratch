## Principal Component Analysis (PCA)

### Introduction

Principal Component Analysis (PCA) is a powerful dimensionality reduction technique commonly used in machine learning and data analysis. It transforms a dataset into a set of linearly uncorrelated variables called principal components. The primary goal of PCA is to reduce the dimensionality of the data while retaining as much variability as possible.

### Why Use PCA?

- **Dimensionality Reduction**: Simplifies the dataset by reducing the number of features.
- **Noise Reduction**: Helps in removing noise and redundant features.
- **Visualization**: Makes it easier to visualize high-dimensional data in 2D or 3D space.
- **Improved Performance**: Enhances the performance of machine learning algorithms by reducing overfitting.

### How PCA Works

1. **Standardize the Data**: PCA is affected by the scale of the variables, so it's essential to standardize the dataset.
   \[
   z = \frac{x - \mu}{\sigma}
   \]
   Where \( z \) is the standardized value, \( x \) is the original value, \( \mu \) is the mean, and \( \sigma \) is the standard deviation.

2. **Compute the Covariance Matrix**: Measure the variance and the relationship between different variables.
   \[
   \mathbf{C} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(x_i - \bar{x})^T
   \]
   Where \( \mathbf{C} \) is the covariance matrix, \( n \) is the number of samples, \( x_i \) is the \( i \)-th sample, and \( \bar{x} \) is the mean vector.

3. **Calculate the Eigenvalues and Eigenvectors**: Eigenvectors determine the direction of the new feature space, and eigenvalues determine their magnitude (importance).
   \[
   \mathbf{C} \mathbf{v} = \lambda \mathbf{v}
   \]
   Where \( \mathbf{v} \) is the eigenvector and \( \lambda \) is the eigenvalue.

4. **Sort Eigenvalues and Eigenvectors**: Rank the eigenvalues and their corresponding eigenvectors in descending order.

5. **Select Principal Components**: Choose the top \( k \) eigenvectors based on the largest eigenvalues to form a new matrix \( \mathbf{W} \).

6. **Transform the Data**: Project the original dataset onto the new feature space.
   \[
   \mathbf{Y} = \mathbf{W}^T \mathbf{X}
   \]
   Where \( \mathbf{Y} \) is the transformed dataset, \( \mathbf{W} \) is the matrix of selected eigenvectors, and \( \mathbf{X} \) is the original dataset.

### Example Code

Here is a simple example of how to perform PCA using Python's `scikit-learn` library:

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample data
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

print("Principal Components:\n", principal_components)
