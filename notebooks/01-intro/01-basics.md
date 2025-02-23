## Machine Learning Basics

### Supervised Learning
Supervised learning is a type of machine learning where the model is trained on a labeled dataset, meaning that each training example is paired with an output label. The goal of supervised learning is to learn a mapping from inputs to outputs that can be used to predict the output for new, unseen data.

- **Training Data Requirements**: Requires labeled data, where each instance includes both independent variables (features) and a dependent variable (label).
- **Labeling**: The presence of labeled data allows the algorithm to learn by comparing its predictions to the true labels, thereby adjusting its parameters to minimize prediction error.
- **Common Use Cases**:
  - **Regression Models**: Used when the dependent variable is continuous. Examples include Linear Regression, Ridge Regression, and Lasso Regression.
  - **Classification Models**: Used when the dependent variable is categorical. Examples include Logistic Regression, Support Vector Machines, and Decision Trees.
- **Goal**: To predict or classify new data based on the patterns learned from the labeled training data.
- **Complexity**: The complexity of supervised learning models can vary widely, from simple linear models to complex neural networks. The need for labeled data and the process of model evaluation add to the overall complexity.

### Unsupervised Learning
Unsupervised learning is a type of machine learning where the model is trained on unlabeled data. The goal is to discover hidden patterns, structures, or relationships within the data without the guidance of labeled examples.

- **Training Data Requirements**: Requires unlabeled data, consisting only of independent variables (features) without any dependent variable (label).
- **Labeling**: Since there are no labels, the algorithm must find structure in the data autonomously, without any explicit supervision.
- **Common Use Cases**:
  - **Clustering Models**: Used to group similar data points together. Examples include K-Means, DBSCAN, and Hierarchical Clustering.
  - **Dimensionality Reduction Models**: Used to reduce the number of features while preserving the essential structure of the data. Examples include Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE).
  - **Outlier Detection Models**: Used to identify unusual data points that do not conform to the expected pattern. Examples include Isolation Forest and One-Class SVM.
- **Goal**: To discover hidden patterns or structures in the data, such as clusters, dimensions, or anomalies.
- **Complexity**: Unsupervised learning models are generally simpler in terms of data preparation, as they do not require labeled data. However, interpreting the results can be more challenging, as there is no ground truth to compare against.

### Regression
Regression analysis is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It is particularly useful when the dependent variable is continuous.

- **Use Case**: Applicable when the response variable to be predicted is a continuous variable (scalar).
- **Examples**:
  - **Linear Regression**: Models the relationship between the dependent variable and one or more independent variables using a linear predictor function.
  - **Ridge Regression**: A regularization technique that adds a penalty to the size of the coefficients to prevent overfitting.
  - **Lasso Regression**: A regularization technique that can shrink some coefficients to zero, effectively performing feature selection.
  - **XGBoost Regression**: An advanced gradient boosting algorithm that builds an ensemble of weak prediction models, typically decision trees.

### Classification
Classification is a supervised learning technique used to predict the category or class label of new observations based on a training dataset. It is particularly useful when the dependent variable is categorical.

- **Use Case**: Applicable when the response variable to be predicted is a categorical variable (scalar).
- **Examples**:
  - **Logistic Regression**: Models the probability of the default class (e.g., 0 or 1) using a logistic function.
  - **Support Vector Machines (SVM)**: Finds the optimal hyperplane that maximizes the margin between two classes.
  - **Decision Trees**: A non-parametric method that splits the data into subsets based on feature values to make predictions.
  - **XGBoost Classification**: An advanced gradient boosting algorithm that builds an ensemble of weak prediction models, typically decision trees.


## Regression Performance Metrics
Regression performance metrics are essential tools used to evaluate the accuracy and reliability of regression models. These metrics quantify the differences between the observed values and the values predicted by the model, providing insights into the model's predictive power and fit to the data.

### Residual Sum of Squares (RSS)
The Residual Sum of Squares (RSS) is the sum of the squared differences between the observed values and the predicted values. It measures the total deviation of the response values from the fit to the response values. It is a measure of the discrepancy between the data and the estimated model.

$$
\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where:
- $ y_i $ is the observed value of the dependent variable for the $ i $-th data point.
- $ \hat{y}_i $ is the predicted value of the dependent variable for the $ i $-th data point.
- $ n $ is the total number of data points.

### Mean Squared Error (MSE)
The Mean Squared Error (MSE) is the average of the squared differences between the observed values and the predicted values. It is a measure of the quality of an estimator or a predictor. It is always non-negative, and values closer to zero are better.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where:
- $ y_i $ is the observed value of the dependent variable for the \( i \)-th data point.
- $ \hat{y}_i $ is the predicted value of the dependent variable for the \( i \)-th data point.
- $ n $ is the total number of data points.

### Root Mean Squared Error (RMSE)
The Root Mean Squared Error (RMSE) is the square root of the mean squared error. It is a measure of the standard deviation of the residuals (prediction errors). It is a measure of how well the model fits the data, and it is expressed in the same units as the dependent variable.

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

where:
- $ y_i $ is the observed value of the dependent variable for the \( i \)-th data point.
- $ \hat{y}_i $ is the predicted value of the dependent variable for the \( i \)-th data point.
- $ n $ is the total number of data points.

### Mean Absolute Error (MAE)
The Mean Absolute Error (MAE) is the average of the absolute differences between the observed values and the predicted values. It is a measure of the average magnitude of the errors in a set of predictions, without considering their direction. It is always non-negative, and values closer to zero are better.

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

where:
- $ y_i $ is the observed value of the dependent variable for the $ i $-th data point.
- $ \hat{y}_i $ is the predicted value of the dependent variable for the $ i $-th data point.
- $ n $ is the total number of data points.