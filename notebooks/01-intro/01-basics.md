# Machine Learning Basics

### Supervised Learning
Supervised learning is a type of machine learning where the model is trained on a labeled dataset, meaning that each training example is paired with an output label. The goal of supervised learning is to learn a mapping from inputs to outputs that can be used to predict the output for new, unseen data.

- **Training Data Requirements**: Requires labeled data, where each instance includes both independent variables (features) and a dependent variable (label).
- **Labeling**: The presence of labeled data allows the algorithm to learn by comparing its predictions to the true labels, thereby adjusting its parameters to minimize prediction error.
- **Common Use Cases**:
  - **Regression Models**: Used when the dependent variable is continuous. Examples include Linear Regression, Ridge Regression, and Lasso Regression.
  - **Classification Models**: Used when the dependent variable is categorical. Examples include Logistic Regression, Support Vector Machines, and Decision Trees.
- **Goal**: To predict or classify new data based on the patterns learned from the labeled training data.
- **Complexity**: The complexity of supervised learning models can vary widely, from simple linear models to complex neural networks. The need for labeled data and the process of model evaluation add to the overall complexity.

---

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

--- 

### Regression
Regression analysis is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It is particularly useful when the dependent variable is continuous.

- **Use Case**: Applicable when the response variable to be predicted is a continuous variable (scalar).
- **Examples**:
  - **Linear Regression**: Models the relationship between the dependent variable and one or more independent variables using a linear predictor function.
  - **Ridge Regression**: A regularization technique that adds a penalty to the size of the coefficients to prevent overfitting.
  - **Lasso Regression**: A regularization technique that can shrink some coefficients to zero, effectively performing feature selection.
  - **XGBoost Regression**: An advanced gradient boosting algorithm that builds an ensemble of weak prediction models, typically decision trees.

---

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

---

### Mean Squared Error (MSE)
The Mean Squared Error (MSE) is the average of the squared differences between the observed values and the predicted values. It is a measure of the quality of an estimator or a predictor. It is always non-negative, and values closer to zero are better.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where:
- $ y_i $ is the observed value of the dependent variable for the \( i \)-th data point.
- $ \hat{y}_i $ is the predicted value of the dependent variable for the \( i \)-th data point.
- $ n $ is the total number of data points.

---

### Root Mean Squared Error (RMSE)
The Root Mean Squared Error (RMSE) is the square root of the mean squared error. It is a measure of the standard deviation of the residuals (prediction errors). It is a measure of how well the model fits the data, and it is expressed in the same units as the dependent variable.

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

where:
- $ y_i $ is the observed value of the dependent variable for the \( i \)-th data point.
- $ \hat{y}_i $ is the predicted value of the dependent variable for the \( i \)-th data point.
- $ n $ is the total number of data points.

---

### Mean Absolute Error (MAE)
The Mean Absolute Error (MAE) is the average of the absolute differences between the observed values and the predicted values. It is a measure of the average magnitude of the errors in a set of predictions, without considering their direction. It is always non-negative, and values closer to zero are better.

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

where:
- $ y_i $ is the observed value of the dependent variable for the $ i $-th data point.
- $ \hat{y}_i $ is the predicted value of the dependent variable for the $ i $-th data point.
- $ n $ is the total number of data points.

## Classification Performance Metrics

In the field of machine learning and computer science, evaluating the performance of classification models is crucial for understanding their effectiveness and making informed decisions. This section provides a detailed explanation of four key metrics: Accuracy, Precision, Recall, and F-1 Score.

### 1. Accuracy

**Definition:**  
Accuracy is a fundamental metric that measures the proportion of correct predictions made by a classification model. It is the ratio of the number of correct predictions to the total number of predictions.

**Formula:**

$$
\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{True Positives} + \text{True Negatives} + \text{False Positives} + \text{False Negatives}}
$$

**Explanation:**  
- **True Positives (TP):** Correctly predicted positive instances.  
- **True Negatives (TN):** Correctly predicted negative instances.  
- **False Positives (FP):** Incorrectly predicted positive instances (Type I Error).  
- **False Negatives (FN):** Incorrectly predicted negative instances (Type II Error).  

**Use Case:**  
Accuracy is a straightforward metric and is most useful when the dataset is balanced (i.e., an equal number of positive and negative instances). However, it can be misleading in imbalanced datasets, where one class significantly outnumbers the other.

---

### 2. Precision

**Definition:**  
Precision, also known as the Positive Predictive Value, measures the proportion of true positive predictions out of all positive predictions made by the model. It is particularly useful when the cost of false positives is high.

**Formula:**

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

**Explanation:**  
Precision focuses on the accuracy of the positive predictions. A high precision indicates that the model is confident in its positive predictions and has a low rate of false positives.

**Use Case:**  
Precision is critical in applications where false positives are costly, such as medical diagnoses or spam detection. For example, in spam detection, a high precision ensures that few non-spam emails are incorrectly marked as spam.

---

### 3. Recall

**Definition:**  
Recall, also known as Sensitivity or True Positive Rate, measures the proportion of true positive predictions out of all actual positive cases. It is crucial when the cost of false negatives is high.

**Formula:**

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

**Explanation:**  
Recall focuses on the model's ability to identify all positive instances. A high recall indicates that the model captures most of the actual positive cases, minimizing false negatives.

**Use Case:**  
Recall is essential in applications where missing positive instances is critical, such as disease detection or fraud detection. For example, in disease detection, a high recall ensures that most patients with the disease are correctly identified.

---

### 4. F-1 Score

**Definition:**  
The F-1 Score is the harmonic mean of precision and recall, providing a single metric that balances both. It is particularly useful when you need to consider both precision and recall equally.

**Formula:**

$$
\text{F-1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Explanation:**  
The F-1 Score combines the strengths of precision and recall into a single metric. It is especially valuable when dealing with imbalanced datasets or when both precision and recall are important.

**Use Case:**  
The F-1 Score is widely used in scenarios where false positives and false negatives have similar costs. For example, in information retrieval, the F-1 Score helps balance the trade-off between retrieving relevant documents (precision) and missing relevant documents (recall).

## Clustering Performance Metrics

Clustering is an unsupervised learning technique used to group similar data points into clusters without predefined labels. Evaluating the performance of clustering algorithms is essential to assess the quality of the resulting clusters. This section provides detailed explanations of three key clustering performance metrics: Homogeneity, Silhouette Score, and Completeness.

### 1. Homogeneity

**Definition:**  
Homogeneity measures the extent to which each cluster contains only members of a single class. It evaluates whether the clustering algorithm has grouped data points from the same class together.

**Formula:**

$$
\text{Homogeneity} (h) = 1 - \frac{\text{Conditional entropy given cluster assignments}}{\text{Entropy of (predicted) class}}
$$

**Explanation:**  
- **Conditional Entropy \( H(C|K) \):** Measures the uncertainty in the class labels given the cluster assignments.
- **Entropy of Predicted Class \( H(K) \):** Measures the uncertainty in the cluster assignments themselves.

The homogeneity score ranges from 0 to 1:
- **1:** Each cluster contains only members of a single class (perfect homogeneity).
- **0:** Clusters are completely random with respect to the class labels.

**Use Case:**  
Homogeneity is particularly useful when evaluating clustering results against known class labels. It is commonly used in scenarios where the ground truth is available, such as in semi-supervised learning.

---

### 2. Silhouette Score

**Definition:**  
The Silhouette Score measures how similar a data point is to its own cluster compared to other clusters. It combines both the cohesion within clusters and the separation between clusters.

**Formula:**

For a single data point \( o \):

$$
\text{Silhouette Score} (s(o)) = \frac{b(o) - a(o)}{\max\{a(o), b(o)\}}
$$

Where:
- \( a(o) \): Average distance of point \( o \) to all other points in the same cluster (measures intra-cluster similarity).
- \( b(o) \): Average distance of point \( o \) to points in the nearest neighboring cluster (measures inter-cluster separation).

The overall Silhouette Score for a dataset is the average of the silhouette scores for all data points.

**Explanation:**  
- **Silhouette Score:** Ranges from -1 to 1.
  - **+1:** Indicates well-separated and cohesive clusters.
  - **0:** Indicates overlapping clusters.
  - **-1:** Indicates poor clustering where points are closer to other clusters.

**Use Case:**  
The Silhouette Score is widely used to evaluate the quality of clustering algorithms like K-Means, DBSCAN, and Hierarchical Clustering. It helps in determining the optimal number of clusters by comparing scores for different configurations.

---

### 3. Completeness

**Definition:**  
Completeness measures the extent to which all members of a given class are assigned to the same cluster. It evaluates whether the clustering algorithm has grouped all instances of the same class together.

**Formula:**

$$
\text{Completeness} (c) = 1 - \frac{\text{Conditional entropy given cluster assignments}}{\text{Entropy of (actual) class}}
$$

**Explanation:**  
- **Conditional Entropy \( H(K|C) \):** Measures the uncertainty in the cluster assignments given the class labels.
- **Entropy of Actual Class \( H(C) \):** Measures the uncertainty in the class labels themselves.

The completeness score ranges from 0 to 1:
- **1:** All members of a given class are assigned to the same cluster (perfect completeness).
- **0:** Members of the same class are scattered across multiple clusters.

**Use Case:**  
Completeness is useful for evaluating clustering performance in scenarios where the ground truth labels are available. It complements homogeneity by ensuring that all instances of the same class are grouped together.

### Machine Learning Model Evaluation
- **Step 1 >>** __Data Preparation__: Split the data into train, validation and test.
- **Step 2 >>** __Model Training__: Train the model on the training data and save the fitted model.
- **Step 3 >>** __Hyper-Parameter Tunning__: Use the fitted Model and Validation Set to find the optimal set of parameters where the model performs the best.
- **Step 4 >>** __Prediction__: Use the optimal set of parameters from Hyper-Parameter Tuning Stage and traning data, to train the model again with these hyper parameters, use this best fitted model to predictions on test data.
- **Step 5 >>** __Test Error Rate__: Compute the performance metrics for your model using the predictions and real values of the target varaiable from your test data.
