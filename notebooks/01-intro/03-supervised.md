# Supervised Learning

Supervised learning is a type of machine learning where the model is trained on labeled data, meaning that each input sample is associated with a corresponding output label. The goal is to learn a mapping from inputs to outputs, which can then be used to make predictions on new, unseen data.

## Formal Definition

In supervised learning, we are given a training dataset $\mathcal{D}$ consisting of $n$ input-output pairs:

$$
\mathcal{D} = \{ (\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n) \}
$$

where:
- $\mathbf{x}_i \in \mathbb{R}^d$ is the $i$-th input feature vector.
- $y_i \in \mathcal{Y}$ is the corresponding output label for the $i$-th input.

The objective is to learn a function $f: \mathbb{R}^d \to \mathcal{Y}$ such that $f(\mathbf{x}_i) \approx y_i$ for all $i$.

## Feature Vector 
The feature vector $\mathbf{x}_i$ represents the input data for the $i$-th sample. Each feature vector $\mathbf{x}_i$ is a $d$-dimensional vector, where $d$ is the number of features (or attributes) in the dataset. For example, if we are working with a dataset of images of cars, planes, and motorcycles, each image can be represented as a feature vector where each element of the vector corresponds to a pixel value or a higher-level feature extracted from the image.

---

### Example of a Feature Vector

Consider a simple example where we have a dataset of images of cars, planes, and motorcycles. Each image is represented by a feature vector $\mathbf{x}_i$ that includes the following features:
- Pixel values (if using raw pixel data)
- Edge detection features
- Texture features
- Color histograms

For instance, if we are using a 3x3 grayscale image, the feature vector might look like this:

$$
\mathbf{x}_i = [x_{i1}, x_{i2}, \ldots, x_{i9}]
$$

where each $x_{ij}$ represents the intensity of a pixel in the image.

## Diagrammatic Representation

The process of supervised learning can be visualized as follows:

```
    +------------------+
    |                  | ---> Prediction (Output)
    |     Model        |
    |                  |
    +------------------+
      ^      ^      ^
      |      |      |
 Input_1  Input_2  ...  Input_n
```

## Features 

### Qualitative (Categorical) Data 
Categorical data representing discrete groups or categories. Also known as categorical variables.

Can be further divided into:

- **Nominal Data**: Categories without any inherent order (e.g., colors, countries).
- **Ordinal Data**: Categories with a meaningful order but no fixed intervals (e.g., rankings, education levels).

#### One-Hot Encoding
Machine learning algorithms typically require numerical input. Therefore, nominal data must be transformed into numerical format. One prevalent method for this transformation is **One-Hot Encoding**.

**One-Hot Encoding** involves representing each category as a binary vector. For a feature with $\mathcal{k}$  unique categories, one-hot encoding creates $\mathcal{k}$ binary features. Each binary feature corresponds to one category, taking the value 1 if the original feature matches that category and 0 otherwise.


Consider a categorical feature representing countries: [USA, India, Canada, France].

Applying one-hot encoding results in:

| Country | USA | India | Canada | France |
|---------|-----|-------|--------|--------|
| USA     |  1  |   0   |    0   |    0   |
| India   |  0  |   1   |    0   |    0   |
| Canada  |  0  |   0   |    1   |    0   |
| France  |  0  |   0   |    0   |    1   |

In this representation:
- Each country is transformed into a binary vector.
- The presence of a '1' indicates the corresponding category, while '0's denote the absence of other categories.

This encoding ensures that the machine learning model interprets each category as a distinct entity without implying any ordinal relationship between them.

### Qualitative (Numerical) Data 
Quantitative data represents measurable quantities and is expressed numerically. It can be further categorized into:

- **Discrete Data**: Numerical values that represent countable quantities and can take only specific values, typically integers. Examples include the number of students in a class or the number of cars in a parking lot.
- **Continuous Data**: Numerical values that represent measurable quantities and can take any value within a range. Examples include height, weight, temperature, and time.

#### Encoding Quantitative Data

While quantitative data is inherently numerical, certain scenarios require encoding or transformation to enhance model performance:

1. **Normalization and Standardization**:
   - **Normalization**: Rescales the data to a [0, 1] range. Useful when the algorithm assumes a bounded input.
   - **Standardization**: Centers the data to have a mean of 0 and a standard deviation of 1. Beneficial when the algorithm assumes data is normally distributed.

   These techniques ensure that features with different scales do not disproportionately influence the model.

2. **Binning (Discretization)**:
   - Converts continuous data into discrete bins or intervals. This can help capture non-linear relationships and reduce the impact of outliers.

   **Example**:

   Consider a continuous feature representing age. Binning could categorize ages into groups:

   | Age | Age Group |
   |-----|-----------|
   | 23  | 20-30     |
   | 37  | 30-40     |
   | 45  | 40-50     |
   | 52  | 50-60     |

   This transformation can make patterns more apparent to certain algorithms.

3. **Polynomial Features**:
   - Generates new features by raising existing features to a power. This can help in modeling non-linear relationships.

   **Example**:

   For a feature $ x $, polynomial features would include $ x^2 $, $ x^3 $, etc.

   This expansion allows linear models to fit non-linear data.

4. **Log Transformation**:
   - Applies the logarithm function to compress the range of the data, which can help in handling skewed distributions.

   **Example**:

   Transforming a feature $ x $ to $ \log(x) $ can reduce the impact of large values.

   This is particularly useful when dealing with data that spans several orders of magnitude.

## Types of Predictions

In supervised learning, prediction tasks are primarily categorized based on the nature of the output variable. The two principal types are:

1. **Regression**: Predicting continuous numerical values.
2. **Classification**: Assigning inputs to discrete categories or classes.

### Regression

Regression analysis is a fundamental statistical method used to model and analyze the relationships between a dependent variable and one or more independent variables. The primary objective of regression is to predict continuous outcomes, making it indispensable in various scientific and engineering disciplines.

Regression techniques can be broadly categorized based on the nature of the relationship between variables and the form of the regression equation. The principal types include:

1. **Linear Regression**
2. **Multiple Linear Regression**
3. **Polynomial Regression**
4. **Ridge Regression**
5. **Lasso Regression**

### 1. Linear Regression

**Definition**: Linear regression models the relationship between a dependent variable $ y $ and a single independent variable $ x $ by fitting a linear equation to observed data.

**Mathematical Formulation**:

The linear regression model is expressed as:

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i 
$$

where:
- $ y_i $ is the dependent variable.
- $ x_i $ is the independent variable.
- $ \beta_0 $ is the y-intercept.
- $ \beta_1 $ is the slope coefficient.
- $ \varepsilon_i $ represents the error term for observation $ i $.

The parameters $ \beta_0 $ and $ \beta_1 $ are estimated using the **Ordinary Least Squares (OLS)** method, which minimizes the sum of the squared residuals:

$$ 
\min_{\beta_0, \beta_1} \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2 
$$

### 2. Multiple Linear Regression

**Definition**: Extends linear regression by modeling the relationship between a dependent variable and multiple independent variables.

**Mathematical Formulation**:

The model is represented as:

$$ 
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i 
$$

or in matrix notation:

$$
 \mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon} 
$$

where:
- $ \mathbf{y} $ is an $ n \times 1 $ vector of observations.
- $ \mathbf{X} $ is an $ n \times (p + 1) $ design matrix including a column of ones for the intercept.
- $ \boldsymbol{\beta} $ is a $ (p + 1) \times 1 $ vector of coefficients.
- $ \boldsymbol{\varepsilon} $ is an $ n \times 1 $ vector of error terms.

The OLS estimator for $ \boldsymbol{\beta} $ is:

$$
 \hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} 
$$

### 3. Polynomial Regression

**Definition**: A form of regression analysis where the relationship between the independent variable $ x $ and the dependent variable $ y $ is modeled as an $ n $th degree polynomial.

**Mathematical Formulation**:

$$
 y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \cdots + \beta_n x_i^n + \varepsilon_i 
$$

This can be viewed as a linear regression problem with respect to the new features $ x_i, x_i^2, \ldots, x_i^n $.

### 4. Ridge Regression

**Definition**: A regularization technique that addresses multicollinearity by adding a penalty equivalent to the square of the magnitude of coefficients to the loss function.

**Mathematical Formulation**:

The ridge regression estimator is obtained by minimizing:

$$
 \min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^n (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\} 
$$

where $ \lambda \geq 0 $ is the regularization parameter controlling the strength of the penalty.

In matrix form:

$$
 \hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y} 
$$

### 5. Lasso Regression

**Definition**: Another regularization technique that adds a penalty equal to the absolute value of the magnitude of coefficients, promoting sparsity in the model.

**Mathematical Formulation**:

The lasso estimator is obtained by minimizing:

$$
 \min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^n (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2 + \lambda \sum_{j=1}^p |\beta_j| \right\} 
$$

The $ \ell_1 $ penalty can shrink some coefficients to exactly zero, effectively performing variable selection.


### Classification

Classification involves predicting discrete class labels for given input data. The objective is to assign each input instance to one of the predefined categories based on its features.

#### Binary Classification

In binary classification, the model distinguishes between two distinct classes. This scenario is prevalent in various applications, such as:

- **Spam Detection**: Classifying emails as 'Spam' or 'Not Spam'.
- **Medical Diagnosis**: Identifying the presence ('Positive') or absence ('Negative') of a disease.

**Mathematical Formulation**:

Given a dataset $ \mathcal{D} = \{ (\mathbf{x}_i, y_i) \}_{i=1}^n $, where:

- $ \mathbf{x}_i \in \mathbb{R}^d $ represents the feature vector.
- $ y_i \in \{0, 1\} $ denotes the binary class label.

The goal is to learn a function $ f: \mathbb{R}^d \to \{0, 1\} $ that maps input features to a binary outcome.

**Logistic Regression** is a common algorithm used for binary classification. It models the probability that a given input $ \mathbf{x} $ belongs to class 1 as:

$$
P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)
$$

where:

- $ \mathbf{w} $ is the weight vector.
- $ b $ is the bias term.
- $ \sigma(z) = \frac{1}{1 + e^{-z}} $ is the sigmoid function.

The predicted class $ \hat{y} $ is determined by thresholding the probability:

$$
\hat{y} = 
\begin{cases} 
1 & \text{if } P(y = 1 \mid \mathbf{x}) \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

#### Multiclass Classification

Multiclass classification extends binary classification to scenarios where there are more than two classes. Examples include:

- **Handwritten Digit Recognition**: Classifying images as digits '0' through '9'.
- **Object Recognition**: Identifying objects in images as 'Cat', 'Dog', 'Bird', etc.

**Mathematical Formulation**:

For a dataset $ \mathcal{D} = \{ (\mathbf{x}_i, y_i) \}_{i=1}^n $, where:

- $ \mathbf{x}_i \in \mathbb{R}^d $ represents the feature vector.
- $ y_i \in \{1, 2, \ldots, K\} $ denotes the class label among $ K $ possible classes.

The objective is to learn a function $ f: \mathbb{R}^d \to \{1, 2, \ldots, K\} $.

**Softmax Regression** (also known as Multinomial Logistic Regression) is commonly employed for multiclass classification. It models the probability of class $ k $ as:

$$
P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^\top \mathbf{x} + b_j}}
$$

where:

- $ \mathbf{w}_k $ and $ b_k $ are the weight vector and bias for class $ k $, respectively.

The predicted class $ \hat{y} $ is:

$$
\hat{y} = \arg\max_{k \in \{1, 2, \ldots, K\}} P(y = k \mid \mathbf{x})
$$

**Strategies for Multiclass Classification**:

1. **One-vs-Rest (OvR)**:
   - **Approach**: Train $ K $ binary classifiers, each distinguishing one class from the rest.
   - **Prediction**: The classifier with the highest confidence score determines the output class.
   - **Considerations**: OvR can be computationally efficient but may face challenges with imbalanced datasets and overlapping classes.

2. **One-vs-One (OvO)**:
   - **Approach**: Train a binary classifier for every pair of classes, resulting in $ \frac{K(K-1)}{2} $ classifiers.
   - **Prediction**: Use a voting mechanism where each classifier votes for one class; the class with the most votes is selected.
   - **Considerations**: OvO can handle class imbalances better but may become computationally intensive as $ K $ increases.

3. **Direct Multiclass Classification**:
   - **Approach**: Utilize algorithms inherently designed for multiclass problems, such as Decision Trees, Random Forests, or Neural Networks.
   - **Considerations**: These models can capture complex relationships but may require more data and computational resources.

**Evaluation Metrics**:

- **Accuracy**: The proportion of correctly predicted instances among all instances.
- **Precision, Recall, and F1-Score**: Especially important in imbalanced datasets to assess the quality of predictions for each class.
- **Confusion Matrix**: A tabular representation showing the actual vs. predicted classifications, aiding in the visualization of model performance across classes.