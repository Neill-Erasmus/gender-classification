# Gender Classification

Comparing different classification models on the same dataset to classify people into genders based on several features.

## Classification

Classification in machine learning refers to the process of categorizing data points into predefined classes or categories based on their features. It is a supervised learning task, meaning that the algorithm learns from labeled training data, where each data point is associated with a class label. The goal of classification is to build a model that can accurately predict the class labels of unseen data points.

## Classification Models

### Logistic Regression

Logistic regression is a statistical method used for binary classification, which means it's used when the target variable (the variable you're trying to predict) has only two possible outcomes or classes. Despite its name, logistic regression is used for classification, not regression.

### K-Nearest Neighbours

k-Nearest Neighbors (k-NN) is a simple and versatile machine learning algorithm used for both classification and regression tasks. It's a non-parametric method, meaning it doesn't make any assumptions about the underlying data distribution. Instead, it uses the entire training dataset to make predictions.

### Support Vector Machine (SVM)

Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification, regression, and outlier detection tasks. It's particularly effective in high-dimensional spaces and when the number of features exceeds the number of samples.

### Kernel SVM

Kernel SVM, or Support Vector Machine with a kernel, is an extension of the traditional SVM algorithm that allows for non-linear decision boundaries by implicitly mapping the input data into a higher-dimensional feature space. This transformation enables the SVM to handle complex relationships between features and improve its classification performance.

### Naive Bayes

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' theorem, which predicts the probability of a given data point belonging to a particular class based on the features of that data point. Despite its simplicity, Naive Bayes often performs well in classification tasks, especially in text classification and spam filtering.

### Decision Tree Classification

Decision tree classification is a supervised learning algorithm used for both classification and regression tasks. It works by recursively partitioning the feature space into regions and assigning class labels to those regions based on the majority class of the training samples within each region. Decision trees are intuitive, easy to interpret, and can handle both numerical and categorical data.

### Random Forest Classification

Random Forest is an ensemble learning technique based on decision trees, used primarily for classification and regression tasks. It builds multiple decision trees during training and combines their predictions to improve accuracy and reduce overfitting. Random Forest is highly flexible, robust, and suitable for a wide range of applications.

## Dataset Overview

The dataset contains features related to facial characteristics and gender classification. Here's an overview of the features:

1. long_hair - This feature indicates whether a person has long hair or not. It is likely represented as a binary variable, where 1 could denote the presence of long hair and 0 the absence.

2. forehead_width - This feature represents the width of the forehead of the individuals in the dataset. It's a numerical variable, measured in some appropriate unit of length.

3. forehead_height - This feature represents the height of the forehead of the individuals in the dataset. Similar to forehead width, it's a numerical variable.

4. nose_width - This feature represents the width of the nose of the individuals in the dataset. It's a numerical variable, measured in some appropriate unit of length.

5. nose_length - This feature represents the length of the nose of the individuals in the dataset. Similar to nose width, it's a numerical variable.

6. thin_lips - This feature indicates whether a person's lips are thin or not. Like the long hair feature, it could be represented as a binary variable, with 1 indicating thin lips and 0 indicating otherwise.

7. distance_nose_to_lip - This feature represents the distance from the nose to the lips of the individuals in the dataset. It's a numerical variable, measured in some appropriate unit of length.

8. gender - This feature is the target variable that indicates the gender of the individuals in the dataset. It could be represented as a binary variable, where 1 denotes male and 0 denotes female.

The dataset appears to be intended for gender classification based on facial features. The provided features, such as forehead width and height, nose width and length, and the presence of long hair or thin lips, could be used as input to train a machine learning model to predict gender.

[DATASET](https://www.kaggle.com/datasets/elakiricoder/gender-classification-dataset)

## Evaluating the Various Classification Models

1. Decision Tree Classification: Approximately 96.5%
2. Logistic Regression: Approximately 97.5%
3. K-Nearest Neighbors (KNN): Approximately 95.9%
4. Kernel SVM: Approximately 97.5%
5. Naive Bayes: Approximately 97.5%
6. Random Forest Classification: Approximately 97.4%
7. Support Vector Machine (SVM): Approximately 97.4%

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.