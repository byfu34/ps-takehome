import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
# Load iris dataset
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Convert iris to pandas DataFrame
iris_df['target'] = pd.Series(iris.target)
# Format the target column of the DataFrame

sepal_length = iris_df['sepal length (cm)']
sepal_width = iris_df['sepal width (cm)']
petal_length = iris_df['petal length (cm)']
petal_width = iris_df['petal width (cm)']

y = iris_df['target']

# Data Visualization
plt.figure(1)
plt.scatter(sepal_length, sepal_width, c=y, cmap=plt.cm.Set1)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
# plt.show()

plt.figure(2)
plt.scatter(petal_length, petal_width, c=y, cmap=plt.cm.Set1)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
# plt.show()

# model = KNeighborsClassifier(n_neighbors=1)
# Initialize K-nearest neighbors model with hyperparameter of 3 nearest neighbors
# features = iris_df[['petal length (cm)', 'petal width (cm)']]
# Set petal length and petal width as features for the model due to distinct separations for these variables in visualizations
# Mean cross validation score = .974

model = GaussianNB()
# Initialize Gaussian Naive Bayes model
features = iris_df[['petal length (cm)', 'petal width (cm)', 'sepal length (cm)', 'sepal width (cm)']]

model.fit(features, y)
# Fit the model
cross_val_scores = cross_val_score(model, features, y, cv=10)
# Calculate 10-fold cross validation scores
print(cross_val_scores)
# Print array of scores from a 10-fold cross validation
print(np.mean(cross_val_scores))