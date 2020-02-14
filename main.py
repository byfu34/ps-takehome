from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd

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
plt.show()

plt.figure(2)
plt.scatter(petal_length, petal_width, c=y, cmap=plt.cm.Set1)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.show()