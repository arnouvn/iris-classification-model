from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#import classification algorithm
from sklearn.neighbors import KNeighborsClassifier

#Import classification_report and confusion_matrix to evaluate model performance 
from sklearn.metrics import classification_report, confusion_matrix

#immport matplotlib to plot data
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

#load iris dataset
iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# print("X_train shape: ", X_train.shape)

# # Create a DataFrame for easier data manipulation
# iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# iris_df['species'] = iris.target

# # Replace numeric species codes with descriptive names
# species_map = dict(zip(range(3), iris.target_names))
# iris_df['species'] = iris_df['species'].map(species_map)

# # Display the first few rows of the DataFrame
# print(iris_df.head())

# # Display basic statistical details
# print(iris_df.describe())

# # Show the distribution of species
# print(iris_df['species'].value_counts())

# # Randomly sample 10 entries from the dataset
# sampled_data = iris_df.sample(n=10, random_state=42)
# print(sampled_data)

# # Plot the data
# plt.scatter(iris_df['sepal length (cm)'], iris_df['sepal width (cm)'], c=iris.target)
# plt.xlabel('sepal length (cm)')
# plt.ylabel('sepal width (cm)')
# plt.show()

#Create instance of KNN classifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)

#Make Prediction
Y_pred = knn.predict(X_test)

#Calculate and print confusion matrix and classification report
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()