import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Load dataset
filename = '../datasets/iris.csv' #locates the iris.csv file in the /datasets folder
df = pd.read_csv(filename, header=0) 

# It's always a good idea to check out your data
print(df.columns.values) # Column values (headers)
print(df.shape) # dimensions of the dataframe
print(df.head(10)) # Inspect first ten data instances
print(df.groupby('Species').size()) # The number of data instances belonging to each class

# Visualize data
scatter_matrix(df) 
plt.show()

# Create matrix X containing the features
X = df.iloc[:, :4] 
# Create target vector y containing the classes
y = df.iloc[:, 4] # Alternatively: y = df['Species']

# Split the dataset: Use 80% for training and hold back 20% for validation (or testing)
train_size = 0.8
seed = 7 # for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=seed)

# Instantiate learning model
clf = GaussianNB()

# Fit model to training set
clf.fit(X_train, y_train)

# Predict labels of the test set
y_pred = clf.predict(X_test)

# Compare predicted labels y_pred and true test labels y_test
results = pd.DataFrame({'Predicted label': y_pred, 'True label': y_test})
print(results)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred) # Alternatively: accuracy = np.mean(y_pred == y_test) 
print("Gaussian Naive Bayes: " + str(accuracy))

# Or, more concisely (predicts y_pred and evaluates accuracy in a single line of code)
accuracy = clf.score(X_test, y_test)
print("Gaussian Naive Bayes: " + str(accuracy))

# To evaluate multiple classifiers, store classifiers into a dictionary:
classifiers = dict() 
classifiers['Gaussian Naive Bayes'] = GaussianNB()
classifiers['Decision Tree Classifier'] = DecisionTreeClassifier(random_state=seed)
classifiers['Support Vector Machines'] = SVC()	

# Iterate over dictionary
for clf_name, clf in classifiers.items(): #clf_name is the key, clf is the value
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print(clf_name + ': ' + str(accuracy))