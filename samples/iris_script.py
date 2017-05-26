import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Load dataset
filename = '../datasets/iris.csv' #locates the iris.csv file in the /datasets folder
dataframe = pd.read_csv(filename, header=0) 

# It's always a good idea to check out your data
print(dataframe.columns.values) # Column values (headers)
print(dataframe.shape) # dimensions of the dataframe
print(dataframe.head(10)) # first ten data instances
print(dataframe.groupby('Species').size()) # The number of data instances belonging to each class

# Visualize data
scatter_matrix(dataframe) 
plt.show()

# Extract features
X = dataframe.iloc[:, :4] 
#Extract classes
y = dataframe.iloc[:, 4] 

# Splits the dataset: use 80% for training and hold back 20% for validation
train_size = 0.8
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=seed)

# Instantiate model
clf = GaussianNB()

# Fit model to training set
clf.fit(X_train, y_train)
print(clf)

# Predict y for test set
y_pred = clf.predict(X_test)

# Compare predicted labels y_pred and true test labels y_test
results = pd.DataFrame({'Predicted label': y_pred, 'True label': y_test})
print(results)

# Evaluate results
accuracy = np.mean(y_pred == y_test)
print("Gaussian Naive Bayes: " + str(accuracy))

# Or, more concisely:
accuracy = clf.score(X_test, y_test)
print("Gaussian Naive Bayes: " + str(accuracy))

# To evaluate multiple classifiers, store classifiers into a dictionary
classifiers = dict() 
classifiers['Gaussian Naive Bayes'] = GaussianNB()
classifiers['Decision Tree Classifier'] = DecisionTreeClassifier(random_state=seed)
classifiers['Support Vector Machines'] = SVC()	

# Iterate over dictionary
for clf_name, clf in classifiers.items(): #clf_name is the key, clf is the value
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print(clf_name + ': ' + str(accuracy))