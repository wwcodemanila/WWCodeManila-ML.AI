import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 0. Import pandas, numpy
#    from sklearn, import train_test_split, accuracy_score

# 1. Load Dataset 
filename = None
df = None

# 2. Create matrix X containing the features
# 	 Create target vector y containing the classes
X = None
y = None

# 3. Split the dataset
X_train, X_test, y_train, y_test = None, None, None, None

# 4. KNearestNeighbor Algorithm
#    Note that the steps listed are just guideline. 
#    You're free to implement the KNN in any way you choose

def predict(X_train, y_train, x_test, k):
	"""
	Takes as input the training features X_train, the correspond training labels y_train,
	a new observation x_test, and the hyperparameter k.

	Returns the predicted label of x_test.
	"""

	distances = []
	# Compute distance of x_test to every training example in X_train  
	# 4.1 Create a for loop iterating over the rows of X_train.
	#     e.g. For each row i in the range (0, len(X_train)): 
	#          4.1.1 Set x_train = the training example at row i
	#                Question: How do you access the training example of X_train at row i?
	#          4.1.2 Compute the euclidean distance between x_test and x_train
	#                Tip: There are a number of ways to do this. 
	#                     - One way is to use np.linalg.norm
	#                     - Or you can use the formula for euclidean distance
	#                     You can try googling how to do this. :D
	#          4.1.3 Append the tuple (computed_euclidean_distance, i) to distances

	# 4.2 Sort elements in list 'distances' in ascending order

	# 4.3 Define k_neighbors to be the first k elements in the sorted distances list
	k_neighbors = []

	# 4.4 Perform a majority vote on nearest neighbors k_neighbors
	#     Define votes to be a dictionary containing the class labels as keys and the votes as values
	#     Initially, set the votes for each label to zero.
	#     i.e. {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}
	#     Note: this shouldn't be hard-coded. 
	#     Tip: To get the list of the unique values of y, use:
	#          - np.unique(y) - for Numpy arrays
	#          - y.unique() - for DataFrames
	votes = dict()

	# Each element in k_neighbors should be a tuple (distance, i)
	# where distance is its euclidean distance from x_test and
	#       i is its corresponding index in the list X_train
	# 4.4.1 For each tuple_ in k_neighbors:
	#       4.4.1.1 Set index i = the second element in the tuple_
	#       4.4.1.2 Set label = the value of y_train at index i
	#       4.4.1.3 Increment votes[label] by one
	#       Note: Try printing neighbor in this part

	# 4.5 Return the key with the highest value in the votes dictionary
	#     Go ahead and google how to do this! xD Googling solutions is a huge part of programming :P
	return None

y_pred = []
# 5. Create a for loop iterating over the rows of X_test.
#    e.g. For each row i in the range (0, len(X_test)):
#       5.1 Set x_test = the test example at row i
#       5.1 Predict the label of x_test using our predict() function
#       5.2 Append the predicted label to y_pred
print("KNN Accuracy: " + str(accuracy_score(y_test, y_pred)))
