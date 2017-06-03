# 0. Import pandas, numpy
#    from sklearn, import train_test_split, accuracy_score

# 1. Load Dataset 
filename = None
df = None

# 2. Create matrix X containing the features
#    Create target vector y containing the classes
#    Note: df.iloc returns a DataFrame
#          df.values returns a Numpy array
X = None
y = None

# 3. Split the dataset
X_train, X_test, y_train, y_test = None, None, None, None

# 4. KNearestNeighbor Algorithm
#    Note that the steps listed are just guidelines. 
#    You're free to implement the KNN in any way you choose

def predict(X_train, y_train, x_test, k):
	"""
	Takes as input the training features X_train, the correspond training labels y_train,
	a new observation x_test, and the hyperparameter k.

	Returns the predicted label of x_test.
	"""

	distances = []
	# 4.1 Compute distance of x_test to every training example in X_train  
	#---------------------------------------------------------------------
	#     Create a for loop iterating over the rows of X_train.
	#     e.g. For each row i within the total number of rows in X_train,
	#          4.1.1 Set x_train = the training example at row i
	#                Question: How do you access the training example of X_train at row i?
	#          4.1.2 Compute the euclidean distance between x_test and x_train
	#                You can try googling how to do this. :D
	#          4.1.3 Get the corresponding training label of the training example at row i.
	#          4.1.4 Append the tuple (computed_euclidean_distance, corresponding_training_label) to distances

	# 4.2 Sort elements in list 'distances' in ascending order
	#----------------------------------------------------------

	# 4.3 Define k_neighbors to be the k nearest neighbors of x-test 
	#----------------------------------------------------------------
	# (i.e. k_neighbors = first k elements in the sorted distances list)
	k_neighbors = []

	# 4.4 Perform a majority vote on nearest k neighbors
	#--------------------------------------------------------------
	#     Define votes to be a dictionary containing the class labels as keys and the number of votes as values
	#     Initially, set the votes for each label to zero.
	#     i.e. {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}
	#     Note: this shouldn't be hard-coded. 
	#     Tip: To get the list of the unique values of y, use:
	#          - np.unique(y) - for Numpy arrays
	#          - y.unique() - for DataFrames
	votes = dict()

	# Each element in k_neighbors should be a tuple (distance, label)
	# where distance is its euclidean distance from x_test and
	#       label is its corresponding class label
	# 4.4.1 For each tuple_ in k_neighbors,
	#       4.4.1.1 Set label = the second element in the tuple
	#       4.4.1.2 Increment votes[label] by one

	# 4.5 Return the key with the highest value in the votes dictionary
	return None

y_pred = []
# 5. Create a for loop iterating over the rows of X_test.
#    e.g. For each row i within the total number of rows in X_test:
#       5.1 Set x_test = the test example at row i
#       5.1 Predict the label of x_test using our predict() function
#       5.2 Append the predicted label to y_pred
print("KNN Accuracy: " + str(accuracy_score(y_test, y_pred)))
