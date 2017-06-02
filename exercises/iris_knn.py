import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

###################################
# PART 1: MACHINE LEARNING BASICS #
###################################

# 1. Load Dataset 
filename = None
df = None

# 2. Create matrix X containing the features
# 	 Create target vector y containing the classes
#	 Tip: Use .iloc instead of .values 
#	    - df.values() returns an NumPy array.
#	    - df.iloc() returns a DataFrame.
X = None
y = None

# 3. Split the dataset into X_train, X_test, y_train, y_test 
# 	 Set the train_size = 0.8
# 	 Set seed (random_state) = 7 for reproducibility

# 4. Instantiate Learning Model for KNN. 
# 	 Set K = 3 (i.e. n_neighbors)

# 5. Fit KNN model to training set

# 6. Predict labels of the test set using the KNN model

# 7. Evaluate accuracy

##################################################
# PART 2: Parameter Tuning with Cross Validation #
##################################################

# 1. Import (at the beginning of the script) the sklearn library for cross validation: cross_val_score

# 2. Create a list neighbors (a list of k's) which will be a list of odd numbers from 1 to 50:
# 	 *We want k's to be odd in order to avoid ties.*
# 	 e.g. neighbors = [1, 3, 5, 7, 9, 11, ..., 49]
neighbors = []
# print neighbors for debugging

cv_scores = []
# 3. For each k in neighbors,
# 	 	3.1 Instantiate a learning model for KNN and set n_neighbors to k
# 	 	3.2 Perform cross validation using the cross_val_score function (set number of folds cv=10, scoring='accuracy')
# 	 	3.3 cv_scores.append(average of the list of scores returned by cross_val_score)

# 4. Get the misclassification errors and store them in list 'mse'
#    For each cv_score in cv_scores:
# 		mse.append(misclassification error), where the misclassification error is equal to (1- cv_score)
mse = []

# 5. Get the optimal k
# 	 5.1. Set min_val = minimum value in mse (code for getting minimum value is: min(mse))
# 	 5.2. Set index_min = index of the minimum value in mse (mse.index(min_val))
# 	 5.3. Set optimal_k as the value of neighbor at index_min
optimal_k = None

# Plotting misclassification errors vs neighbors (k's)
# Uncomment this part to display graph
"""
plt.plot(neighbors, mse)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
"""

# For more information: https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/


