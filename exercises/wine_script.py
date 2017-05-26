""""
About the dataset: wine.csv
These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. 
The analysis determined the quantities of 13 constituents found in each of the three types of wines

The data consists of:
	13 features (see dataset/wine.headers)
	3 classes - types of wine (I'm not sure what these types particularly are. In the dataset, they're just denoted as 1, 2, and 3. Sarreh.) 

See more detail here:
https://archive.ics.uci.edu/ml/datasets/wine
"""

# 1. Import all libraries that we used from the previous session
# 	  plus the scikit-learn library for data processing

# 2. Load the dataset wine.csv located in the /datasets folder.
#	 Set header=None since the dataset has no headers.
filename = ...
df = pandas.read_csv(...)

# The headers are actually stored in the file /datasets/wine.names. 
# We load this file as follows:
headers = []
with open('datasets/wine.names','r') as file:
    for line in file:
    	# 3. Append each line to the 'headers' list.
    	#    Code for appending an element to a list is: list.append(element) 
    	# 	 Important: Strip each line of line breaks(\n) and trailing white spaces. 
    	#               i.e. append line.strip() to headers

df.columns = headers #Set the headers

# 4. Print the first 10 instances of the dataframe using head() 
#    just to check that we loaded the data and headers properly.

# 5. Extract features and store them in X.
#    Extract class labels and store them in y.
#    Tip: Instead of using df.values(), you can use df.iloc(). 
#	    - df.values() returns an NumPy array.
#	    - df.iloc() returns a DataFrame.
# 	    With DataFrames, you can access certain features using their column names/headers. 
#	    This will be useful for us later.
X = ...
y = ...

# 5.1. 	You can check the data type of X by printing type(X). This should be a Dataframe. 
# 		If you opted to use df.values to extract the features, you should cast X as a DataFrame:
# 				X = pd.DataFrame(X, columns = df.columns[1:]) 
# 		If you used df.iloc, you can skip this.

# 6. Scale X using standardization
X_standard = ...

print('Standardization')
for feature in X_standard:
	mean = X_standard[feature].mean() # get mean
	variance = X_standard[feature].std() # get variance
	print(feature + 'mean = {:.2f}, std = {:.2f}'.format(mean, variance))

#Each feature vector should have a mean = 0, variance = 1

# 7. Print the first 10 instances of X_standard (using head(), again) to see how our data has transformed

# 8. Scale X so that it is within the range 0 to 1.
X_minmax = ...

print('\n\nMin-max Scaling')
for feature in X_minmax:
	min_ = X_minmax[feature].min()
	max_ = X_minmax[feature].max()
	print(feature + 'min = {:.2f}, max = {:.2f}'.format(min_, max_))

# The lowest value (min) in each feature vector should be 0
# The highest value (max) should be 1.

# 9. Print the first 10 instances of X_minman (using head(), again) to see how our data has transformed


# We'll now plot the data to see (visually) how the data has been transformed.
# Here, we use only the first two attributes: Alcohol and Malic acid
def plot():
	plt.figure(figsize=(10, 10))
	feature1 = 'Alcohol' #alternatively, you can call headers[1] to avoid hard-coding
	feature2 = 'Malic acid' #headers[2]

	plt.scatter(X[feature1], X[feature2], color='green', label='raw input scale', alpha=0.5)
	plt.scatter(X_standard[feature1], X_standard[feature2], color='red', label='Standardized (mean centered on 0, variance = 1)', alpha=0.5)
	plt.scatter(X_minmax[feature1], X_minmax[feature2], color='blue', label='Scaled between 0 to 1 (MinMaxScaler)', alpha=0.5)
	plt.xlabel(feature1)
	plt.ylabel(feature2)
	plt.legend(loc='upper left')
	plt.show()
plot()

# Alright! It's time to compare the performances of the classifiers for the scaled and unscaled data.
def classify(string, X):
	print('\n' + string)

	# As we've done before, let's apply our machine learning algorithms!

	# 10. Split the dataset into training and testing data
	#    Set the train size to 80%, and hold back 20% for testing 
	#	 For random_state, set the seed to 7 (para uniform results natin lahat :D).

	# 11. 	Train X using different classifiers.
	#		I suggest using DecisionTreeClassifier, SVM, MLPClassifier. Add others!

	# 12. Calculate the accuracy for each classifier model, then print it. 


classify('Not scaled', X)
classify('Standardized', X_standard)
classify('Scaled to range 0 -1', X_minmax)

# Share your results! :D

"""
This tutorial is based on http://sebastianraschka.com/Articles/2014_about_feature_scaling.html. 
* This also shows you how to implement scaling manually (equations are given).
* If you want a more detailed explanation, check it out!
"""
