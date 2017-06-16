"""
Adult Census Income Data Set: datasets/adult.csv
Prediction task is to determine whether a person makes over 50K a year.
See for more information: # https://www.kaggle.com/uciml/adult-census-income
"""

# 1. Import relevant libraries (don't forget the scikit preprocessing library)

# 2. Load adult.csv dataset
filename = ...
df = pandas.read_csv(...)

# 3. Inspect your data. 
#	 Notice that some values are '?'. These are missing values. 

# To check which features have missing values '?':
for feature in df.columns.values:
	if '?' in df[feature].unique(): #unique values of each feature
		print(feature)
# Note this will cause a warning. 
# This is due to comparing the string '?' to numerical data in features like age. Just ignore it.

#. 4. You have the option to remove rows with missing values:
#	  e.g. df = df[df['workclass'] != '?']
# Warning: Removing rows with missing values isn't ideal since you may lose valuable information by doing so.
# 		   You can try using Imputation to guess missing values.
# 		   Refer to the previous sessions for how to deal with missing values.

# 5. Extract features X and class labels y.
X = ...
y = ...

# Show a description of your categorical data
print(X.describe(include=['O'])) # include=['O'] means include objects. 

# Notice how education_num is an encoded version of education. This is a redundant feature.
# 6. Delete the column 'education' from the dataframe.
# 	 See this link for all the ways to remove or drop a dataframe column: http://stackoverflow.com/questions/13411544/delete-column-from-pandas-dataframe

# 7. Identify features with binary, ordinal, and nominal values 
# 	 Using the techniques presented in https://gitlab.com/issatingzon/WWCodeManila-ML.AI/blob/master/tutorials/categorical_data.md,
# 	 Transform the values into numerical data.
# 	 For ideas on how to do this, you can also refer to: https://www.kaggle.com/bananuhbeatdown/multiple-ml-techniques-and-analysis-of-dataset

# 8. Optional: native.country features
# 	 This has 42 unique values. Since it's nominal type, this means 42 additional features. Having a larger number of features than necessary may not always be great.
# 	 Notice how United-States has the largest frequency.
# 	 Try making native.country into a binary vector where the value is 1 if df['native.country'] == 'United-States' and 0 otherwise.

# Check out your features:
print(X.columns.values)

# 9. You can try scaling thefeatures as we did in the previous session.

# 10. Split your dataset, train you favorite classifiers, and evaluate its accuracy. 
#     This might run reaaally slowly since it's a huge dataset. 
#     You can try limiting the number of rows by modifying the size of the dataset you want to use as follows:
#		X = df.iloc[0:10000, :-1] --> use the first 10,000 rows of the dataset
#		y = df.iloc[0:10000, -1]