# Encoding Categorical Data
So far, we've been dealing mostly with numerical data. What happens when we have categorical data? 

Classifiers generally accept discrete (integer) or continous (floats) input. However, not all classifiers can process string input like "Male" or "Female".  Therefore, we need a way to convert these categorical data into numerical data. 

First we need to identify if the data is nominal or ordinal. 
* **Nominal data** - no intrinsic ordering
	* Hair color: red, black, blonde
	* Race: Asian, Hispanic, African
	* OS type: Chrome, Firefox, IE, Safari

* **Ordinal** - there exits an ordering
	* Highest attained educational background: Elementary, High School, College
	* Military Rank: Captain, Major, Colonel, General
	* Ratings: Poor, Neutral, Good, Excellent

Note: These techniques are generally used for features, not class labels. 

## Binary Categories
For features with only two unique value, the most common technique is to use a binary encoding.

Example: For the feature `gender`, we encode `Female as 1` and `Male as 0`.

 user ID | education | gender | eye_color
 ------- |-----------| ----|----------
 1	| Elementary | Female | blue
 2	| College    |   Male | brown
 3	| High School| Female | green
 4	| College    |   Male | brown
 5	| High School| Female | green

 user ID | education | gender | eye_color
 --- |---| ---|---
 1	| Elementary | 1 | blue
 2	| College    | 0 | brown
 3	| High School| 1 | green
 4	| College    | 0 | brown
 5	| High School| 1 | green

There are a number of way to do this. Listed are just some ways:

### Using `LabelEncoder`
```python
encoder = preprocessing.LabelEncoder()
data['gender'] = encoder.fit_transform(data['gender'])
```

### Using Panda's `factorize()`
```python
data['sex'] = pd.factorize(data['sex'])[0] 
```

### Using Numpy's `where()`
```python
import numpy as np 
data['gender'] = np.where(data['gender'] == 'Female', 1, 0)
```

### Casting the feature as a category
```python
data['gender'] = data['gender'].astype("category")
data['gender'].cat.categories = [0,1]
```
[See this link for additional info.](https://pandas.pydata.org/pandas-docs/stable/categorical.html)

## Ordinal Data
A possible way to encode ordinal data is to convert the input into numerical variables.

Example: For the ordinal feature `education` with unique values `Elementary, High School, and College`, we can map each value as follows:
* Elementary -> 0
* High School -> 1
* College -> 2

 user ID | education | gender | eye_color
 --- |---| ---|---
 1	| Elementary | 1 | blue
 2	| College    | 0 | brown
 3	| High School| 1 | green
 4	| College    | 0 | brown
 5	| High School| 1 | green

 user ID | education | gender | eye_color
 --- |---| ---|---
 1	| 0| 1 | blue
 2	| 2|   0 | brown
 3	| 1| 1 | green
 4	| 2|   0 | brown
 5	| 1| 1 | green

This can be done by manually mapping the values of each feature to a number as follows:
```python
data['education'] = data['education'].map({'Elementary': 0, 'High School': 1, 'College': 2})
```

We can use `LabelEncoder` again:
```python
encoder = preprocessing.LabelEncoder()
data['education'] = encoder.fit_transform(data['education'])
```

But note that it's a good idea to let the numerical values reflect the order or ranking: `elementary < high school < college`. `LabelEncoder` may not necessarily capture this intrinsic ordering.

## Nominal Data
While it's possible to assign numerical attributes to each features, this might not always be a good ideas since classifiers can interpret the categories as being ordered, which is not the case for nominal data.

One possible technique is to transform each categorical feature with `m` possible values (i.e. `m` levels) into `m` binary *dummy features*, with only one active.

Example: For the nominal feature `eye_color` with values `blue, brown and green`, we create *dummy features* and call them `eye_color.blue`, `eye_color.brown`, and `eye_color.green`.
 
 user ID | education | gender | eye_color
 --- |---| ---|---
 1	| 0| 1 | blue
 2	| 2|   0 | brown
 3	| 1| 1 | green
 4	| 2|   0 | brown
 5	| 1| 1 | green

  user ID | education | gender | eye_color. blue |eye_color.brown | eye_color.green 
 --- |---| ---|--- | --- | ---
 1	| 0| 1 | 1 |0 | 0
 2	| 2|   0 | 0| 1 | 0
 3	| 1| 1 | 0| 0 | 1
 4	| 2|   0 | 0| 1 | 0
 5	| 1| 1 | 0 | 0 | 1

 One way to do this is by using `OneHotEncoder`. [See this tutorial for implementation.](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)  Note that OneHotEncoder cannot process string values directly. If your nominal features are strings, then you need to first map them into integers (using `LabelEncoder()`, perhaps).

 Another way is to use panda's `get_dummies()` method. Suppose we have another nominal feature called `hair_color`. The way to code this would be:
 ```python
import pandas as pd
nominals = ['eye_color', 'hair_color']
data = pd.get_dummies(data, prefix=nominals, columns=nominals)
```
This creates additional features `eye_color.brown`, `eye_color.blue`, ..., `eye_color.brown`, `hair_color.brown`, ... `hair_color.blonde` with binary values (indicating True or False for each feature). 

Note that pandas.get_dummies, unlike 'OneHotEncode', only converts string columns into one-hot representation, unless columns are specified.
([Source](https://stackoverflow.com/questions/36631163/pandas-get-dummies-vs-sklearns-onehotencoder-what-is-more-efficient))

## Resources
* [Nominal vs Ordinal Encoding](http://stackoverflow.com/questions/34087329/categorical-and-ordinal-feature-data-representation-in-regression-analysis)
* [Simple Methods to deal with Categorical Variables in Predictive Modeling](https://www.analyticsvidhya.com/blog/2015/11/easy-methods-deal-categorical-variables-predictive-modeling/)
* [Kaggle discussion for Adult Dataset](https://www.kaggle.com/bananuhbeatdown/multiple-ml-techniques-and-analysis-of-dataset)

