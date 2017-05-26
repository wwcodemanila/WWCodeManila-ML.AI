# Feature scaling

Different features can be measured on different scales. 

For example:
* height can be measure in centimeters
* weight in kilograms
* blood pressure in mmHg
* etc. 

Some classifiers combine and compare feature values, e.g. computing distance using Euclidean distance. 

The problem is that if one of the features has a broad range of values, the distance will be governed by this particular feature!

> For example, consider the two features:
> * the percentage of unemployment in a city - ranges from 0.0 to 1.0
> * the population of the city - can range up to 500,000
>
> Population will govern percentage by far. 

Scaling transforms the data so that the features have, more or less, uniform range.

> The main advantage of scaling is to avoid attributes in greater numeric ranges dominating those in smaller numeric ranges.

As we'll see later, scaling features can lead to a faster optimization process and better results. 

But for other algorithms like the Decision Tree Classifier, scaling is *not* necessary (i.e. it's scale-invariant). 

## Get started
Scikit-learn comes with a package for data preprocessing `sklearn.preprocessing`. Let's start by importing that.
```python
from sklearn import preprocessing
```

## Standardization 
Some machine learning algorithms (e.g. SVM) behave badly when their individual features don't look like standard normally distributed data (i.e. Gaussian distribution with zero mean and unit variance). 
<center><img src="https://docs-assets.developer.apple.com/published/568138fb95/gaussian_labeled_2x_b95dbf3b-42f2-44b8-8034-88835c10b3e4.png" alt="Drawing" width="400" /> </center>
Data scaled using standardization has zero mean and unit variance:

The function `scale` provides a quick and easy way to perform this operation on a single array-like dataset:
```python 
X = ... #code for extracting the features
X_standard = preprocessing.scale(X)
```
Another way to do this is to use `StandardScaler` as follows:
```python
scaler = preprocessing.StandardScaler().fit(X)
X_standard = scaler.transform(X)
```

## Scaling Features to a range
For some classifiers, it's useful to scale the features down to a value within the range 0 to 1. 

In scikit-learn, we use `MinMaxScaler` to scale the matrix to a `[0,1]` range:
```python 
scaler = preprocessing.MinMaxScaler()
X_minmax = scaler.fit_transform(X)
X_minmax = pd.DataFrame(X_minmax, columns = X.columns)
```

## Feature Binarization
Feature binarization is the process of thresholding numerical features to get boolean values.
```python
binarizer = preprocessing.Binarizer()
X_binarized = pd.DataFrame(binarizer.transform(X))
```
Basically, the feature values take on either a value of 0 or 1 (i.e. binary values).

# Resources
For more information, check out [the Scikit-learn page for data pre-processing](http://scikit-learn.org/stable/modules/preprocessing.html).








