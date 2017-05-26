# Kaggle
Kaggle is a platform for data science competitions. There are tons of datasets available on this site, making it an ideal place to start studying and practicing machine learning. 

Once you've done the [Handwritten Digit Recognition exercise](https://gitlab.com/issatingzon/WWCodeManila-ML.AI/blob/master/exercises/mnist_script.py), you're ready to submit your first predicition to Kaggle. 

You can check out the [Kaggle competition for Digit Recognizer here](https://www.kaggle.com/c/digit-recognizer#description).

## Your First Kaggle Submission
1. Create a new profile in Kaggle
2. [Download the training and test data here.](https://www.kaggle.com/c/digit-recognizer/data) This includes three files:
	- train.csv
	- test.csv
	- sample_submission.csv
3. Modify your code for the [Handwritten Digit Recognition exercise](https://gitlab.com/issatingzon/WWCodeManila-ML.AI/blob/master/exercises/mnist_script.py) as follows:
	- Load `train.csv` as your training set.
	- Train your classifier on the entire training set. This means omitting the `train_test_split` part of your code since both training and test sets are given already.
	- Using your best classifier, predict the labels for the test set `test.csv` and save it to a new file `result.csv` as follows:
		```shell
		test_data = pd.read_csv('test.csv')
		results = clf.predict(test_data[:])

		df = pd.DataFrame(results)
		df.index += 1 #indexing starts from 1
		df.index.names = ['ImageId']
		df.columns = ['Label']
		df.to_csv('results.csv', header=True)
		```
4. [Submit your predictions (results.csv) here](https://www.kaggle.com/c/digit-recognizer/submit) and wait for Kaggle to score it. :D