"""
Using our iris classification code from the first session (complete code can be found in: samples/iris_script.py) as a template,
create a Handwritten Digit Recognizer using the mnist dataset (dataset/mnist.csv) using different classifiers.

About the MNIST dataset:
The data files mnist.csv contains grey-scale images of hand-drawn digits, from 0 through 9. 
Each image is 28 x 28 pixels, for a total of 784 pixels. 
Each pixel has a single pixel-value, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. 
Each pixel-value is an integer between 0 and 255, inclusive.

More information here: https://www.kaggle.com/c/digit-recognizer/data

Features/attributes:  784 pixels values
Classes: Labels 0-9 (first column of the dataset)

Tip: After loading the dataset, you can visualize an instance i in [0, 42000] using the following code snippet:

i = 8 #8th row in the dataset
image_data = dataframe.iloc[:, 1:]
img = image_data.iloc[i].as_matrix()
img = img.reshape((28, 28))
plt.imshow(img, cmap='gray_r')
plt.title('Class label: ' + str(dataframe.iloc[i, 0]))
plt.show()

Enjoy!
"""

filename = '../datasets/mnist.csv'