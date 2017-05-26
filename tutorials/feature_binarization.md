# Binarize Features in MNIST

Scale the features so that each value is either 0 if pixel value = 0, and 1 if pixel value > 0.

You can check what an image looks like after feature binarization using the following code.

```python
i = 8
img = X_binarized.iloc[i].as_matrix()
img = img.reshape((28, 28))
plt.imshow(img, cmap='binary')
plt.title('Class label: ' + str(labels[i]))
plt.show()
```

Note that you might need to cast `X_binarized` as a DataFrame. 
```python
X_binarized = pd.DataFrame(X_binarized) 
```

Check the accuracy before and after scaling. 

(Note: SVM has notable results.)