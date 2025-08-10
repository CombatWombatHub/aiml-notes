# Classification Methods

List basic notes on various types of models that can be used for Classification tasks

- most of these will assume you have already split data into
    - `X`: features (inputs)
    - `y`: target (output)

## k-Nearest Neighbors (knn)

- binary output like "will this customer leave (`1`) or stay (`0`) based on account `age` and `customer service call count`"
- sets the result for the target based on nearby points (must be an odd number of points to prevent a "tie")
- it determines "nearness" by mapping the inputs in vector space and computing the length of the vector between each training data point and test data point

```python
from sklearn.neighbors import KNeighborsClassifier
# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)
# Fit the classifier to the data
knn.fit(X, y)
```