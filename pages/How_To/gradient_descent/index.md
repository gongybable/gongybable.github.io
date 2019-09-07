# Implement Gradient Descent in Python

[Input Data](binary.csv)

## Data Preparation
```python
import numpy as np
import pandas as pd

np.random.seed(42)

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank and admit
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop(['rank'], axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop(['admit'], axis=1).values, data['admit'].values.reshape(1, -1)
features_test, targets_test = test_data.drop(['admit'], axis=1).values, test_data['admit'].values.reshape(1, -1)
```

## Run Gradient on Perceptron
```python
import numpy as np
from data_prep import features, targets, features_test, targets_test
# Setting the random seed
np.random.seed(42)

# initialize dimensions
def initialize_with_zeros(dim):
    w = np.random.normal(scale=1 / dim**.5, size=(1, dim))
    b = 0
    return w, b

# use sigmoid function as activation function
def sigmoid(h):
    return 1 / (1 + np.exp(-h))

# predition for the probability of each class
def prediction(X, w, b):
    return sigmoid(np.dot(w, X.T) + b)

def propagate(w, b, X, Y):
    m = X.shape[0]
    A = prediction(X, w, b)
    cost = -(np.sum(np.dot(Y,np.log(A).T) + np.dot((1-Y),np.log(1-A).T)))/m

    dw = np.dot((A-Y), X)/m
    db = np.sum(A-Y)/m

    cost = np.squeeze(cost)
    grads = {"dw": dw, "db": db}
    return grads, cost

def trainAlgorithm(X, y, learning_rate = 0.005, num_iterations = 1000):
    n_records, n_features = X.shape
    weights, b = initialize_with_zeros(n_features)

    costs = []
    for i in range(num_iterations):
        # In each epoch, we apply the propagate step.
        grads, cost = propagate(weights, b, X, y)
        dw = grads["dw"]
        db = grads["db"]
        weights = weights - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            
    params = {"w": weights, "b": b}

    return params, costs

# train the network
params, costs = trainAlgorithm(features, targets, 0.005, 1000)
print(costs)
# Calculate accuracy on test data
predictions = prediction(features_test, params["w"], params["b"])
accuracy = np.mean((predictions > 0.5) == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
```
The running result is:
```
[0.72369269440026818, 0.68967828561167899, 0.66448512007979776, 0.64578769741098929, 0.63184873843757094, 0.62139323271662716, 0.61349349527302166, 0.60747649363000644, 0.60285358577702997, 0.59926894972632561]
Prediction accuracy: 0.775
```