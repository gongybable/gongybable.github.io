Implement Gradient Descent in Python

[Input Data](binary.csv)

## Data Preparation
```python
import numpy as np
import pandas as pd

np.random.seed(42)

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank and admit
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = pd.concat([data, pd.get_dummies(data['admit'], prefix='admit')], axis=1)
data = data.drop(['rank', 'admit'], axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop(['admit_0', 'admit_1'], axis=1).values, pd.concat([data['admit_0'], data['admit_1']], axis=1).values
features_test, targets_test = test_data.drop(['admit_0', 'admit_1'], axis=1).values, pd.concat([test_data['admit_0'], test_data['admit_1']], axis=1).values
```

## Run Gradient on Perceptron
```python
import numpy as np
from data_prep import features, targets, features_test, targets_test
# Setting the random seed
np.random.seed(42)

# use sigmoid function as activation function
def sigmoid(h):
    return 1 / (1 + np.exp(-h))

# predition for the probability of each class
def prediction(x, weights):
    P1 = sigmoid(np.dot(x, weights))
    P2 = 1 - P1
    return P1, P2

# error / loss function
def lossFunction(outputs, y):
    # calculate loss for each pair of results
    def calculateLoss(data):
        output = data[0]
        result = data[1]
        loss = 0
        for i in range(len(output)):
            loss += result[i] * np.log(output[i])
        return loss

    # get the total loss
    losses = [calculateLoss(item) for item in zip(outputs,y)]
    return -np.mean(losses)

def updatePerceptron(X, y, W, learn_rate = 0.005):
    del_w = np.zeros(W.shape)
    n_records = len(X)
    for i in range(n_records):
        P1, P2 = prediction(X[i], W)
        # error for the probability of class 1 and class 2 is the same
        # so we can calculate error with any one of them
        error = y[i][0] - P1
        # The gradient descent step, the error times the gradient times the inputs
        del_w += error * X[i]
    # Update the weights here.
    W += learn_rate * del_w
    return W

def trainAlgorithm(X, y, learn_rate = 0.005, num_epochs = 1000):
    n_records, n_features = X.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        weights = updatePerceptron(X, y, weights, learn_rate)
        # Printing out the mean square error on the training set
        if i % (num_epochs / 10) == 0:
            P1, P2 = prediction(X, weights)
            outputs = zip(P1, P2)
            loss = lossFunction(outputs, y)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
    return weights

# train the network
weights = trainAlgorithm(features, targets, 0.005, 1000)
# Calculate accuracy on test data
P1, P2 = prediction(features_test, weights)
predictions = P1 > 0.5
accuracy = np.mean(predictions == targets_test[:,0])
print("Prediction accuracy: {:.3f}".format(accuracy))
```
The running result is:
```
Train loss:  0.6613580375785758
Train loss:  0.5782721399941655
Train loss:  0.5782705527073997
Train loss:  0.5782705525116583
Train loss:  0.5782705525116338
Train loss:  0.5782705525116338
Train loss:  0.5782705525116338
Train loss:  0.578270552511634   WARNING - Loss Increasing
Train loss:  0.5782705525116338
Train loss:  0.5782705525116338
Prediction accuracy: 0.725
```