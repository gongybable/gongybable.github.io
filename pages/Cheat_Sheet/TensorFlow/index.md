# TensorFlow

## Basic Concepts

### Session
TensorFlow’s api is built around the idea of a <a href="https://medium.com/tebs-lab/deep-neural-networks-as-computational-graphs-867fcaa56c9" download>computational graph</a>, a way of visualizing a mathematical process. A "TensorFlow Session" is an environment for running a graph.
```python
import tensorflow as tf
str = tf.constant('Hello World!')

with tf.Session() as sess:
    output = sess.run(hello_constant)
    print(output)
```

### Tensor
* In tensorflow, data (integers, floats, or strings) are encapsulated in an object called a **tensor**.

```python
# str is a 0-dimensional constant string tensor
str = tf.constant('Hello World!')
# A is a 0-dimensional constant int32 tensor
A = tf.constant(1234)
# C is a 2-dimensional constant int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])
```

* We can also use **_placeholder_** and **_feed_dict_** to set the input right before the session runs.

```python
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 123, y: 45.67})
```

* Set variables (placeholder and constant are not modifiable)

```python
x = tf.Variable(5)
# use global_variables_initializer to initialize the state of all the variable tensors
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    output = sess.run(x) #5
```

### Basic Math
```python
tf.cast(tf.constant(2.0), tf.int32) # convert float into integer

a = tf.add(5, 2)  # 7
b = tf.subtract(10, 4) # 6
c = tf.multiply(2, 5)  # 10
d = tf.divide(tf.constant(10), tf.constant(2)) # 5.0
with tf.Session() as sess:
    output1 = sess.run(a) # 7
    output2 = sess.run(b) # 6
    output3 = sess.run(c) # 10
    output4 = sess.run(d) # 5.0
```

### Model Training Tips
* When doing computations in computers, it is always better to have all the variables have zero mean, and small equal variance whenever possible.
    - large variance will result in peaks in probility distribution
    - small variance will have no peaks in probability, means the model is uncertain about things, which is good to start with

* When training a model and things do not work, the first thing to do is usually lower the learning rate.

### Tensor Flow Sample
```python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import math

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    """
    assert len(features) == len(labels)
    outout_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)
        
    return outout_batches

"""
An epoch is a single forward and backward pass of the whole dataset.
This is used to increase the accuracy of the model without requiring more data. 
"""
def print_epoch_stats(epoch_i, sess, last_features, last_labels):
    """
    Print cost and validation accuracy of an epoch
    """
    current_cost = sess.run(
        cost,
        feed_dict={features: last_features, labels: last_labels})
    valid_accuracy = sess.run(
        accuracy,
        feed_dict={features: valid_features, labels: valid_labels})
    print('Epoch: {:<4} - Cost: {:<8.3} Valid Accuracy: {:<5.3}'.format(
        epoch_i,
        current_cost,
        valid_accuracy))

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
valid_features = mnist.validation.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
valid_labels = mnist.validation.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Features and Labels
"""
The None dimension is a placeholder for the batch size.
At runtime, TensorFlow will accept any batch size greater than 0.
In this example, this setup allows you to feed features and labels into the model as either the batches of 128 or the single batch of 104.
"""
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

batch_size = 128
epochs = 10
learn_rate = 0.001

train_batches = batches(batch_size, train_features, train_labels)

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch_i in range(epochs):

        # Loop over all batches
        for batch_features, batch_labels in train_batches:
            train_feed_dict = {
                features: batch_features,
                labels: batch_labels,
                learning_rate: learn_rate}
            sess.run(optimizer, feed_dict=train_feed_dict)

        # Print cost and validation accuracy of an epoch
        print_epoch_stats(epoch_i, sess, batch_features, batch_labels)

    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})

print('Test Accuracy: {}'.format(test_accuracy))
```