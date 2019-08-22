# Machine Learning Cheat Sheet — Supervised Learning

## Regression vs. Classification
* A **regression** model predicts continuous values. For example, regression models make predictions for the value of a house in California.

* A **classification** model predicts discrete values. For example, classification models make predictions whether an email message is spam or not spam.

## Key Assumptions
The following three basic assumptions guide generalization:

* We draw examples independently and identically (i.i.d) at random from the distribution. In other words, examples don't influence each other. (An alternate explanation: i.i.d. is a way of referring to the randomness of variables.)
* The distribution is stationary; that is the distribution doesn't change within the data set.
* We draw examples from partitions from the same distribution.

If the key assumptions of supervised ML are not met, then we **lose important theoretical guarantees** on our ability to predict on new data. For example:

1. Consider a model that chooses ads to display. The i.i.d. assumption would be violated if the model bases its choice of ads, in part, on what ads the user has previously seen.
2. Consider a data set that contains retail sales information for a year. User's purchases change seasonally, which would violate stationarity.

## Linear Regression
![alt text](eqn_linear_reg.png) <br />
<small>*W: weights; X: features*</small>

Works best when the data is linear. If the data is not linear, then we may need to transform the data, add features, or use another model. <br />
Sensitive to outliers. Outliers contribute too much to the errors, so will impact the model. We may need to determine the outliers and remove them if necessary.

### Error functions:
![alt text](eqn_abs_error.png) <br />
<small>*Mean Absolute Error*</small>

![alt text](eqn_sqr_error.png) <br />
<small>*Mean Squared Error*</small>

### Gradient Descent:
> Change the weights to move in the direction that descent the error the most.

![alt text](eqn_gradient_descent.png)

In linear regression, split the data into many small batches. Each batch, with roughly the same number of points. Then, use each batch to update your weights. This is still called mini-batch gradient descent.

### Gradient Descent Vs Closed Form Solution:
We can solve W (weights) by setting derivatives of the Error to weights to 0. Then it is just a solving of n*n matrix (n is the number of features). But when n is too large, this requires a lot of computing power. So gradient descent will be a better solution to find the results which is close enough, but requires less computing power.

### Polynomial Regression:
Fit the data to a higher degree polynomials.

## Decision Trees
![alt text](decision_tree.png) <br />
<small>*Decision Tree Example*</small>

### Entropy
Lower entropy means the state is more stable; higher entropy means the state has more randomness in it.

![alt text](eqn_entropy.png) <br />
<small>*Entropy Equation*</small>

![alt text](eqn_information_gain.png) <br />
<small>*Information Gain*</small>

Decision Tree is to split the data into groups so that the information gain is maximized. However it is very easy to lead into **over fitting**.

### Random Forest to Avoid Over Fitting
1. Split features into groups
2. Each group forms a decision tree
3. Let the results from each decision tree to vote

### Decision Tree Hyper-parameters to Avoid Over Fitting
1. Maximum Depth — The largest possible length between the root to a leaf
2. Minimum number of samples to split
3. Minimum number of samples per leaf

## Naive Bayes
![alt text](eqn_bayes.png) <br />
<small>*Bayes Theorem*</small>

Naive Bayes is to use the Bayes Theorem, make some assumptions (e.g. assume the features are independent, `P(A,B) = P(A)*P(B)` ), and calculate the proportions of the probability to simplify the calculation. For example: `P(A|B) ~ P(B|A)P(A)`

## Support Vector Machines
`Error = Error(Classification) + Error(Margin)`

![alt text](svm.png) <br />
<small>*Margin Error (Same as L2 Regularization)*</small>

### C Parameter
`Error =C * Error(Classification) + Error(Margin)`

![alt text](svm_c.png) <br />
<small>*Impact of C*</small>

### Polynomial Kernel
Set a polynomial degree to separate the data.

### RBF Kernel
Project the points to a higher dimension with Gaussian distribution, and cut the mountains by a plane; the circles of the cuts are the boundaries.

![alt text](rbf.png) <br />
<small>*Larger gamma, Narrower the Gaussian*</small>

## Ensemble Methods
Combine multiple models into a better model.

### Bagging
Split data into smaller subsets, and get the models for each subset; then to get the predictions, run under all the models and vote for the final prediction.

### Boosting
1. Get a simple model on the data
2. Punish the classified data more and re-run to get a new model
3. Repeat step 2 for n times
4. Combine all the n models together by model weights

### Model Weights
![alt text](eqn_model_weights.png) <br />
