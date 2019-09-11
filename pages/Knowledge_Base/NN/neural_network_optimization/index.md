# Neural Network Optimization Techniques

## Regularization
Regularization takes the coefficients into part of the error, as large coefficients may result in over fitting.

* **L1 Regularization** — Add the absolute of the coefficients into the error.

* **L2 Regularization** — Add the squares of the coefficients into the error.

![alt text](regularization_1.png) <br />
<small>***Computation Efficiency**: Absolute values are difficult to calculate the derivation. <br />*
***Sparse Outputs**: Lots of features, but only a few is relevant to the results. <br />*
***Feature Selection**: L1 Regularization can find the features that are important and relevant. For example, for weights `(1, 0)` and `(0.5, 0.5)`, they are the same amount of errors in in L1, but in L2, `(0.5, 0.5)` is favored. <br />*</small>

### Neural Network Regularization
![alt text](eqn_nn_regularization.png) <br />
<small>*We usually only do regularization on weights (`w`), and ommit `b`, as there are way more `w` parameters so that we can simply ignore `b` parameters. <br />
We usually use L2 regularizations; as L1 regularization will make the w parameters very sparse, which has no benefit for neural network.*</small>

Why we need regularization in Neural Networks:

1. Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in high dimensions, result in over fitting. <br />
![alt text](regularization_2.png)
    * Model on the left (small coefficients)has larger errors, model on the right (large coefficients) has smaller errors.

    * Model on the right is better than the model on the right; Model on the right is too certain, and has little room to apply gradient descent; easily results in over fitting.

    * Bad models are usually too certain of themselves; good models are full of doubts.

    * So we add regularization to punish on the large coefficients.

2. By penalizing on the weight parameters, we can have smaller `w` values, results in lighter neurons, and simplify the network to avoid overfitting.

3. With regularization, we will have small values for weights, which will results in smaller input data into the activation functions. By having small values (which are close to 0) into the sigmoid/tanh function, it is close to linear (as per the red line in the image above). So it will result in simpler network.

## Dropout Regularization
```python
keep_prob = 0.8

dl = np.random.rand(al.shape[0], al.shape[1]) < keep_prob
al = np.multiply(al, dl)

# Inverted dropout - make sure the expected output from layer remains the same, which makes test easier as there is no scaling problem
al /= keep_prob
```
1. This is the probability that each node gets dropped at each epoch during training. In a fully connected layer, neurons develop co-dependency with each other during training, and will result in overfitting. Dropout can prevent over fitting.

2. Dropout also forces the model to learn a redundant representation for everything to make sure that at least some of the information remains. By learning redundantly, it also means the neuron in the later layer cannot trust the neurons int the previous layer (as it can get dropped), so it will spread the weights across the neurons instead of have a large weight on one neuron. This effect is also similar to L2 regularization.

4. During testing, set dropout probability to 0 to keep all units and maximize the power of the model. Also we do not want our output to be random.

5. We can have different dropout probability on different layers, depends on the number of neurons on the layer.

6. Usually do not use dropout unless the algorithm is overfitting. It is mainly used in Computer Vision as you usually just don't have enough data (for all these pixels with all the possible values), so you're almost always overfitting.

7. Disadvantage of dropout is that the cost function will not be well defined.

## Normalizing Inputs
Steps to normalize the input data:

1. Zero out the mean <br />
![alt text](eqn_mean.png)

2. Normalize the variances <br />
![alt text](eqn_var.png)

3. Use the same `μ` and `σ` to normalize the test set

![alt text](normalize.png)

The reason to normalize the input data is that:
1. If the features are on very different scales, and we do not normalized input features, then the range of values for the parameters w1 and w2 will end up taking on very different values. And the cost function will look like a very squished out bowl. The gradient descent will have to use a very small learning rate and need a lot of steps to find the minimum.

2. If you normalize the features, then your cost function will on average look more symmetric, and looks like spherical contours. The gradient descent can take much larger steps to get to the minimum.