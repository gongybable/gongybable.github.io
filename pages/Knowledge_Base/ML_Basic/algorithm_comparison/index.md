# Algorithm Comparison and Selection

<a href="https://download.microsoft.com/download/A/6/1/A613E11E-8F9C-424A-B99D-65344785C288/microsoft-machine-learning-algorithm-cheat-sheet-v7.pdf" download>Microsoft Machine Learning Algorithm Cheat Sheet</a>


## Supervised / Unsupervised Learning and Reinforcement Learning

### Supervised learning
Supervised learning trains on labelled data.

1. **Classification** <br />
For predicting a category. When there are only two labels, this is called binomial classification. When there are more than two categories, the problems are called multi-class classification. Usually we use crossentropy as the loss function.

2. **Regression** <br />
For predicting values. Usually we use MSE as the loss function.

3. **Forecasting** <br />
For making predictions about the future based on the past and present data. It is most commonly used to analyse trends.

4. **Anomaly detection** <br />
To identify data points that are unusual. For example, in credit card fraud detection, the possible variations are so numerous and the training examples so few, that it’s not feasible to learn what fraudulent activity looks like. The approach that anomaly detection takes is to learn what normal activity looks like (using a history of non-fraudulent transactions) and identify anything that is significantly different.

### Unsupervised learning
Unsupervised learning trains on unlabelled data.

1. **Clustering:** <br />
Grouping a set of data examples so that examples in one group (or one cluster) are more similar (according to some criteria) than those in other groups. This is often used to segment the whole dataset into several groups. Analysis can be performed in each group to help users to find intrinsic patterns.

2. **Dimension reduction:** <br />
Reducing the number of variables under consideration. In many applications, the raw data have very high dimensional features and some features are redundant or irrelevant to the task. Reducing the dimensionality helps to find the true, latent relationship. 

### Reinforcement learning
Reinforcement learning analyses and optimises the behaviour of an agent based on the feedback from the environment. Machines try different scenarios to discover which actions yield the greatest reward, rather than being told which actions to take. Trial-and-error and delayed reward distinguishes reinforcement learning from other techniques.

## Algorithm Selection
When choosing an algorithm, we can start with algorithms that are easy to implement and can obtain results quickly. After we obtain some results and become familiar with the data, we may spend more time using more sophisticated algorithms to strengthen the results. We need to take following aspects into account:

* Accuracy and Training Time <br />
They are closely tied to each other. Sometimes we do not need to get the most accurate result, an approximation is adequate. If that’s the case, we can cut the training time dramatically by sticking with more approximate methods. Another advantage of more approximate methods is that they naturally tend to avoid overfitting.

* Data Linearity <br />

* Number of Parameters <br />
Algorithms with large numbers of parameters require the most trial and error to find a good combination. The upside is that having many parameters typically indicates that an algorithm has greater flexibility. It can often achieve very good accuracy, after finding the right combination of parameter settings.

* Number of features <br />
For certain types of data, the number of features can be very large compared to the number of data points (for example, textual data).The large number of features will result in long training time. Support Vector Machines are particularly well suited to this case.