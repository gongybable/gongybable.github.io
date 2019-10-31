# Model Evaluation and Validation

## Model Evaluation
### Confusion Matrix Summary
![alt text](confusion_matrix.png) <br />
<small>*Blue Points — labelled positive; Red Points — labelled negative*</small>

True Positive: 6 blue points <br />
True Negative: 5 red points <br />
False Positive (Type 1 Error): 2 red points <br />
False Negative (Type 2 Error): 1 blue point <br />


### Accuracy
![alt text](eqn_accuracy.png)

Accuracy is not always perfect for model evaluation, especially for **imbalanced data sets**.

For example, if we have a data set, where 99% of the data is positive, 1% of the data is negative (skewed data). We can simply have a model which always predict positive, then the model accuracy is 99%, while we are not catching any of the negative data.

### Precision
![alt text](eqn_precision.png)

Of all the items classified as Y, what % of them are actually Y.
* Precision focuses on **False Positive** errors.
* Higher threshold increases precision score.

### Recall
![alt text](eqn_recall.png)

Of all the items are Y, what % of them are actually classified as Y.
* Recall focuses on **False Negative** errors.
* Higher threshold decreases recall score.

### F1 Score
![alt text](eqn_f1_score.png)

F1 Score is a harmonic mean between Precision and Recall.

### F beta Score
![alt text](eqn_f_beta__score.png)

If &#946; is close to 0, then F beta is close to precision, the result is more sensitive to **False Positive** errors; <br />
If &#946; is large, then F beta is close to recall, the result is more sensitive to **False Negative** errors;

_**Examples:**_ <br />
&#946; should be large if the model is to detect the malfunctioning parts in cars; <br />
&#946; should be small if the model is to detect the potential clients to send promotion materials. <br />

### ROC Curve
Look at every possible classification threshold and look at the true positive and false positive rates at that threshold.

To find ROC Curve:
1. True Positive Rate = True Positives / All Positives <br />
2. False Positive Rate = False Positives / All Negatives <br />
3. Find all the (_True Positive Rate_, _False Positive Rate_), and plot them on a plane to get a curve <br />
4. Calculate the area under the curve - AUC <br />
5. If area is close to 1, then we can get a good split from the data   set; <br />
   If area is close to 0.5, then the data set is random, we cannot get a good split; <br />
   If area is close to 0, we can flip the data and get a good split. <br />

**AUC** stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve.

### R2 Score
![alt text](eqn_r2.png)

R2 Score measures the percentage of data variation not described by the model:
* R2 score close to 0, then it is a bad model; <br />
* R2 score close to 1, then it is a good model. <br />

## Model Validation
**Under Fitting:** Model is too simple; high training error. <br />
**Over Fitting:** Model is too complicated, and memorise the training data; has high test error.

**Training Set:** Used to train the model. <br />
**Validation Set:** Used for model selection, tune the hyper-parameters. <br />
**Test Set:** Test the model on un-seen data. <br />
* If we do not have validation set, and simply use the test results to decide on the parameters of the model, then after many iterations of this procedure, we are overfitting to the test set of data.

* Usually the data split is `60-20-20`; in the big data era, we may now only need 1% or even less for validation and test dataset, when the entire data set is very large.

### K-Fold Cross Validation
1. Split the data into training set and test test
2. Split the training set into K subsets
3. Use K-1 subsets to train the model, and the 1 set to validate the model
4. Rotate step 3 K times so that each subset is used as validation set for one time
5. All of the validation error rates are averaged together.

**Pros:** <br />
* Allows us not to set aside a validation data set, which is beneficial when we have a small data set. <br />
* Prevents over fitting from over-tuning the model during grid search. If we use single validation set then there is the risk that we just select the best parameters for that specific validation set. But using k-fold we perform grid search on various validation sets so we select best hyper-parameter for generalisation.

### Learning Curve
![alt text](learning_curve.png) <br />
<small>*For a good model, the validation error and training error (may use accuracy for classification problems) converges with larger data set, and error is low.*</small>

* The training error and validation error tends to converge with more data; but more data is not always helpful if both errors are already converged to the optimal scores. <br />
* From learning curve, if the model is under fitting, then we can try increase the model complexity by adding more features, or decrease the regularisation parameter. <br />
* If the model is over fitting, we can try simplify the model by setting a smaller set of features, or increase the regularisation parameter.

### Basic Recipt for ML
![alt text](bias_variance.png) <br />

1. If there is high bias:
    - Try bigger network (more features)
    - Try train longer
    - Try different network architecture

2. If there is high variance:
    - Try more data
    - Try smaller network (less features)
    - Try regularization
    - Try different network architecture

### Grid Search
Training the model on different combinations of hyper-parameters, and select the combination with highest score on the validation set.

### Debugging Gradient Descent
We can plot the loss function over iterations to troubleshoot how our gradient descent is working. If the loss is increasing over iterations, then we should try a smaller learning rate.

We can declear that the loss function is converged if it decreases by less than E in one iteration, where E is some small value such as 10−3.

![alt text](learning_rate.png) <br />

- A: Good learning rate
- B: Learning rate too small, slow convergence
- C: Learning rate too large, loss is diverging