# Data Processing Techniques
We should preprocess the data inside the model so that when we export the model it includes the preprocessing. This way we can pass the raw data directly to the model.

## Numerical Data
### Skewed Data / Imbalanced Data
![alt text](skewed_data.png)

Outliers affect the distribution. If a value is significantly below the expected range, it will drag the distribution to the left, making the graph left-skewed or negative. Alternatively, if a value is significantly above the expected range, it will drag the distribution to the right, making the graph right-skewed or positive.

There are different ways to handle skewed data:
* Log Function (`log(x+1)`), then Normalization
* Hyperbolic Tangent
* Percentile Linearization

### Data Normalization
For tree based models, we may not need data normalization; <br />
For linear models, we need to normalize the data, so that the feature values fall in range  like _(0, 1)_.

#### Advantages
1. Helps gradient descent converge more quickly when the values fall into similar range. <br/>
![alt text](feature_scaleing.png)

2. Helps avoid the "NaN trap," in which one number in the model becomes a NaN when a value exceeds the floating-point precision limit during training.

3. In case the model prediction results will be biased on the features with large values.

We don't have to give every feature exactly the same scale. Nothing terrible will happen if Feature A is scaled from -1 to +1 while Feature B is scaled from -3 to +3.

#### Disadvantage
1. Data normalization is sensitive to outliers.

#### Methods
1. scaling to a range <br/>
Scaling to a range is a good choice when both of the following conditions are met:
    - You know the approximate upper and lower bounds on your data with few or no outliers.
    - Your data is approximately uniformly distributed across that range.
    - `x = (x - mean)/(max- min)` or `x = (x - mean)/std`

2. clipping <br/>
If your data set contains extreme outliers, you might try feature clipping, which caps all feature values above (or below) a certain value to fixed value. For example, you could clip all temperature values above 40 to be exactly 40.

3. log scaling <br/>
Log scaling is helpful when a handful of your values have many points, while most other values have few points (e.g. power law distribution). Log scaling changes the distribution, helping to improve linear model performance.

4. z-score <br/>
You use z-score to ensure your feature distributions have `mean = 0` and `std = 1`. 

### Binning / Bucketing
For example, we can create bins to convert geo-location data (latitude, longitude) into bins to make them categorical features. The float numbers of coordinates has no linear relation ship with the target. By making them into bins makes more sense.

1. Buckets with equally spaced boundaries <br/>
The boundaries are fixed and encompass the same range. Some buckets could contain many points, while others could have few or none.

2. Buckets with quantile boundaries <br/>
Each bucket has the same number of points. The boundaries are not fixed and could encompass a narrow or wide span of values. Bucketing by quantile completely removes the need to worry about outliers.

## Categorical Data
### One-hot Encoding
Convert categorical data into binary variables. For example, convert feature gender into two columns, male and female, with value 0 or 1.

One-hot encoding converts a categorical feature into multiple columns of numerial features with 0 and 1. Why not convert categorical feature into a single column with numerical values like 1, 2, 3, 4...? The constraints are:

1. We'll be learning a single weight that applies to all categories for that feature.
2. We aren't accounting for cases where the data point belongs to multiple categories.

## Data Preprocessing
1. Drop rarely used discrete feature values <br />
For features like `unique_id`, we can simply drop them.

2. Prefer clear and obvious meanings <br />
For features like `house_age`, we prefer it to be in years instead of epoch time.

3. Clean up noises / Filling up missing values <br />
Some feature values may should lie within a range (e.g. percentage should be within 0 and 1). If we see noises like -1 which is definitly out of range, we can replace them with: mean values (for numerical data) or a new value (for categorical data)

4. Feature Cross <br />
    - Feature cross is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together. (The term cross comes from cross product.) For example, corssing one hot encoded features of `country` and `language`, get us more meaningful features. Or we can also do <code>x<sup>2</sup></code>.
    - However, this also introduces complexity into the model, which may result in a bad performance on the test data (over fitting). This is where regularization comes into play.

5. Account for upstream instability <br />
The definition of a feature shouldn't change over time. 

## Data Leakage
Data leakage happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction. This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production.

There are two main types of leakage: **target leakage** and **train-test contamination**.

### Target leakage
Target leakage occurs when your predictors include data that will not be available at the time you make predictions. It is important to think about target leakage in terms of the timing order that data becomes available, not merely whether a feature helps make good predictions.

Imagine you want to predict who will get sick with pneumonia. The top few rows of your raw data look like this:

![alt text](target_leakage_table.png)

People take antibiotic medicines after getting pneumonia in order to recover. The raw data shows a strong relationship between those columns, but took_antibiotic_medicine is frequently changed after the value for got_pneumonia is determined. This is target leakage.

The model would see that anyone who has a value of False for took_antibiotic_medicine didn't have pneumonia. Since validation data comes from the same source as training data, the pattern will repeat itself in validation, and the model will have great validation (or cross-validation) scores.

But the model will be very inaccurate when subsequently deployed in the real world, because even patients who will get pneumonia won't have received antibiotics yet when we need to make predictions about their future health.

To prevent this type of data leakage, any variable updated (or created) after the target value is realized should be excluded.

![alt text](target_leakage_time.png)

### Train-Test Contamination
A different type of leak occurs when you aren't careful to distinguish training data from validation data.

Recall that validation is meant to be a measure of how the model does on data that it hasn't considered before. You can corrupt this process in subtle ways if the validation data affects the preprocessing behavior. This is sometimes called train-test contamination.

For example, if you normalize or standardize your entire dataset, then estimate the performance of your model using cross validation, you have committed the sin of data leakage.

The data rescaling process that you performed had knowledge of the full distribution of data in the training dataset when calculating the scaling factors (like min and max or mean and standard deviation). This knowledge was stamped into the rescaled values and exploited by all algorithms in your cross validation test harness.

A non-leaky evaluation of machine learning algorithms in this situation would calculate the parameters for rescaling data within each fold of the cross validation and use those parameters to prepare the data on the held out test fold on each cycle.

### 5 Tips to Combat Data Leakage
1. **Temporal Cutoff**. Remove all data just prior to the event of interest, focusing on the time you learned about a fact or observation rather than the time the observation occurred.

2. **Add Noise**. Add random noise to input data to try and smooth out the effects of possibly leaking variables.

3. **Remove Leaky Variables**. Evaluate simple rule based models line OneR using variables like account numbers and IDs and the like to see if these variables are leaky, and if so, remove them. If you suspect a variable is leaky, consider removing it.

4. **Use Pipelines**. Heavily use pipeline architectures that allow a sequence of data preparation steps to be performed within cross validation folds, such as the caret package in R and Pipelines in scikit-learn.

5. **Use a Holdout Dataset**. Hold back an unseen validation dataset as a final sanity check of your model before you use it.

## Imbalanced Data Set
Imbalanced data set means the data is not well distributed among different classes. For example, only less than 0.1% of all the bacnk transactions are fraud.

1. **Use proper evaluation metrics**

2. **Under-sampling** <br />
This method is used when quantity of data is sufficient. Keep all samples in the rare class and randomly select an equal number of samples in the abundant class to form a balanced new dataset. <br />
Or use _Tomek Links_, which are pairs of points (A,B) such that A and B are each other’s nearest neighbor, and they have opposing labels. And remove from the abundant class along the _Tomek Links_.

3. **Over-sampling** <br />
This method is used when the quantity of data is insufficient. It tries to balance dataset by increasing the size of rare samples. <br />
    - Randomly pick a point from the minority class.
    - Compute the k-nearest neighbors (for some pre-specified k) for this point.
    - Add k new points somewhere between the chosen point and each of its neighbors.

4. **Use K-fold Cross-Validation in the right way** <br />
If cross-validation is applied after over-sampling, basically what we are doing is overfitting our model to a specific artificial bootstrapping result. So cross-validation should always be done before over-sampling the data. Only by resampling the data repeatedly, randomness can be introduced into the dataset to make sure that there won’t be an overfitting problem.

5. **Ensemble different resampled datasets** <br />
Building n models that use all the samples of the rare class and n-differing samples of the abundant class, then ensemble the models.

6. **Cluster the abundant class** <br />
Cluster the abundant class in r groups, and for each group, only the medoid (centre of cluster) is kept. The model is then trained with the rare class and the medoids only.

7. **Use penalized models** <br />
Penalized models impose an additional cost on the model for making classification mistakes on the minority class during training. These penalties can bias the model to pay more attention to the minority class. For example, give a high weight to the errors on the minority class on the loss function. There are penalized versions of algorithms such as penalized-SVM and penalized-LDA.

## Outliers: To Drop or Not to Drop
* If it is obvious that the outlier is due to incorrectly entered or measured data, you should drop the outlier.

* If the outlier does not change the results but does affect assumptions, you may drop the outlier. But keep a note of it. <br />
![alt text](outlier_1.png)

* If the outlier affects both results and assumptions, we cannot simply drop the outlier. We need to run the analysis both with and without it, and keep a not of how the results changed by dropping the outlier. <br />
![alt text](outlier_2.png)

* If the outlier creates a significant association, you should drop the outlier and should not report any significance from your analysis. <br />
![alt text](outlier_3.png) <br />
<small>*Drop the Outlier*</small>

For cases where we cannot drop a outlier, we can do the following:
1. Try a transformation. Square root and log transformations both pull in high numbers. This can make assumptions work better if the outlier is a dependent variable and can reduce the impact of a single point if the outlier is an independent variable.

2. Try clipping the data. For example `value = min(value, 10)` will clip the outlier and map it to `10`.

3. Minkowski Error. Instead of measuring our model using the squared error, we raise error to a power less than two: say 1.5. In this way, the contribution that an outlier gives is lessened and we can keep the data.

4. Try a different model.

## Feature Selection / Dimension Reduction
A good feature set contains features that are highly correlated with the class, yet uncorrelated with each other. It will not only speed up algorithm execution, but may also improve model scores.

1. **Missing Values Ratio** <br />
Data columns with too many missing values can be removed.

2. **Low Variance Filter** <br />
Data columns with little changes (low variance)in the data can be removed. Variance is range dependent; therefore normalization is required before applying this technique.

3. **High Correlation Filter** <br />
Data columns with high correlations can be reduced to only one. Correlation is scale sensitive; therefore column normalization is required for a meaningful correlation comparison.

4. **Random Forests / Ensemble Trees** <br />
Generate a large set of shallow trees (e.g. 2 levels), with each tree being trained on a small fraction of the total number of attributes. If an attribute is often selected as best split, it is most likely an informative feature to retain.

5. **PCA** <br />
PCA is a statistical procedure that transforms the original n coordinates of a data set into a new set of n coordinates called principal components. As a result of the transformation, the first principal component has the largest possible variance; each succeeding component has the highest possible variance under the constraint that it is orthogonal to (i.e., uncorrelated with) the preceding components. Keeping only the first m < n components reduces the data dimensionality while retaining most of the data information, i.e. the variation in the data. Notice that the PCA transformation is sensitive to the relative scaling of the original variables. Data column ranges need to be normalized before applying PCA. Also notice that the new coordinates (PCs) are not real system-produced variables anymore. Applying PCA to your data set loses its interpretability. If interpretability of the results is important for your analysis, PCA is not the transformation for your project.

6. **Backward Feature Elimination** <br />
In this technique, at a given iteration, the selected classification algorithm is trained on n input features. Then we remove one input feature at a time and train the same model on n-1 input features n times. The input feature whose removal has produced the smallest increase in the error rate is removed, leaving us with n-1 input features. The classification is then repeated using n-2 features, and so on.

7. **Forward Feature Construction** <br />
This is the inverse process to the Backward Feature Elimination. We start with 1 feature only, progressively adding 1 feature at a time, i.e. the feature that produces the highest increase in performance.