# Embedding Layers
An embedding is a mapping of discrete categorical variables to a vector of continuous numbers. In the context of neural networks, embeddings are low-dimensional, learned continuous vector representations of discrete variables. Neural network embeddings are useful because they can reduce the dimensionality of categorical variables and meaningfully represent categories in the transformed space.

Neural network embeddings have 3 primary purposes:

1. Finding nearest neighbors in the embedding space. These can be used to make recommendations based on user interests or cluster categories.

2. As input to a machine learning model for a supervised task.

3. For visualization of concepts and relations between categories.

Neural network embeddings overcome the two limitations of one-hot encoding:

1. For high-cardinality variables — those with many unique categories — the dimensionality of the transformed vector becomes unmanageable.

2. The mapping is completely uninformed: “similar” categories are not placed closer to each other in embedding space.

## Choose DNN
### Word2Vec
We can use word embedding together with RNN to do predicitons on sentiment: <br />
![alt text](sentiment.png) <br />
<small>*many to one RNN architecture*</small>

Word embedding process:
1. Randomly pick a word as **content**, and randomly pick another word within a certain range as **tartget**.

2. Now we have one-hot vector of the content <code>O<sub>c</sub></code>, an embedding matrix `E`, and we can get the embedding vector <code>e<sub>c</sub></code>.

3. Feed <code>e<sub>c</sub></code> to the softmax neural network, and get the prediction <code>y&#770;</code>. And by minimizing the loss we can get a prediction of <code>e<sub>c</sub></code>. (Note. we can use hierarchical softmax for faster computation.) <br />
![alt text](eqn_wv_cost.png) <br />

#### Negative Sampling
We can use nagetive sampling as well, since the softmax in the above algorithm is too slow.

1. Generate a positive example by randomly picking a word as **content**, and randomly picking another word within a certain range as **tartget**, and label it 1.

2. Generate negative examples by randomly picking target words in the dictionary (~5 negative exmaples for large training set; ~15 negative examples for small training set), and label them 0s, with sampling probability depends partially on the word frequency: <br />
![alt text](eqn_sample_frequency.png) <br />

3. The output probability is: <br />
![alt text](eqn_negative_sampling.png) <br />

#### GloVe Algorithm
1. Let <code>X<sub>ij</sub></code> be the number of times `j` appears in context of `i` within a certain range.

2. Minimize the following algorithm: <br />
![alt text](eqn_glove.png) <br />

#### Addressing Bias in Word Embeddings
1. Identify bias direction. e.g. find the vector between `he` `she`, `male` `female` etc. and average them.

2. Neutralize: For every word that is not definitional, project then to the axis perpendicular to the bias to get rid of the bias.

3. Equalize pairs. e.g. for word like `father` and `mother`, they should only differ in bias, and have equal distance to the axis perpendicular to the bias.

### Autoencoder
A DNN that learns embeddings of input data by predicting the input data itself is called an **autoencoder**. Because an autoencoder’s hidden layers are smaller than the input and output layers, the autoencoder is forced to learn a compressed representation of the input feature data. Once the DNN is trained, you extract the embeddings from the last hidden layer to calculate similarity.

### Predictor
An autoencoder isn't the optimal choice when certain features could be more important than others in determining similarity. For example, in house data, let's assume “price” is more important than “postal code". In such cases, use only the important feature as the training label for the DNN. Since this DNN predicts a specific input feature instead of predicting all input features, it is called a **predictor** DNN. Use the following guidelines to choose a feature as the label:

* Prefer numeric features to categorical features as labels because loss is easier to calculate and interpret for numeric features.

* Do not use categorical features with cardinality  100 as labels. If you do, the DNN will not be forced to reduce your input data to embeddings because a DNN can easily predict low-cardinality categorical labels.

* Remove the feature that you use as the label from the input to the DNN; otherwise, the DNN will perfectly predict the output.

### Loss Function for DNN
To train the DNN, you need to create a loss function by following these steps:

1. Calculate the loss for every output of the DNN. For outputs that are:
    - Numeric, use mean square error (MSE).
    - Univalent categorical, use log loss.
    - Multivalent categorical, use softmax cross entropy loss.

2. Calculate the total loss by summing the loss for every output.
    - When summing the losses, ensure that each feature contributes proportionately to the loss. For example, if you convert color data to RGB values, then you have three outputs. After summing the loss for three outputs you should then multiply by 1/3.

### Similarity Measurement
![alt text](similarity.png) <br />

For word embedding, to find the word for `king`, which has the same relationship like `man` and `woman`, we just need to find the word which max the similarity `argmax sim(w, king-man+woman)`.

1. Items that appear very frequently in the training set tend to have embeddings with large norms. If capturing popularity information is desirable, then you should prefer dot product. However, if you're not careful, the popular items may end up dominating the recommendations. In practice, you can use other variants of similarity measures that put less emphasis on the norm of the item: <code>s(q,x)=||q||<sup>α</sup>||x||<sup>α</sup>cos(q,x)</code> for some <code>α ∈ (0, 1)</code>.

2. Items that appear very rarely may not be updated frequently during training. Consequently, if they are initialized with a large norm, the system may recommend rare items over more relevant items. To avoid this problem, be careful about embedding initialization, and use appropriate regularization.

## Recommendation System
### Candidate Generation
1. Content-based Filtering <br />
Uses similarity between items to recommend items similar to what the user likes. If user A watches two cute cat videos, then the system can recommend cute animal videos to that user.

2. Collaborative Filtering <br />
Uses similarities between queries and items simultaneously to provide recommendations. If user A is similar to user B, and user B likes video 1, then the system can recommend video 1 to user A

3. Softmax Model <br />
    - Matrix factorization is usually the better choice for large corpora. It is easier to scale, cheaper to query, and less prone to folding.

    - DNN models can better capture personalized preferences, but are harder to train and more expensive to query. DNN models are preferable to matrix factorization for scoring because DNN models can use more features to better capture relevance. Also, it is usually acceptable for DNN models to fold, since you mostly care about ranking a pre-filtered set of candidates assumed to be relevant.

## Sample Code
![alt text](embedding_layers.png) <br />

<details>
    <summary>Sample Code</summary>

```python
hidden_units = (32,4)
movie_embedding_size = 8
user_embedding_size = 8

# Each instance will consist of two inputs: a single user id, and a single movie id
user_id_input = keras.Input(shape=(1,), name='user_id')
movie_id_input = keras.Input(shape=(1,), name='movie_id')

user_embedded = keras.layers.Embedding(
    df.userId.max()+1,
    user_embedding_size,
    input_length=1,
    name='user_embedding'
)(user_id_input)

movie_embedded = keras.layers.Embedding(
    df.movieId.max()+1,
    movie_embedding_size,
    input_length=1,
    name='movie_embedding'
)(movie_id_input)

# Concatenate the embeddings (and remove the useless extra dimension)
concatenated = keras.layers.Concatenate()([user_embedded, movie_embedded])
out = keras.layers.Flatten()(concatenated)

# Add one or more hidden layers
for n_hidden in hidden_units:
    out = keras.layers.Dense(n_hidden, activation='relu')(out)

# A single output: our predicted rating
out = keras.layers.Dense(1, activation='linear', name='prediction')(out)

# Add biases
bias_embedded = keras.layers.Embedding(
    df.movieId.max()+1,
    1,
    input_length=1,
    name='bias',
)(movie_id_input)
movie_bias = keras.layers.Flatten()(bias_embedded)
out = keras.layers.Add()([out, movie_bias])

model = keras.Model(
    inputs = [user_id_input, movie_id_input],
    outputs = out,
)

model.compile(
    # Technical note: when using embedding layers, I highly recommend using one of the optimizers
    # found  in tf.train: https://www.tensorflow.org/api_guides/python/train#Optimizers
    # Passing in a string like 'adam' or 'SGD' will load one of keras's optimizers (found under 
    # tf.keras.optimizers). They seem to be much slower on problems like this, because they
    # don't efficiently handle sparse gradient updates.
    tf.train.AdamOptimizer(0.005),
    loss='MSE',
    metrics=['MAE'],
)

history = model.fit(
    [df.userId, df.movieId],
    df.y,
    batch_size=5000,
    epochs=20,
    verbose=0,
    validation_split=.05,
)
```
</details>

## Matrix Factorization
![alt text](matrix_factorization.png) <br />

<details>
    <summary>Sample Code</summary>

```python
movie_embedding_size = user_embedding_size = 8

# Each instance consists of two inputs: a single user id, and a single movie id
user_id_input = keras.Input(shape=(1,), name='user_id')
movie_id_input = keras.Input(shape=(1,), name='movie_id')

user_embedded = keras.layers.Embedding(
    df.userId.max()+1,
    user_embedding_size,
    input_length=1,
    name='user_embedding'
)(user_id_input)
movie_embedded = keras.layers.Embedding(
    df.movieId.max()+1,
    movie_embedding_size,
    input_length=1,
    name='movie_embedding'
)(movie_id_input)

dotted = keras.layers.Dot(2)([user_embedded, movie_embedded])
out = keras.layers.Flatten()(dotted)

model = keras.Model(
    inputs = [user_id_input, movie_id_input],
    outputs = out,
)
model.compile(
    tf.train.AdamOptimizer(0.001),
    loss='MSE',
    metrics=['MAE'],
)
model.summary(line_length=88)
```

</details>

## Tips
### Embedding Size
Increasing the size of our embeddings will:
* Increases model's capacity, and the model will be able to recognize more complex patterns, increasing our accuracy.

* The downside is that the model might be overfitting.

### Adding Biases
* Adding biases gives our model more numbers to tune, and in this sense it's strictly increasing its "capacity".

* Because our biases get added at the very end, our model has a lot less flexibility in how to use them. And this can be a good thing. At a high level, we're imposing a prior belief - that some movies are intrinsically better or worse than others. This is a form of regularization!

### Regularization
By adding L2 regularization, we can fix the obscure recommendation problem.

In the absence of regularization, even if a movie has only a single rating, the model will try to move its embedding around to match that one rating. However, if the model has a budget for movie weights, it's not very efficient to spend it on improving the accuracy of one rating out of 20,000,000. Popular movies will be worth assigning large weights. Obscure movies should have weights close to 0.

If a movie's embedding vector is all zeros, our model's output will always be zero after dot product. For output value of 0, it corresponds to a predicted rating equal to the overall average in the training set. This seems like a reasonable behaviour to tend toward for movies we have little information about.

<details>
    <summary>Sample Code</summary>

```python
movie_embedding_size = user_embedding_size = 8
user_id_input = keras.Input(shape=(1,), name='user_id')
movie_id_input = keras.Input(shape=(1,), name='movie_id')

movie_r12n = keras.regularizers.l2(1e-6)
user_r12n = keras.regularizers.l2(1e-7)
user_embedded = keras.layers.Embedding(
    df.userId.max()+1,
    user_embedding_size,
    embeddings_initializer='glorot_uniform',
    embeddings_regularizer=user_r12n,
    input_length=1,
    name='user_embedding'
)(user_id_input)
movie_embedded = keras.layers.Embedding(
    df.movieId.max()+1,
    movie_embedding_size,
    embeddings_initializer='glorot_uniform',
    embeddings_regularizer=movie_r12n,
    input_length=1,
    name='movie_embedding'
)(movie_id_input)

dotted = keras.layers.Dot(2)([user_embedded, movie_embedded])
out = keras.layers.Flatten()(dotted)

l2_model = keras.Model(
    inputs = [user_id_input, movie_id_input],
    outputs = out,
)
```
</details>