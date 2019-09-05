# CNN

## MLP Vs CNN
* MLP only use fully connected layers

* MLP has to convert the images into vectors, and has no knowledge of each pixel’s neighbors;

* CNN use sparsely connected layers (locally connected)

* CNN accept matrix as input

## Convolutional Layer
![alt text](convolutional_layer.png) <br />
* Filter determines the pattern to detect; filter weights are learnt during training while it tries to minimize the loss function

* With N filters, we can get N convolved feature maps; then we can stack the N feature maps together as the input for the next convolutional layer

## Convolutional Layer Parameters
* filters — The number of filters.

* kernel_size — Number specifying both the height and width of the (square) convolution window.

* input_shape - Tuple specifying the height, width, and depth of the input. Required when it is first hidden layer.

* Stride — Amount by which the filter slides over the image, default is 1. Smaller stride will help with block artifact.

* Padding — When set to `Valid` (default), we loose some information of edge nodes when the edge nodes cannot fully cover the filter; When set to `same`, we padding the image with 0s to give the filter more space to move.

* activation — Activation function, usually set to `relu`.

## Formulas
* number of parameters of a Convolutional Layer
```
number of parameters = number_of_filters * filter_width * filter_width * prev_layer_depth + number_of_filters
```

* Shape of a Convolutional Layer
```
depth = number_of_filters
For padding = 'same':
    height = ceil(float(prev_layer_height) / float(stride))
    width = ceil(float(prev_layer_width) / float(stride)
For padding = 'valid':
    height = ceil(float(prev_layer_height-filter_width+1) / float(stride))
    width = ceil(float(prev_layer_width-filter_width+1) / float(stride)
```

* Pooling Layer
    - Max Pooling Layer: Take the maximum value in the window
    - Average Pooling Layer: Take average in the window
    - Global Max Pooling Layer: Take the maximum of the entire map in the stack
    - Global Average Pooling Layer: Take the average of the entire map in the stack

## CNN Architecture
![alt text](cnn.png) <br />
<small>*Spatial Information is lost; Gaining Feature Information*</small>

1. Resize the images to the same size before feed into the model; usually resize the images into square with each dimension equal to a power of two.

2. Input layer is followed by a sequence of convolutional layers and pooling layers, to generate feature maps <br />
**convolutional layers**: Maker the array deeper; kernel size is usually between 2 and 5; strides is usually set to 1; padding is usually set to same; activation function is usually set to relu; number of filters usually increases in the sequence. <br />
**pooling layers**: Decrease the spatial dimensions; for example, set pool size to 2, and stride to 2 will result in half the dimension size.

3. Flatten layer to flat the feature maps into a vector

4. Dense layers to elucidate the content of the image

5. Output layer for prediction (for classification, this should be a dense layer with number of nodes the same as number of classes)

## 1X1 Convoluction
A problem with deep convolutional neural networks is that the number of feature maps often increases with the depth of the network. This problem can result in a dramatic increase in the number of parameters and computation required when larger filter sizes are used, such as 5×5 and 7×7.

* The 1×1 filter can be used to create a linear projection of a stack of feature maps.

* The projection created by a 1×1 can act like channel-wise pooling and be used for dimensionality reduction.

* The projection created by a 1×1 can also be used directly or be used to increase the number of feature maps in a model.

## Inception Module
![alt text](inception.png) <br />

The input data may have huge variation in the location of the information, choosing the right kernel size for the convolution operation becomes tough. A larger kernel is preferred for information that is distributed more globally, and a smaller kernel is preferred for information that is distributed more locally.

So we use inception module to overcome it. It performs convolution on an input, with 3 different sizes of filters (1x1, 3x3, 5x5). Additionally, max pooling is also performed. The outputs are concatenated and sent to the next inception module. To make it cheaper, we can limit the number of input channels by adding an extra 1x1 convolution.

## Image Augmentation
> To help make the models more statistically invariant, we can try introducing rotations, translations, etc. into our training images, so that the training set is expanded by augmenting the data. This will improve the model performance.

```python
from keras.preprocessing.image import ImageDataGenerator
# create and configure augmented image generator
datagen_train = ImageDataGenerator(
    # randomly shift images horizontally (10% of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (10% of total height)
    height_shift_range=0.1,
    # randomly flip images horizontally
    horizontal_flip=True)
# fit augmented image generator on data
datagen_train.fit(x_train)
###############################
# define and compile the model#
###############################
from keras.callbacks import ModelCheckpoint
batch_size = 32
epochs = 100
# train the model
checkpointer = ModelCheckpoint(
    filepath='aug_model.weights.best.hdf5',
    verbose=1,
    save_best_only=True
)
model.fit_generator(
    datagen_train.flow( x_train, y_train, batch_size=batch_size ),
    # x_train.shape[0] is the number of unique samples in train set
    # steps_per_epoch ensure that the model sees x_train.shape[0] augmented images in each epoch.
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    verbose=2,
    callbacks=[checkpointer],
    validation_data=(x_valid, y_valid),
    validation_steps=x_valid.shape[0] // batch_size
)
```

## Transfer Learning
> Transfer learning involves taking a pre-trained neural network and adapting the neural network to a new, different data set.

![alt text](transfer_learning.png) <br />
<small>*Four Cases when Using Transfer Learning*</small>

### Case 1: Small Data Set, Similar Data
* slice off the end of the neural network

* add a new fully connected layer that matches the number of classes in the new data set

* randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network

* train the network to update the weights of the new fully connected layer

To avoid overfitting on the small data set, the weights of the original network will be held constant rather than re-training the weights.
Since the data sets are similar, images from each data set will have similar higher level features. Therefore most or all of the pre-trained neural network layers already contain relevant information about the new data set and should be kept.

![alt text](case_1.png) <br />
<small>*Neural Network with Small Data Set, Similar Data*</small>

### Case 2: Small Data Set, Different Data
* slice off most of the pre-trained layers near the beginning of the network

* add to the remaining pre-trained layers a new fully connected layer that matches the number of classes in the new data set

* randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network

* train the network to update the weights of the new fully connected layer

Because the data set is small, overfitting is still a concern. To combat overfitting, the weights of the original neural network will be held constant, like in the first case.
But the original training set and the new data set do not share higher level features. In this case, the new network will only use the layers containing lower level features.

![alt text](case_2.png) <br />
<small>*Neural Network with Small Data Set, Different Data*</small>

### Case 3: Large Data Set, Similar Data
* remove the last fully connected layer and replace with a layer matching the number of classes in the new data set

* randomly initialize the weights in the new fully connected layer

* initialize the rest of the weights using the pre-trained weights

* re-train the entire neural network

Overfitting is not as much of a concern when training on a large data set; therefore, you can re-train all of the weights.
Because the original training set and the new data set share higher level features, the entire neural network is used as well.

![alt text](case_3.png) <br />
<small>*Neural Network with Large Data Set, Similar Data*</small>

### Case 4: Large Data Set, Different Data
* If the new data set is large and different from the original training data:

* remove the last fully connected layer and replace with a layer matching the number of classes in the new data set

* retrain the network from scratch with randomly initialized weights

* alternatively, you could just use the same strategy as the “large and similar” data case

Even though the data set is different from the training data, initializing the weights from the pre-trained network might make training faster. So this case is exactly the same as the case with a large, similar data set.
If using the pre-trained network as a starting point does not produce a successful model, another option is to randomly initialize the convolutional neural network weights and train the network from scratch.

![alt text](case_4.png) <br />
<small>*Neural Network with Large Data Set, Different Data*</small>

## Visualize Filters
The filter weights are useful to visualize because well-trained networks usually display nice and smooth filters without any noisy patterns. Noisy patterns can be an indicator of a network that hasn’t been trained for long enough, or possibly a very low regularization strength that may have led to overfitting.

[This blog](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html?source=post_page---------------------------) shows how we use gradient ascent to generate images that maximize the activation of a filter.