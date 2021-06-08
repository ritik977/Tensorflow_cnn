#Tensorflow for Image recognigation 
It usually involve these steps to build a successful model

    Examine and understand data
    Build an input pipeline
    Build the model
    Train the model
    Test the model
    Improve the model and repeat the process

Here are the steps used for this project:-

1) Define some parameters for the loader:
    batch_size = 32
    img_height = 180
    img_width = 180

2) Spliting Dataset
    We will use 80% of the images for training, and 20% for validation.

3) Visualize the data
    we can visualize the data using matplotlib.pyplot library, I have commented out that code and we can uncomment for the same

4) Standardize the data
    The RGB channel values are in the [0, 255] range. This is not ideal for a neural network; in general you should seek to make your input values small. Here, we will standardize values to be in the [0, 1] by using a Rescaling layer.

5) Configure the dataset for performance
    Let's make sure to use buffered prefetching so we can yield data from disk without having I/O become blocking. These are two important methods you should use when loading data.

    .cache() keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.

6) Create the model
    The model consists of three convolution blocks with a max pool layer in each of them. There's a fully connected layer with 128 units on top of it that is activated by a relu activation function

7)Compile the model
    For this project I have chose  optimizers.Adam optimizer and losses.SparseCategoricalCrossentropy loss function. To view training and validation accuracy for each training epoch, pass the metrics argument.

8)Model summary
    View all the layers of the network using the model's summary method.

9)Train the model
    Using 10 epochs, we will visualise results and see for overfitting and then will take decision accordingly

10)Visualize training results
    Create plots of loss and accuracy on the training and validation sets. I have commented out that code but we can uncomment for visualizing training results

11) Overfitting
    After plotting the curve I saw vadidation accuracy and validation accuracy is off by some margin and this is a sign of overfitting

12) Data augmentation
    Overfitting generally occurs when there are a small number of training examples. Data augmentation takes the approach of generating additional training data from your existing examples by augmenting them using random transformations that yield believable-looking images. This helps expose the model to more aspects of the data and generalize better.

13) DropOut
    Another technique to reduce overfitting is to introduce Dropout to the network, a form of regularization.

    When you apply Dropout to a layer it randomly drops out (by setting the activation to zero) a number of output units from the layer during the training process. Dropout takes a fractional number as its input value, in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20% or 40% of the output units randomly from the applied layer.

14) Visualize new training training results
    Now after plotting the curve I saw vadidation accuracy and validation accuracy are following a parallel like curve and accuracy is arround 97% which is a good model overall

15) Saving model for prediction
    Saving model in Json format and weights in h5 format.

16) Using model for Prediction
    I have used flask framework for building API and inside test/test/ folder I have my test pics, we can copy any pic to that folder and pass the name in api to get back prediction.
    api would be "/api/v1/<string:img_name>"

#-------------------------------------------------------------