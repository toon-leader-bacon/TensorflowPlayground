import tensorflow as tf
import keras

print("Hello world! Does my import statement work properly?")
print("TensorFlow version:", tf.__version__)


def blab():
    # x_train: unit8 NumPy array of grayscale image data with shapes (60_000, 28, 28). 
    #          Pix values range from 0 to 255. Effectively 60k pictures
    # y_train: unit 8 NumPy array of digit labels (range 0-9) with shape (60_000)
    # x_test: same as x_train
    # y_test: same as y_train
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Divide every value in the images by 255. Effectively converting the range of [0,255]
    # to the range [0,1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Sequential neural net, as apposed to a cyclical
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Input dimension 28x28 pix image
        tf.keras.layers.Dense(128, activation='relu'),  # Rectilinear unit activation
        tf.keras.layers.Dropout(0.2),  # 20% drop out layer
        tf.keras.layers.Dense(10)  # output layer for signaling label
    ])
    
    single_training_image = x_train[:1]
    # Pass in a single image to model. 
    # Predictions is a collection of model outputs. In this case, because
    # we only put in a single image, the size of predictions is 1.
    # The model outputs from 10 neurons, so the length of that single prediction
    # is 10. Each value ranging from [-1, 1] is the estimate value.
    # Wildly inaccurate without training.
    predictions = model(single_training_image).numpy()
    
    # The softmax function will normalize the data so they are all positive and sum to exactly 1
    tf.nn.softmax(predictions).numpy()
    
    # Function used to calculate the value that should be minimized during training
    # From the docs "Use this crossentropy loss function when there are two or more label classes"
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Use the loss function to calculate the loss!
    # Provide the first collection of true labels, and the first predicted labels 
    loss_fn(y_train[:1], predictions).numpy()
    
    # Configure the model for training
    model.compile(optimizer='adam', # Stochastic gradient descent. A classic!
                  loss=loss_fn, 
                  metrics=['accuracy']) # Typical use, according to the docs

    # Train the model for 5 epochs (All data is processed 5 times)
    model.fit(x_train, y_train, epochs=5)
    
    # Run the model and calculate loss values and metrics
    model.evaluate(x_test, y_test, verbose=2)

blab()
