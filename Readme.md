## Model Architecture
- The CNN model architecture is defined in the create_model() function. 
- The model consists of the following layers:

1. Convolutional layer with 32 filters, a kernel size of (3, 3), and ReLU activation.
2. MaxPooling layer with a pool size of (2, 2) to downsample the feature maps.
3. Convolutional layer with 64 filters, a kernel size of (3, 3), and ReLU activation.
4. MaxPooling layer with a pool size of (2, 2).
5. latten layer to convert the 2D feature maps into a 1D feature vector.
6. Dense layer with 128 units and ReLU activation.
7. Dropout layer with a dropout rate of 0.5 to prevent overfitting.
8. Dense output layer with the number of units equal to the number of classes (10) and softmax activation for multi-class classification.


### Training
- The train_model() function is responsible for compiling and training the model.
- The model is compiled with the Adam optimizer, a learning rate of 0.001, and the categorical cross-entropy loss function. It is trained for 10 epochs with a batch size of 128. 
- The training data and validation data are passed to the fit() function for training.

### Evaluation
- After training the model, the evaluate() function is used to calculate the loss and accuracy on the test dataset. The results are printed to the console.

### Prediction on Test Image
- To demonstrate the model's prediction capabilities, a test image from the MNIST dataset is selected (X_test[2190]). The model predicts the label for this image using predict() and displays the predicted label and the image using plt.imshow().

### Prediction on External Image
- An external image is loaded using the cv2.imread() function.
- The image is preprocessed by converting it to grayscale, resizing it to (28, 28) pixels, and normalizing the pixel values. The preprocessed image is then passed to the trained model for prediction.
- The predicted label and the image are displayed using plt.imshow().
