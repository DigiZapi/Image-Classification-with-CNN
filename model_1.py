import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from print_model_score import print_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def init_model_1(input_shape=(32, 32, 3)):
  # init model
  vgg_model = Sequential()

  # add Convolutional-Layer 
  vgg_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
  vgg_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

  # add MaxPooling-Layer
  vgg_model.add(MaxPooling2D((2, 2), padding='same'))

  # add 2 more Convolutional-Layer 
  vgg_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
  vgg_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

  # improve training speed and stability
  vgg_model.add(BatchNormalization())


  # add MaxPooling-Layer
  vgg_model.add(MaxPooling2D((2, 2), padding='same'))

  # Flatten output
  vgg_model.add(Flatten())

  # add Dense-Layer with 128 units
  vgg_model.add(Dense(128, activation='relu'))

  # add dropout after dense-layer to  prevent overfitting
  vgg_model.add(Dropout(0.5))

  # add classification layer (5 classes)
  vgg_model.add(Dense(6, activation='softmax'))

  # model summary
  vgg_model.summary()

  return vgg_model

def model_1(x_train, y_train, x_test, y_test, input_shape=(32, 32, 3)):
    model = init_model_1(input_shape)
    batch_size = 512
    num_classes = 6
    epochs = 10

    adam_opt = Adam(learning_rate=0.001)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

    # train model
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))

    # predict output for test data
    predictions = model.predict(x_test)

    print(predictions.shape)
    predictions = np.argmax(predictions, axis=1)

    # print confusion matrix
    gt = np.argmax(y_test, axis=1)
    confusion_matrix(gt, predictions)



    # Predictions and confusion matrix
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)
    gt = np.argmax(y_test, axis=1)

    #y_test_conv = np.argmax(y_test_conv, axis=1)
    recall = recall_score(gt, predictions, average='weighted')  # Use 'weighted' for multiclass
    f1 = f1_score(gt, predictions, average='weighted')  # Use 'weighted' for multiclass

    
    cm = confusion_matrix(gt, predictions)
    print("Confusion Matrix:\n", cm)

    # Plotting
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='red', label='val')
    plt.legend()

    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='green', label='train')
    plt.plot(history.history['val_accuracy'], color='red', label='val')
    plt.legend()

    plt.tight_layout()  # Improve layout
    plt.show()

    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    # Print the test accuracy
    print(f'Test Accuracy: {test_accuracy:.3f}')
    print(f'Test Loss: {test_loss:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')