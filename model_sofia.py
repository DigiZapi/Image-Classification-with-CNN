import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def init_model_sofia(input_shape=(32, 32, 3), num_classes=6):
  # Builds a CNN model
  model = Sequential()

  # Convolutional and pooling layers
  # add Convolutional-Layer 
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  # add MaxPooling-Layer
  model.add(MaxPooling2D((2, 2)))

  # add Convolutional-Layer 
  model.add(Conv2D(64, (3, 3), activation='relu'))
  # add MaxPooling-Layer
  model.add(MaxPooling2D((2, 2)))

  # Flatten output
  model.add(Flatten())

  # add Dense-Layer with 128 units
  model.add(Dense(128, activation='relu'))

  # add dropout after dense-layer to  prevent overfitting
  model.add(Dropout(0.5))

  # add classification layer (9 classes)
  model.add(Dense(6, activation='softmax'))

  # model summary
  model.summary()

  return model

def model_sofia(x_train, y_train, x_test, y_test, input_shape=(32, 32, 3)):

    model = init_model_sofia(input_shape)
    batch_size = 64
    epochs = 30
    learning_rate = 0.001

    adam_opt = Adam(learning_rate=learning_rate)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

    # train model
    history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test)
    )

    # predict output for test data
    predictions = model.predict(x_test)
    print(predictions.shape)

    predictions = np.argmax(predictions, axis=1)
    ground_truth = np.argmax(y_test, axis=1)

    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    print("Confusion Matrix:\n", cm)