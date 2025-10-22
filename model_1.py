import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

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
    epochs = 45

    adam_opt = Adam(learning_rate=0.001)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

    # train model
    history_m3 = model.fit(x_train, y_train, 
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