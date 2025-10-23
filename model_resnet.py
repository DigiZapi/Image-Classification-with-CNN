import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow import keras

def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])  # Add the input (shortcut) to the output
    x = layers.ReLU()(x)
    return x

def build_resnet(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Using a loop for flexible residual block definition
    for _ in range(4):  # Define three residual blocks
        x = residual_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)  # Use pooling before dense layer
    x = layers.Dense(128, activation='relu')(x)

    # Add dropout after dense layer to prevent overfitting
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(6, activation='softmax')(x)  # Adjust to output 6 classes

    model = models.Model(inputs, output)
    return model



def model_resnet(x_train, y_train, x_test, y_test):
    model = build_resnet((32, 32, 3))

    # Optimizer with a learning rate
    adam_opt = Adam(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=adam_opt, 
                  metrics=['accuracy'])

    # Implementing learning rate scheduling
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                      factor=0.4, 
                                      patience=5, 
                                      min_lr=1e-6, 
                                      verbose=1)

    # Train the model with learning rate reduction
    model.fit(x_train, y_train, 
              epochs=40, 
              batch_size=256, 
              validation_data=(x_test, y_test), 
              callbacks=[lr_reduction])

    return model
