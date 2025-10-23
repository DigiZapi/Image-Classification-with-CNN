import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import torch


def create_model(num_classes, train_dataset):

    IMG_SIZE = (32, 32)  # Define the image size
    IMG_SHAPE = IMG_SIZE + (3,)

    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    
    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)
    

    base_model.trainable = False

    base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)


    data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),      # Randomly flip input images
    tf.keras.layers.RandomRotation(0.2),                        # Random rotation
    tf.keras.layers.RandomZoom(0.1),                            # Random zoom
    tf.keras.layers.RandomContrast(0.1)                         # Random contrast adjustment
    ])

    
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.summary()

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                                                    loss=tf.keras.losses.BinaryCrossentropy(),
                                                    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])


    
    return model

def train_model(train_dataset, test_dataset, epochs=30):

    torch.cuda.empty_cache()

    # rescaling pictures for this model
       
    #rescaling_layer = tf.keras.layers.Rescaling(1/127.5, offset=-1)

      
    model = create_model(10, train_dataset)

    history = model.fit(train_dataset,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_dataset)
    

    loss0, accuracy0 = model.evaluate(test_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    print(loss0, accuracy0)



