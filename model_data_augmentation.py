import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def init_data_augmentation(input_shape=(32, 32, 3)):
    num_classes = 6

    model = Sequential(    [
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'), # Camada "gargalo" para processar features
        BatchNormalization(),          # BN também ajuda em camadas densas
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    return model;

def matheus_model(x_train, y_train, x_test, y_test, input_shape=(32, 32, 3)):
    model = init_data_augmentation(input_shape)
    batch_size = 128
    num_classes = 6
    epochs = 50

    adam_opt = Adam(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

    # 1. Crie o gerador de Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,       
        width_shift_range=0.1,   
        height_shift_range=0.1,  
        shear_range=0.1,         
        zoom_range=0.1,          
        horizontal_flip=True,    
        fill_mode='nearest'      
    )

    datagen.fit(x_train)

    history_m3 = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size), 
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        steps_per_epoch=len(x_train) // batch_size 
    )
    
    predictions = model.predict(x_test)

    print(predictions.shape)
    predictions = np.argmax(predictions, axis=1)

    # print confusion matrix
    gt = np.argmax(y_test, axis=1)
    confusion_matrix(gt, predictions)