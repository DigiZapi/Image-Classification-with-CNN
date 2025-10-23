import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def init_matheus(x_train, y_train, x_test, y_test, input_shape=(32, 32, 3)):
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
    model = init_matheus(x_train, y_train, x_test, y_test, input_shape)
    batch_size = 128
    num_classes = 6
    epochs = 50  # <-- Já aumentado

    adam_opt = Adam(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

    # 1. Crie o gerador de Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,       # gira a imagem aleatoriamente em até 15 graus
        width_shift_range=0.1,   # move a imagem horizontalmente
        height_shift_range=0.1,  # move a imagem verticalmente
        shear_range=0.1,         # aplica cisalhamento
        zoom_range=0.1,          # aplica zoom aleatório
        horizontal_flip=True,    # inverte a imagem horizontalmente
        fill_mode='nearest'      # preenche pixels novos da forma mais próxima
    )
    
    # Não precisamos "aumentar" os dados de teste, apenas os de treino
    # O gerador DEVE ser "fitado" nos dados de treino
    datagen.fit(x_train)

    # 2. Mude o model.fit para usar o gerador
    history_m3 = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size), # <-- MUDANÇA AQUI
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        steps_per_epoch=len(x_train) // batch_size # Necessário ao usar gerador
    )
    
    predictions = model.predict(x_test)

    print(predictions.shape)
    predictions = np.argmax(predictions, axis=1)

    # print confusion matrix
    gt = np.argmax(y_test, axis=1)
    confusion_matrix(gt, predictions)