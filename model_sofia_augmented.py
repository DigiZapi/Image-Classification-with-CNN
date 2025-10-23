import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

def init_model_sofia_augmented(input_shape=(32, 32, 3)):

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.summary()
    return model


def model_sofia_augmented(x_train, y_train, x_test, y_test, input_shape=(32, 32, 3)):

    # Normalize images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Split train/validation
    x_train_aug, x_val, y_train_aug, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    #------------

    # Initialize model
    model = init_model_sofia_augmented(input_shape)

    # Compile (setup model before training)
    model.compile(optimizer='adam', # default learning_rate=0.001
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    # Early stopping (same as in your notebook)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Data Augmentation
    # Split your data manually
    x_train_aug, x_val, y_train_aug, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train_aug)

    # Train model
    history_data_aug = model.fit(
        datagen.flow(x_train_aug, y_train_aug, batch_size=64),
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=[early_stop]
    )

    # Double check that early stopped wa striggered
    print("Early stopping triggered at epoch:", early_stop.stopped_epoch + 1)  # +1 because it's 0-indexed
    
    # Evaluate model
    # test_loss → numeric value representing how wrong the model is on the test data (smaller is better).
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # Predict classes for test set
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return model, history_data_aug