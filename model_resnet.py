import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow import keras
from print_model_score import print_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



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
    history = model.fit(x_train, y_train, 
                        epochs=2, 
                        batch_size=256, 
                        validation_data=(x_test, y_test), 
                        callbacks=[lr_reduction])
                


    # Predictions and confusion matrix
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)
    gt = np.argmax(y_test, axis=1)

    #y_test_conv = np.argmax(y_test_conv, axis=1)
    recall = recall_score(gt, predictions, average='weighted')  # Use 'weighted' for multiclass
    f1 = f1_score(gt, predictions, average='weighted')  # Use 'weighted' for multiclass

    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')
    
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

    return model
