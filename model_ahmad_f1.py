# ================================================================
# Project: Deep Learning + Linear Regression Analysis
# ================================================================

# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model_f1():
    # ================================================================
    # PART 1: CNN (VGG16) - Image Classification
    # ================================================================

    print("\n--- PART 1: CNN Model (VGG16) ---")

    # Step 1: Load the VGG16 base model
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze convolutional base
    for layer in vgg_base.layers:
        layer.trainable = False

    # Step 2: Add classification layers
    x = Flatten()(vgg_base.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)  # change 10 to number of your classes

    vgg_complete = Model(inputs=vgg_base.input, outputs=output)

    # Step 3: Compile model
    vgg_complete.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Step 4: Prepare your image data
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Step 5: Train the model
    vgg_complete.fit(train_generator, epochs=5, validation_data=test_generator)

    # Step 6: Evaluate the model
    loss, acc = vgg_complete.evaluate(test_generator)
    print(f"✅ CNN Test Accuracy: {acc:.2f}")

    # Step 7: Confusion Matrix
    y_true = test_generator.classes
    y_pred_probs = vgg_complete.predict(test_generator)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_generator.class_indices.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - VGG16 Model")
    plt.show()