# Project: Deep Learning - Image Classification with CNN
# Dataset: CIFAR-10

# --- Step 1: Import Libraries ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
def build_model():
    # --- Step 2: Load CIFAR-10 dataset ---
    print("Loading CIFAR-10 dataset...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # --- Step 3: Visualize some images ---
    plt.figure(figsize=(10,4))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(X_train[i])
        plt.title(class_names[y_train[i][0]])
        plt.axis('off')
    plt.show()

    # --- Step 4: Build CNN model ---
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # --- Step 5: Train the model with EarlyStopping ---
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        callbacks=[early_stop]
    )

    # --- Step 6: Evaluate model ---
    loss, acc = model.evaluate(X_test, y_test_cat)
    print(f"Test Accuracy: {acc:.2f}")

    # --- Step 7: Predict and calculate metrics ---
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)

    # F1 Score, Precision, Recall
    report = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)
    print(classification_report(y_test, y_pred_classes, target_names=class_names))

    # --- Step 8: Plot metrics ---
    # F1-score per class
    f1_scores = [report[cls]['f1-score'] for cls in class_names]

    plt.figure(figsize=(10,5))
    sns.barplot(x=class_names, y=f1_scores)
    plt.title("F1 Score per Class")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45)
    plt.show()

    # --- Step 9: Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(10,8))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # --- Optional Step 10: Transfer Learning (example with VGG16) ---
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

    # Freeze base layers
    for layer in vgg_base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(vgg_base.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)

    vgg_model = Model(inputs=vgg_base.input, outputs=output)
    vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("VGG16 model ready for training...")
