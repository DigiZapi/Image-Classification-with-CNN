import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import torch
import gc

# ============================================================
# Optimized create_model (keeps your name & parameters)
# ============================================================
def create_model(num_classes, train_dataset):
    IMG_SIZE = (224, 224)  # upscale CIFAR10 for better feature extraction
    IMG_SHAPE = IMG_SIZE + (3,)

    base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    base_model.trainable = False  # freeze base model

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2)
    ], name="data_augmentation")

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Resize CIFAR images to match MobileNetV2 expectations
    x = layers.Resizing(IMG_SIZE[0], IMG_SIZE[1])(inputs)
    x = data_augmentation(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def model_v2_build(train_dataset, test_dataset, epochs=40):
    tf.keras.backend.clear_session()
    torch.cuda.empty_cache()
    gc.collect()

    # Use mixed precision for faster GPU training (if available)
    #tf.keras.mixed_precision.set_global_policy('mixed_float16')

    tf.keras.backend.clear_session()
    
    model = create_model(6, train_dataset)
    model.summary()

    # Prefetch to improve input pipeline performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    print("\n✅ Training classifier head...")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        verbose=1,
        validation_data=test_dataset
    )

    # Evaluate before fine-tuning
    loss0, accuracy0 = model.evaluate(test_dataset, verbose=0)
    print(f"Initial frozen model — loss: {loss0:.4f}, acc: {accuracy0:.4f}")

    # Fine-tuning phase (optional but recommended)
    print("Fine-tuning top layers of MobileNetV2...")
    base_model = model.get_layer(index=4)  # your MobileNetV2 is the 5th layer
    base_model.trainable = True

    # Recompile with a lower LR for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    fine_tune_epochs = max(5, epochs // 2)
    model.fit(
        train_dataset,
        epochs=fine_tune_epochs,
        verbose=1,
        validation_data=test_dataset
    )

    loss, accuracy = model.evaluate(test_dataset)
    print(f"\n✅ Final model — loss: {loss:.4f}, acc: {accuracy:.4f}")

    return model
