import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import torch, gc

def getData(img_size=(64, 64), batch_size=16):
    # 🧹 free memory
    tf.keras.backend.clear_session()
    torch.cuda.empty_cache()
    gc.collect()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert to float32 (avoid int ops later)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # ✅ One-hot encode labels once
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    AUTOTUNE = tf.data.AUTOTUNE

    # ✅ Create datasets WITHOUT resizing yet
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # ✅ Apply resize + preprocess lazily per batch
    def preprocess_fn(x, y):
        x = tf.image.resize(x, img_size)
        x = preprocess_input(x)
        return x, y

    train_dataset = (
        train_dataset
        .shuffle(10000)
        .map(preprocess_fn, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    test_dataset = (
        test_dataset
        .map(preprocess_fn, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_dataset, test_dataset
