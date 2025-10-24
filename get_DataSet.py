import tensorflow as tf
import torch, gc
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



def getData(img_size=(224, 224), batch_size=32, show_sample=True):
    # 🧹 Speicher freigeben
    tf.keras.backend.clear_session()
    torch.cuda.empty_cache()
    gc.collect()

    # 🔹 CIFAR10 laden
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print("Original shapes:")
    print("x_train:", x_train.shape)
    print("x_test:", x_test.shape)



    # Creating a boolean mask to filter only animal classes
    train_mask = np.isin(y_train, [2, 3, 4, 5, 6, 7])
    test_mask  = np.isin(y_test, [2, 3, 4, 5, 6, 7])

    x_train = x_train[train_mask.flatten()]
    y_train_filtered = y_train[train_mask.flatten()]
    x_test = x_test[test_mask.flatten()]
    y_test_filtered  = y_test[test_mask.flatten()]
    
    # Relabel classes
    label_map = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    y_train = np.vectorize(label_map.get)(y_train_filtered)
    y_test  = np.vectorize(label_map.get)(y_test_filtered)




    # 🔍 Channel-Order prüfen (sollte (32, 32, 3) sein)
    if x_train.shape[-1] != 3 and x_train.shape[1] == 3:
        print("⚠️ Channel-first erkannt – wird konvertiert...")
        x_train = np.transpose(x_train, (0, 2, 3, 1))
        x_test = np.transpose(x_test, (0, 2, 3, 1))
        print("Neue shapes:", x_train.shape, x_test.shape)
    else:
        print("✅ Channel-last Format erkannt")

    # Float + One-hot Encoding
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    y_train = to_categorical(y_train, num_classes=6)
    y_test = to_categorical(y_test, num_classes=6)

    AUTOTUNE = tf.data.AUTOTUNE

    # create TensorFlow Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # 🔧 Resize + Preprocess
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

    # 🖼️ Beispielbilder anzeigen (optional)
    if show_sample:
        for batch_x, batch_y in train_dataset.take(1):
            imgs = (batch_x.numpy() + 1.0) / 2.0  # zurückskalieren von [-1,1] auf [0,1]
            plt.figure(figsize=(6, 6))
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(imgs[i])
                plt.axis('off')
            plt.suptitle("Beispielbilder nach Preprocessing")
            plt.show()

    return train_dataset, test_dataset
