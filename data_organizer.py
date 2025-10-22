import numpy as np
from tensorflow.keras.utils import to_categorical

def organize_data(x_train, y_train, x_test, y_test, filter_arr):
    # Creating a boolean mask to filter only animal classes
    train_mask = np.isin(y_train, filter_arr)
    test_mask  = np.isin(y_test, filter_arr)

    x_train_filtered = x_train[train_mask.flatten()]
    y_train_filtered = y_train[train_mask.flatten()]
    x_test_filtered  = x_test[test_mask.flatten()]
    y_test_filtered  = y_test[test_mask.flatten()]
    
    # Relabel classes
    label_map = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    y_train_filtered = np.vectorize(label_map.get)(y_train_filtered)
    y_test_filtered  = np.vectorize(label_map.get)(y_test_filtered)

    x_train_plot = x_train_filtered
    y_train_plot = y_train_filtered

    # Normalize Data
    x_train_filtered = x_train_filtered.astype("float32") / 255.0
    x_test_filtered  = x_test_filtered.astype("float32") / 255.0
    y_train_filtered = to_categorical(y_train_filtered, 6)
    y_test_filtered  = to_categorical(y_test_filtered, 6)

    # Verify if shapes correspond to expectations
    print("x_train_filtered shape:", x_train_filtered.shape)
    print("y_train_filtered shape:", y_train_filtered.shape)
    print("x_test_filtered shape:", x_test_filtered.shape)
    print("y_test_filtered shape:", y_test_filtered.shape)
    return (x_train_filtered, y_train_filtered, x_test_filtered, y_test_filtered, x_train_plot, y_train_plot)