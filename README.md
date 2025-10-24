# Image-Classification-with-CNN
This project performs image classification on the CIFAR-10 dataset using Convolutional Neural Networks (CNNs). It explores different approaches including:

  - VGG16 transfer learning
  - Data augmentation to improve generalization
  - Early stopping to prevent overfitting

The main workflow is controlled via Jupyter notebooks, allowing you to train, evaluate, and visualize different models.

---

## Project Structure

- **`README.md`** – This file.
- **`main.ipynb`** – Main notebook controlling the workflow. Execute models via specific notebook cells.  
- **`data_organizer.py`** – Preprocesses and organizes CIFAR-10 data.  
- **`plot_data.py`** – Visualizes sample images and class distributions.
- **`model_simple_cnn_test_1.py`** – Standard CNN model for CIFAR-10 - test case 1.
- **`model_simple_cnn_test_2.py`** – Standard CNN model for CIFAR-10 - test case 2.
- **`model_data_augmentation_test_1.py`** – CNN with data augmentation applied - test case 1.
- **`model_data_augmentation_test_2.py`** – CNN with data augmentation applied - test case 2.
- **`model_transfer_learning_test_1.py`** – CNN with Transfer Learning (one-hot encoding, early stopping) applied - test case 1.
- **`model_tf.py`** – Standard CNN model for CIFAR-10.  
- **`model_resnet.py`** – CNN using a ResNet backbone.
- **`model_ahmad_f1.py`** – ???
- **`main_sofia.ipynb`** – Notebook focusing on training and evaluating the augmented model.
- **`todo.txt`** – ???
- **`POWERPOINT`** – ???

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/DigiZapi/Image-Classification-with-CNN.git
cd Image-Classification-with-CNN
```

## Usage

You can handle everything directly from the `main.ipynb` notebook.  
Currently, the notebook is set up to run each model in separate code blocks.  

If you want to run an additional model:

1. **Import your model** at the top of the notebook or in a new cell:

```python
from model_new_model import build_model  # Replace with your model file and function
```
2. **Create** a new code block in the notebook.

3. **Run your model** in that code block, e.g.:

```python
model = build_model(input_shape=(32, 32, 3), num_classes=10)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=64)
```
