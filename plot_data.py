import matplotlib.pyplot as plt
import numpy as np

def plot_raw_data(class_names, x_data, y_data, nrows, ncols, figsize=(15, 15), iterations=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for i in range(iterations):
        # Get indices for images of class i
        indices = np.where(y_data == i)[0]
        # Randomly select 10 indices from the class
        random_indices = np.random.choice(indices, size=10, replace=False)
        
        for j, idx in enumerate(random_indices):
            # Plot the image in the appropriate subplot
            axes[i, j].imshow(x_data[idx])
            axes[i, j].axis('off')  # Hide axis ticks
            if j == 0:
                axes[i, j].set_ylabel(class_names[i], fontsize=12)  # Label the class

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.show()