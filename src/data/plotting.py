from sklearn.metrics import confusion_matrix
from matplotlib.colors import LogNorm
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def class_histogram(n_classes, class_names, labels):
    bin_edges = np.arange(-0.5, n_classes)
    plt.figure(figsize=(15, 6))
    plt.hist(labels, bins=bin_edges, edgecolor="black")
    plt.xticks(ticks=range(n_classes), labels=class_names, ha="center")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.show()


def visualize_dataset(dataset, class_names, num_examples=10):
    """
    Visualize examples from a prepared dataset of spectrograms

    Args:
        dataset: A TensorFlow dataset containing (spectrogram, label) pairs
        class_names: List of target class names
        num_examples: Number of examples to display
    """
    plt.figure(figsize=(12, 10))

    # Get a batch of examples
    for i, (spectrograms, labels) in enumerate(dataset.take(1)):
        # Only take up to num_examples
        spectrograms = spectrograms[:num_examples]
        labels = labels[:num_examples]

        # Determine grid size
        grid_size = int(np.ceil(np.sqrt(num_examples)))

        for j in range(min(num_examples, len(spectrograms))):
            # Get spectrogram and label
            spectrogram = spectrograms[j].numpy()
            label = labels[j].numpy()

            # Get class name
            class_idx = np.argmax(label)
            class_name = class_names[class_idx]

            # Plot spectrogram
            plt.subplot(grid_size, grid_size, j + 1)

            # Remove the channel dimension for plotting
            spectrogram = np.squeeze(spectrogram)

            # Display as an image
            plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
            plt.title(f"Class: {class_name}")
            plt.colorbar()

    plt.tight_layout()
    plt.show()


def plot_history(history):
    _, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].plot(history.history["accuracy"], label="train")
    axs[0].plot(history.history["val_accuracy"], label="validation")
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()

    axs[1].plot(history.history["loss"], label="train")
    axs[1].plot(history.history["val_loss"], label="validation")
    axs[1].set_title("Model Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, dataset, class_names):
    y_pred = []
    y_true = []

    for x, y in dataset:
        predictions = model.predict(x)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(y, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        norm=LogNorm(),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def get_audio_samples(dataset, dataset_info, amount):
    for example in dataset.take(amount):
        audio = example[0].numpy()
        sample_rate = dataset_info.features["audio"].sample_rate

        ipd.display(ipd.Audio(audio, rate=sample_rate))

        plt.figure(figsize=(10, 4))
        plt.plot(audio)
        plt.title("Waveform")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()
