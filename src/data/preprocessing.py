import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)

mel_spectrogram_layer = tf.keras.layers.MelSpectrogram(
    sampling_rate=16000,
    num_mel_bins=128,
    sequence_stride=128,
    fft_length=256,
    min_freq=0,
    max_freq=8000,
)


def preprocess_dataset(dataset, target_class_ids):
    logging.info("Preprocessing dataset...")

    def preprocess(sample):
        audio = sample["audio"]
        label = sample["label"]

        audio = normalize_audio_length(audio)
        spectrogram = create_spectrogram(audio)

        one_hot = tf.one_hot(
            tf.cast(tf.equal(label, target_class_ids), tf.int32),
            depth=len(target_class_ids),
        )
        one_hot = tf.reduce_max(one_hot, axis=0)

        return spectrogram, one_hot

    return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)


def normalize_audio_length(audio, target_length=16000):
    audio = tf.cast(audio, tf.float32)
    current_length = tf.shape(audio)[0]

    pad_begin = 0
    pad_end = tf.maximum(0, target_length - current_length)
    paddings = [[pad_begin, pad_end]]
    audio = tf.pad(audio, paddings)

    audio = audio[:target_length]

    return audio


def create_spectrogram(audio):
    audio = tf.expand_dims(audio, 0)

    mel_spectrogram = mel_spectrogram_layer(audio)
    mel_spectrogram = tf.squeeze(mel_spectrogram, axis=0)
    mel_spectrogram = tf.expand_dims(mel_spectrogram, -1)

    return mel_spectrogram


def prepare_datasets(
    train_ds, val_ds, test_ds, target_classes, class_names, batch_size=32
):
    logging.info(f"Preparing datasets with target classes: {target_classes}")

    target_class_ids = [class_names.index(cls) for cls in target_classes]

    train_ds = (
        preprocess_dataset(train_ds, target_class_ids)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = preprocess_dataset(val_ds, target_class_ids).batch(batch_size)
    test_ds = preprocess_dataset(test_ds, target_class_ids).batch(batch_size)

    logging.info("Datasets prepared successfully")
    return train_ds, val_ds, test_ds


def inspect_dataset(dataset):
    for spectrogram, label in dataset.take(1):
        print(f"Spectrogram shape: {spectrogram.shape}")
        print(f"Label shape: {label.shape}")

    return spectrogram.shape[1:]

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
            plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
            plt.title(f'Class: {class_name}')
            plt.colorbar()
            
    plt.tight_layout()
    plt.show()
    
# You can also visualize a single example in more detail
def visualize_single_example(dataset, class_names):
    """Visualize a single example in detail"""
    for spectrograms, labels in dataset.take(1):
        # Get the first example
        spectrogram = spectrograms[0].numpy()
        label = labels[0].numpy()
        
        # Get class name
        class_idx = np.argmax(label)
        class_name = class_names[class_idx]
        
        # Create figure with two subplots
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Remove the channel dimension for plotting
        spectrogram = np.squeeze(spectrogram)
        
        # Display spectrogram
        img = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'Mel Spectrogram - Class: {class_name}')
        ax.set_ylabel('Mel Frequency Bin')
        ax.set_xlabel('Time Frame')
        
        plt.colorbar(img, ax=ax)
        plt.tight_layout()
        plt.show()
        
        break