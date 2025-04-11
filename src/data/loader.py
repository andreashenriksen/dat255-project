import tensorflow as tf
import tensorflow_datasets as tfds
import logging

logging.basicConfig(level=logging.INFO)


def check_gpu():
    """
    Returns:
        Name of the device used for training.
    """
    device_name = tf.test.gpu_device_name()
    if device_name != "/device:GPU:0":
        raise SystemError("GPU device not found")
    logging.info(f"Found GPU at: {device_name}")
    return device_name


def load_speech_commands():
    """
    Loads the google speech commands, and saves them into 3 different datasets for 
    training, validation and testing.
        
    Returns:
        train_ds, val_ds, test_ds
    """
    logging.info("Loading speech_commands dataset...")
    train_ds = tfds.load("speech_commands", split="train", shuffle_files=True)
    val_ds = tfds.load("speech_commands", split="validation", shuffle_files=True)
    test_ds = tfds.load("speech_commands", split="test", shuffle_files=True)
    logging.info("Finished loading dataset.")
    return train_ds, val_ds, test_ds

def get_dataset_classes(dataset):
    builder = tfds.builder('speech_commands')
    class_names = builder.info.features['label'].names
    labels = [example['label'].numpy() for example in dataset]

    return class_names, labels
