import os

import numpy as np
import tensorflow as tf
from PIL import Image


def load(dataset_path, size, interval: tuple[float, float], data_format=None, image_to_take=-1):
    """Load an image dataset as NumPy arrays.

    Args:
    dataset_path: Path to the dataset directory.
    set_names: List of the data subsets (subdirectories of the dataset directory).
    shuffle: Whether to shuffle the samples. If false, instances will be sorted by
        class name and then by file name.
    seed: Random seed used for shuffling.
    x_dtype: NumPy data type for the X arrays.
    y_dtype: NumPy data type for the Y arrays.
    Returns a tuple of (x, y) tuples corresponding to set_names.
    """

    def open_image(path):
        img = Image.open(path).convert("RGB")
        img = img.resize((size[1], size[0]))  # Tensorflow uses HxW notation, while PIL WxH
        img = np.asarray(img)
        if data_format is None or len(data_format) == 0:
            interval_min = interval[0]
            interval_max = interval[1]
            interval_range = interval_max - interval_min
            img = interval_min + (interval_range * img.astype(np.float32) / 255.0)
        else:
            img = tf.keras.applications.imagenet_utils.preprocess_input(img, mode=data_format)
        return img

    dataset = []

    def listdirs(folder):
        return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

    set_names = sorted(listdirs(dataset_path))
    for set_name in set_names:
        full_path = os.path.join(dataset_path, set_name)
        instance_paths = [
            (os.path.join(full_path, name), set_name) for name in os.listdir(full_path)
        ]
        dataset += instance_paths

    if image_to_take > 0:
        dataset = dataset[: min(image_to_take, len(dataset))]

    for img in dataset:
        yield (
            open_image(img[0]),
            img[1] if img[1].isdigit() else set_names.index(img[1]),
        )
