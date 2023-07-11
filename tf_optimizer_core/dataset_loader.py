import os
import random
import numpy as np
from PIL import Image


def load(dataset_path, size, image_to_take=-1, interval=[0, 1]):
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
    interval_min = interval[0]
    interval_max = interval[1]
    interval_range = interval_max - interval_min

    def open_image(path):
        img = Image.open(path)
        img = img.resize(size)
        img = np.asarray(img)
        img = interval_min + (interval_range * img.astype(np.float32) / 255.0)
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
    random.shuffle(dataset)
    # if batch_size < 0:
    for img in dataset:
        yield (
            open_image(img[0]),
            img[1] if img[1].isdigit() else set_names.index(img[1]),
        )
