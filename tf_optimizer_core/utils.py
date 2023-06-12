import sys
import random
import string
import zipfile
import tempfile
import shutil
import os


def file_callback(blocks: int, block_size: int, size: int):
    blocks_amount = size / block_size
    progress = 100 * blocks / blocks_amount
    progress_str = "{0:.2f}%".format(progress)
    print(f"\rUploading: {progress_str}", end="")
    sys.stdout.flush()
    if blocks >= size // block_size:
        print()


def get_random_string(length) -> str:
    # With combination of lower and upper case
    return "".join(random.choice(string.ascii_letters) for i in range(length))


def unzip_file(zipped_path, model_path):
    path = tempfile.mkdtemp()
    with zipfile.ZipFile(zipped_path, "r") as zip:
        zip.extractall(path=path)
    print(f"TEMP DIR {path}")
    print(os.listdir(path))
    model_file = os.listdir(path)[0]
    full_path = os.path.join(path, model_file)
    print(f"MODEL PATH {full_path}")
    shutil.move(full_path, model_path)
    shutil.rmtree(path)
