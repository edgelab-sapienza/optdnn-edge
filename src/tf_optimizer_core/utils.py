import random
import string
import zipfile
import tempfile
import shutil
import os


def get_random_string(length) -> str:
    # With combination of lower and upper case
    return "".join(random.choice(string.ascii_letters) for i in range(length))


def unzip_file(zipped_path, model_path):
    path = tempfile.mkdtemp()
    with zipfile.ZipFile(zipped_path, "r") as zip:
        zip.extractall(path=path)
    model_file = os.listdir(path)[0]
    full_path = os.path.join(path, model_file)
    shutil.move(full_path, model_path)
    shutil.rmtree(path)


def list_of_files(main_path: str):
    paths = []
    for folder in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder)
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            if os.path.isfile(img_path):
                paths.append(img_path)
    return paths
