import sys
import random
import string


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
