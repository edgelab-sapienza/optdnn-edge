from tf_optimizer_core.dataset_loader import load
from tf_optimizer_core.benchmarker_core import BenchmarkerCore
import asyncio


def dataset_test():
    a = load(
        "/home/luca/.keras/datasets/flower_photos", size=(200, 200), image_to_take=50
    )
    print(a[0][0].shape[:2])


async def benchmark():
    bm = BenchmarkerCore("/home/luca/.keras/datasets/flower_photos")
    await bm.test_model("../tf_optimizer/opt.tflite")


if __name__ == "__main__":
    # dataset_test()
    asyncio.run(benchmark())
