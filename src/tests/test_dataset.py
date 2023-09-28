from tf_optimizer_core.benchmarker_core import BenchmarkerCore
import asyncio
import sys

# from tf_optimizer_core.dataset_loader import load
# import numpy as np


class MyCallback(BenchmarkerCore.Callback):
    async def progress_callback(
        self, acc: float, progress: float, tooked_time: float, model_name: str = ""
    ):
        current_accuracy = "{0:.2f}".format(acc)
        formatted_tooked_time = "{0:.2f}".format(tooked_time)
        print(
            f"\rBenchmarking: {model_name} - progress: {int(progress)}% - accuracy: {current_accuracy}% - speed: {formatted_tooked_time} ms",
            end="",
        )
        sys.stdout.flush()


model_path = "../models_generator/model.tflite"
dataset_path = "../models_generator/imagenet_dataset/"


async def test_dataset():
    bmc = BenchmarkerCore(dataset_path, interval=[-1, 1])
    res = await bmc.test_model(model_path, callback=MyCallback())
    print(res)


"""
def test_model():
    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    input_index = input_details["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    pixel_sizes = interpreter.get_input_details()[0]["shape"][1:3]
    input_size = (pixel_sizes[0], pixel_sizes[1])

    ds = load(dataset_path, input_size, interval=[-1, 1], image_to_take=20)
    for image, label in ds:
        image = np.expand_dims(image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, image)
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        predicted_label = np.argmax(output()[0])
        print(f"{label} {predicted_label}")
"""

asyncio.run(test_dataset())
# test_model()
