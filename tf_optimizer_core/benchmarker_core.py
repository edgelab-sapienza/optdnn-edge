import numpy as np
import time
from abc import abstractmethod, ABC
from tf_optimizer_core.dataset_loader import load
import multiprocessing
from tf_optimizer_core.utils import list_of_files

"""
Apparently the TFLite interpreter built in tflite_runtime is extremly slow in x86 machines
It is designed for ARM devices, it's 100x slower.
So if we are in a x86 enviroments, the full tf will be loaded.
Otherwise will be used the tflite_runtime module
"""
try:
    import tflite_runtime.interpreter as tflite

    Interpreter = tflite.Interpreter
except ModuleNotFoundError:
    print("Detected x86 arch")
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter


# Class to evaluate only one model
class BenchmarkerCore:
    class Result:
        accuracy: float
        time: float

        def __str__(self) -> str:
            return f"Accuracy: {self.accuracy} - Tooked time: {self.time}"

    class Callback(ABC):
        @abstractmethod
        async def progress_callback(
            self, acc: float, progress: float, tooked_time: float, model_name: str = ""
        ):
            pass

    def __init__(
        self, dataset_path: str, interval=[0, 1], use_multicore: bool = True
    ) -> None:
        self.dataset_path = dataset_path
        self.__dataset__ = None
        self.interval = interval
        self.__total_images__ = len(list_of_files(dataset_path))
        if use_multicore:
            self.__number_of_threads__ = multiprocessing.cpu_count()
        else:
            self.__number_of_threads__ = None

    def __get_dataset__(self, image_size: tuple):
        self.__dataset__ = load(self.dataset_path, image_size, interval=self.interval)
        return self.__dataset__

    async def test_model(self, model, model_name: str = "", callback: Callback = None):
        if isinstance(model, bytes):
            interpreter = Interpreter(
                model_content=model, num_threads=self.__number_of_threads__
            )
        else:
            interpreter = Interpreter(
                model_path=model, num_threads=self.__number_of_threads__
            )
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        input_index = input_details["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        pixel_sizes = interpreter.get_input_details()[0]["shape"][1:3]
        input_size = (pixel_sizes[0], pixel_sizes[1])
        dataset = self.__get_dataset__(input_size)

        correct = 0
        total = 0
        sum_time = 0
        for image, label in dataset:
            if input_details["dtype"] == np.uint8 or input_details["dtype"] == np.int8:
                input_scale, input_zero_point = input_details["quantization"]
                image = image / input_scale + input_zero_point

            image = np.expand_dims(image, axis=0).astype(input_details["dtype"])
            interpreter.set_tensor(input_index, image)
            start = time.time() * 1000
            interpreter.invoke()
            end = time.time() * 1000
            tooked_time = end - start
            sum_time += tooked_time
            output = interpreter.get_tensor(output_index)
            predicted_label = np.argmax(output[0])
            if int(predicted_label) == int(label):
                correct += 1
            total += 1

            if callback is not None:
                accuracy = 100 * correct / total
                progress = 100 * total / self.__total_images__
                await callback.progress_callback(
                    accuracy, progress, tooked_time, model_name
                )
            # End data display

        print()
        r = BenchmarkerCore.Result()
        r.accuracy = correct / total
        r.time = sum_time / total

        return r
