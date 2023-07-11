import numpy as np
import time
from abc import abstractmethod, ABC
from tf_optimizer_core.dataset_loader import load
import tflite_runtime.interpreter as tflite


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

    def __init__(self, dataset_path: str, interval=[0, 1]) -> None:
        self.dataset_path = dataset_path
        self.__dataset = None
        self.interval = interval

    def __get_dataset__(self, image_size: tuple, images_to_take: int = 100):
        self.__dataset = load(
            self.dataset_path, image_size, images_to_take, interval=self.interval
        )

        return self.__dataset

    async def test_model(
        self, model_path: str, model_name: str = "", callback: Callback = None
    ):
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        input_index = input_details["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        pixel_sizes = interpreter.get_input_details()[0]["shape"][1:3]
        input_size = (pixel_sizes[0], pixel_sizes[1])
        images_to_take = 150
        dataset = self.__get_dataset__(input_size, images_to_take=images_to_take)

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
            output = interpreter.tensor(output_index)
            predicted_label = np.argmax(output()[0])
            if int(predicted_label) == int(label):
                correct += 1
            total += 1

            if callback is not None:
                accuracy = 100 * correct / total
                progress = 100 * total / images_to_take
                await callback.progress_callback(
                    accuracy, progress, tooked_time, model_name
                )
            # End data display

        print()
        r = BenchmarkerCore.Result()
        r.accuracy = 100 * correct / total
        r.time = sum_time / total

        return r
