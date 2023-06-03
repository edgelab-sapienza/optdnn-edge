import tensorflow as tf
import numpy as np
import time
from abc import abstractmethod, ABC


# Class to evaluate only one model
class BenchmarkerCore:
    class Result:
        accuracy: float
        time: float

    class Callback(ABC):
        @abstractmethod
        async def progress_callback(
            self, acc: float, progress: float, tooked_time: float, model_name: str = ""
        ):
            pass

    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        self.__dataset = None

    def __get_dataset__(
        self, image_size: tuple, images_to_take: int = 100
    ) -> tf.data.Dataset:
        if self.__dataset is not None:  # And size match
            sample = next(self.__dataset.as_numpy_iterator())
            if image_size == sample[0].shape[:2]:  # Check is image size is the same
                return self.__dataset

        self.__dataset = (
            tf.keras.utils.image_dataset_from_directory(
                self.dataset_path, image_size=image_size
            )
            .unbatch()
            .take(images_to_take)
        )
        self.__dataset = self.__dataset.map(lambda x, y: (x / 255, y))
        return self.__dataset

    async def test_model(
        self, model_path: str, model_name: str = "", callback: Callback = None
    ):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        input_index = input_details["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        pixel_sizes = interpreter.get_input_details()[0]["shape"][1:3]
        input_size = (pixel_sizes[0], pixel_sizes[1])
        dataset = self.__get_dataset__(input_size, images_to_take=150)

        correct = 0
        total = 0
        sum_time = 0
        for image, label in dataset:
            if input_details["dtype"] in [np.uint8, np.int8]:
                input_scale, input_zero_point = input_details["quantization"]
                image = image / input_scale + input_zero_point
                image = np.expand_dims(image, axis=0).astype(input_details["dtype"])
            else:
                image = np.expand_dims(image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, image)
            start = time.time() * 1000
            interpreter.invoke()
            end = time.time() * 1000
            tooked_time = end - start
            sum_time += tooked_time
            output = interpreter.tensor(output_index)
            predicted_label = np.argmax(output()[0])
            if predicted_label == label:
                correct += 1
            total += 1

            if callback is not None:
                number_of_elements = sum(map(lambda x: 1, dataset))
                accuracy = 100 * correct / total
                progress = 100 * total / number_of_elements
                await callback.progress_callback(
                    accuracy, progress, tooked_time, model_name
                )
            # End data display

        print()
        r = BenchmarkerCore.Result()
        r.accuracy = 100 * correct / total
        r.time = sum_time / total

        return r
