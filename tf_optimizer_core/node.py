# Server executed in the host that performs the optimizations
import asyncio
import websockets

# import zget
# from tf_optimizer_core.utils import file_callback
from tf_optimizer_core.benchmarker_core import BenchmarkerCore
from tf_optimizer_core.protocol import Protocol, PayloadMeans
import shutil
import os
import requests


# Node
class Node:
    workspace = "node_workspace"
    MODEL_PATH = f"{workspace}/model.tflite"
    DATASET_ZIP = f"{workspace}/dataset.zip"
    DATASET_FOLDER = f"{workspace}/dataset"
    benchmarkerCore = None
    remote_address = None

    class RemoteCallback(BenchmarkerCore.Callback):
        def __init__(self, websocket) -> None:
            self.websocket = websocket

        async def progress_callback(
            self, acc: float, progress: float, tooked_time: float, model_name: str = ""
        ):
            current_accuracy = "{0:.2f}".format(acc)
            formatted_tooked_time = "{0:.2f}".format(tooked_time)
            msg = "Benchmarking: {} - progress: {}% - accuracy: {}% - speed: {} ms".format(
                model_name, int(progress), current_accuracy, formatted_tooked_time
            )
            msg_bytes = bytes(msg, "utf-8")
            encapsuled_msg = Protocol(PayloadMeans.ProgressUpdate, msg_bytes)
            await self.websocket.send(encapsuled_msg.to_bytes())

    async def process_received_message(self, message, websocket) -> Protocol:
        protocol = Protocol.build_by_message(message)

        # Download model file
        if protocol.cmd == PayloadMeans.ModelPath:
            content = protocol.payload.decode()
            url, model_name = content.split(Protocol.string_delimiter)
            responce = requests.get(url)
            with open(self.MODEL_PATH, "wb") as f:
                f.write(responce.content)
                f.close()
            print(f"MODEL:{model_name} SAVED IN: {self.MODEL_PATH}")
            # zget.get(model_name, self.MODEL_PATH, file_callback)
            return await self.__test_model(websocket, model_name)
        # Download dataset
        elif protocol.cmd == PayloadMeans.DatasetPath:
            url = requests.get(protocol.payload)
            responce = requests.get(url)
            with open(self.DATASET_ZIP, "wb") as f:
                f.write(responce.content)
                f.close()
            print(f"DATASET SAVED IN:{self.DATASET_ZIP}")
            shutil.unpack_archive(self.DATASET_ZIP, self.DATASET_FOLDER, "zip")
            return Protocol(PayloadMeans.Ok, b"")
        # Delete workspace
        elif protocol.cmd == PayloadMeans.Close:
            shutil.rmtree(self.workspace)
            os.mkdir(self.workspace)
        return None

    def __init__(self, port : int = 12300) -> None:
        os.makedirs(self.workspace, exist_ok=True)
        self.port = port

    def __del__(self):
        shutil.rmtree(self.workspace)

    async def __test_model(self, websocket, model_name) -> Protocol:
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.DATASET_FOLDER):
            callback = Node.RemoteCallback(websocket)
            if self.benchmarkerCore is None:
                self.benchmarkerCore = BenchmarkerCore(self.DATASET_FOLDER)
            result = await self.benchmarkerCore.test_model(
                self.MODEL_PATH, model_name, callback
            )
            return Protocol.build_with_result(result)

    async def recv_msg(self, websocket):
        (remote_ip, _) = websocket.remote_address

        async for message in websocket:
            data = await self.process_received_message(message, websocket)
            if data is not None:
                await websocket.send(data.to_bytes())

    async def serve(self) -> None:
        async with websockets.serve(self.recv_msg, "0.0.0.0", self.port):
            await asyncio.Future()
