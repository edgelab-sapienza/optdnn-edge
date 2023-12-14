# Server executed in the host that performs the optimizations
import asyncio
import os
import shutil
import tempfile

import requests
import websockets
from tqdm.auto import tqdm

from tf_optimizer_core.benchmarker_core import BenchmarkerCore, Result
from tf_optimizer_core.protocol import Protocol, PayloadMeans
from tf_optimizer_core.utils import unzip_file


# Node
class Node:
    remote_address = None
    interval: tuple[float, float] = (0, 1)
    data_format: str = None

    class RemoteCallback(BenchmarkerCore.Callback):
        def __init__(self, websocket) -> None:
            self.websocket = websocket
            print(websocket)

        async def progress_callback(
                self, acc: float, progress: float, tooked_time: float, model_name: str = ""
        ):
            current_accuracy = "{0:.2f}".format(acc)
            formatted_took_time = "{0:.2f}".format(tooked_time)
            (remote_ip, port) = self.websocket.local_address
            msg = f"{remote_ip}:{port} | Benchmarking: {model_name} - progress: {int(progress)}% - accuracy: {current_accuracy}% - speed: {formatted_took_time} ms"
            msg_bytes = bytes(msg, "utf-8")
            encapsuled_msg = Protocol(PayloadMeans.ProgressUpdate, msg_bytes)
            await self.websocket.send(encapsuled_msg.to_bytes())

    async def process_received_message(self, message, websocket) -> Protocol:
        protocol = Protocol.build_by_message(message)

        # Download model file
        if protocol.cmd == PayloadMeans.ModelPath:
            content = protocol.payload.decode()
            url, model_name, node_id = content.split(Protocol.string_delimiter)
            with requests.get(url, stream=True) as r:
                total_length = int(r.headers.get("Content-Length"))
                fd, path = tempfile.mkstemp(".zip")
                with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                    with open(fd, "wb") as f:
                        shutil.copyfileobj(raw, f)
            print(f"DOWNLOADING IN {path}")
            unzip_file(path, self.MODEL_PATH)
            os.remove(path)
            print(f"MODEL:{model_name} SAVED IN: {self.MODEL_PATH}")
            # zget.get(model_name, self.MODEL_PATH, file_callback)
            result = await self.__test_model__(websocket, model_name)
            result.node_id = node_id
            os.remove(self.MODEL_PATH)
            return Protocol.build_with_result(result)
        # Download dataset
        elif protocol.cmd == PayloadMeans.DatasetPath:
            self.clear_content()
            with requests.get(protocol.payload, stream=True) as r:
                total_length = int(r.headers.get("Content-Length"))
                with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                    with open(self.DATASET_ZIP, "wb") as f:
                        shutil.copyfileobj(raw, f)
            print(f"DATASET SAVED IN:{self.DATASET_ZIP}")
            shutil.unpack_archive(self.DATASET_ZIP, self.DATASET_FOLDER, "zip")
            return Protocol(PayloadMeans.Ok, b"")
        # Delete workspace
        elif protocol.cmd == PayloadMeans.Close:
            if os.path.exists(self.workspace):
                shutil.rmtree(self.workspace)
            os.mkdir(self.workspace)
        elif protocol.cmd == PayloadMeans.DatasetScale:
            content = protocol.payload.decode()
            min_val, max_val = content.split(Protocol.string_delimiter)
            self.interval = (float(min_val), float(max_val))
            print(f"RECEIVED INTERVAL {self.interval}")
        elif protocol.cmd == PayloadMeans.DataFormat:
            content = protocol.payload.decode()
            self.data_format = content
            print(f"RECEIVED {self.data_format}")
        return None

    def __init__(self, port: int, use_multi_core: bool) -> None:
        self.workspace = tempfile.mkdtemp(suffix="_edge_optimizer")
        self.MODEL_PATH = f"{self.workspace}/model.tflite"
        self.DATASET_ZIP = f"{self.workspace}/dataset.zip"
        self.DATASET_FOLDER = f"{self.workspace}/dataset"
        os.makedirs(self.workspace, exist_ok=True)
        self.use_multi_core = use_multi_core
        self.port = port

    def clear_content(self):
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
        os.makedirs(self.workspace)

    async def __test_model__(self, websocket, model_name) -> Result:
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.DATASET_FOLDER):
            callback = Node.RemoteCallback(websocket)
            benchmarker_core = BenchmarkerCore(self.DATASET_FOLDER, use_multicore=self.use_multi_core,
                                               interval=self.interval, data_format=self.data_format)
            result = await benchmarker_core.test_model(
                self.MODEL_PATH, model_name, callback
            )
            return result

    async def recv_msg(self, websocket):
        (remote_ip, _) = websocket.remote_address

        async for message in websocket:
            data = await self.process_received_message(message, websocket)
            if data is not None:
                await websocket.send(data.to_bytes())

    async def serve(self) -> None:
        print("Server started")
        async with websockets.serve(
                self.recv_msg, "0.0.0.0", self.port, ping_interval=None
        ):
            await asyncio.Future()
