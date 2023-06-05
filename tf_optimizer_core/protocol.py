from enum import IntEnum, auto
from tf_optimizer_core.benchmarker_core import BenchmarkerCore
import struct


class PayloadMeans(IntEnum):
    ModelPath = auto()
    DatasetPath = auto()
    Result = auto()
    Ok = auto()
    Close = auto()
    ProgressUpdate = auto()
    IdentifierMessage = auto()


class Protocol:
    cmd: PayloadMeans
    payload: bytes
    string_delimiter = "#"

    def __init__(self, cmd: PayloadMeans, payload: bytes) -> None:
        self.cmd = cmd
        self.payload = payload

    def to_bytes(self) -> bytes:
        payload_len = len(self.payload)
        return struct.pack(f"!B{payload_len}s", int(self.cmd), self.payload)

    @classmethod
    def build_put_model_file_request(cls, file_path: str,):
        return Protocol(PayloadMeans.ModelPath, file_path.encode())

    @classmethod
    def build_put_dataset_file_request(cls, dataset_url: bytes):
        return Protocol(PayloadMeans.DatasetPath, dataset_url)

    @classmethod
    def build_with_result(cls, result: BenchmarkerCore.Result):
        data = struct.pack("!ff", result.accuracy, result.time)
        return Protocol(PayloadMeans.Result, data)

    @classmethod
    def build_by_message(cls, message: bytes):
        message_len = len(message)
        payload_len = message_len - 1
        (command, payload) = struct.unpack(f"!B{payload_len}s", message)
        return Protocol(command, payload)

    @staticmethod
    def get_evaulation_by_msg(msg) -> BenchmarkerCore.Result:
        acc, time = struct.unpack("!ff", msg.payload)
        r = BenchmarkerCore.Result()
        r.time = time
        r.accuracy = acc
        return r

    def __str__(self) -> str:
        return f"{self.cmd} - {self.payload}"
