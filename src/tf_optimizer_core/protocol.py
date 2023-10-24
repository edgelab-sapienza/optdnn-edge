import struct
from enum import IntEnum, auto

from tf_optimizer_core.benchmarker_core import Result


class PayloadMeans(IntEnum):
    ModelPath = auto()
    DatasetPath = auto()
    DatasetScale = auto()
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
    def build_put_model_file_request(cls, file_path: str, ):
        return Protocol(PayloadMeans.ModelPath, file_path.encode())

    @classmethod
    def build_put_dataset_file_request(cls, dataset_url: bytes):
        return Protocol(PayloadMeans.DatasetPath, dataset_url)

    @classmethod
    def build_with_result(cls, result: Result):
        data = struct.pack("!Iff", int(result.node_id), result.accuracy, result.time)
        return Protocol(PayloadMeans.Result, data)

    @classmethod
    def build_by_message(cls, message: bytes):
        message_len = len(message)
        payload_len = message_len - 1
        (command, payload) = struct.unpack(f"!B{payload_len}s", message)
        return Protocol(command, payload)

    @staticmethod
    def get_result_by_msg(msg) -> Result:
        node_id, acc, time = struct.unpack("!Iff", msg.payload)
        r = Result()
        r.time = time
        r.accuracy = acc
        r.node_id = node_id
        return r

    def __str__(self) -> str:
        return f"{self.cmd} - {self.payload}"
