import dataclasses
import enum
import logging
import pathlib
import time
from typing import Any

# --- 新增依赖 ---
import msgpack
import numpy as np
import requests
import polars as pl
import rich
import tqdm
import tyro

logger = logging.getLogger(__name__)

# --- 新增: MessagePack与Numpy的序列化/反序列化逻辑 ---
# 与服务器端完全相同


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(data): return msgpack.packb(
    data, default=pack_array, use_bin_type=True)


def unpackb(data): return msgpack.unpackb(
    data, object_hook=unpack_array, raw=False)


# --- Args 和 TimingRecorder 等辅助类保持不变 ---
class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8001
    use_https: bool = False
    num_steps: int = 20
    timing_file: pathlib.Path | None = None
    env: EnvMode = EnvMode.ALOHA_SIM


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        times = self._timings[key]
        return {"mean": float(np.mean(times)), "std": float(np.std(times)), "p25": float(np.quantile(times, 0.25)), "p50": float(np.quantile(times, 0.50)), "p75": float(np.quantile(times, 0.75)), "p90": float(np.quantile(times, 0.90)), "p95": float(np.quantile(times, 0.95)), "p99": float(np.quantile(times, 0.99)), }

    def print_all_stats(self) -> None:
        table = rich.table.Table(title="[bold blue]HTTP Client Timing Statistics[/bold blue]",
                                 show_header=True, header_style="bold white", border_style="blue", title_justify="center")
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)
        stat_columns = [("Mean", "yellow", "mean"), ("Std", "yellow", "std"), ("P25", "magenta", "p25"), ("P50", "magenta", "p50"),
                        ("P75", "magenta", "p75"), ("P90", "magenta", "p90"), ("P95", "magenta", "p95"), ("P99", "magenta", "p99")]
        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


# --- 【核心修改】新的 MessagePack HTTP 客户端类 ---
class MsgPackHttpClientPolicy:
    """一个通过HTTP与FastAPI服务器通信的客户端，使用MessagePack进行序列化。"""

    def __init__(self, host: str, port: int, use_https: bool = False):
        protocol = "https" if use_https else "http"
        if host.startswith("http"):
            self.base_url = host
        else:
            self.base_url = f"{protocol}://{host}:{port}"

        self.infer_url = f"{self.base_url}/infer"
        self.session = requests.Session()
        # 【关键】设置请求头，告知服务器我们发送的是MessagePack数据
        self.session.headers.update({"Content-Type": "application/msgpack"})
        logger.info(
            f"MsgPack client configured to connect to: {self.base_url}")

    def get_server_metadata(self) -> dict:
        try:
            # 根路径仍然返回JSON，方便调试
            response = self.session.get(self.base_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get server metadata: {e}")
            return {"error": str(e)}

    def infer(self, observation: dict) -> dict:
        # 【关键】直接使用msgpack打包包含Numpy数组的字典
        packed_observation = packb(observation)
        try:
            # 【关键】使用 data=... 发送原始二进制数据，而不是 json=...
            response = self.session.post(
                self.infer_url, data=packed_observation, timeout=80)
            response.raise_for_status()

            # 【关键】使用msgpack解包服务器返回的二进制响应
            unpacked_response = unpackb(response.content)
            return unpacked_response
        except requests.exceptions.RequestException as e:
            logger.error(f"Inference request failed: {e}")
            return {"error": str(e)}


def main(args: Args) -> None:
    obs_fn = {
        EnvMode.ALOHA: _random_observation_aloha,
        EnvMode.ALOHA_SIM: _random_observation_aloha,
        EnvMode.DROID: _random_observation_droid,
        EnvMode.LIBERO: _random_observation_libero,
    }[args.env]

    # 使用新的MsgPack客户端
    policy = MsgPackHttpClientPolicy(
        host=args.host, port=args.port, use_https=args.use_https)

    # ... (主循环和计时逻辑与之前完全相同)
    logger.info("Checking server connection...")
    metadata = policy.get_server_metadata()
    # if "error" in metadata:
    #     logger.error("Could not connect to server. Exiting.")
    #     return
    logger.info(f"Successfully connected! Server metadata: {metadata}")

    logger.info("Warming up the server...")
    for _ in range(2):
        policy.infer(obs_fn())

    timing_recorder = TimingRecorder()

    logger.info(f"Running benchmark for {args.num_steps} steps...")
    for _ in tqdm.trange(args.num_steps, desc="Running policy"):
        inference_start_time = time.time()
        action = policy.infer(obs_fn())
        # print(action["actions"].shape)
        if "error" in action:
            logger.error("Stopping benchmark due to an inference error.")
            break
        timing_recorder.record("client_roundtrip_ms",
                               (time.time() - inference_start_time) * 1000)

    timing_recorder.print_all_stats()
    if args.timing_file is not None:
        timing_recorder.write_parquet(args.timing_file)

# --- 随机观测数据生成函数 (完全不变) ---


def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 512, 512), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 512, 512), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 256, 256), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 256, 256), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
    
# uv run /data/yangyi/metaquery_action_refactoring/experiments/aloha/client.py --host "" --env ALOHA