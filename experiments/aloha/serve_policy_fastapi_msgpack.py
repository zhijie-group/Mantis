import dataclasses
import enum
import logging
# import socket
# from typing import Any

# --- 新增导入 ---
import msgpack
import numpy as np
import tyro
import uvicorn
from fastapi import FastAPI, Request, Response  # 导入 Request 和 Response

# from openpi.policies import policy as _policy
# from openpi.policies import policy_config as _policy_config
# from openpi.training import config as _config

from metaquery_vla_utils import MetaQueryVLA
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import v2

# --- 新增: MessagePack与Numpy的序列化/反序列化逻辑 ---
# 我们直接使用官方客户端中的高效实现


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


# --- 模型加载和参数定义的代码完全不变 ---
class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8001
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(config="pi0_aloha", dir="gs://openpi-assets/checkpoints/pi0_base"),
    EnvMode.ALOHA_SIM: Checkpoint(config="pi0_aloha_sim", dir="gs://openpi-assets/checkpoints/pi0_aloha_sim"),
    EnvMode.DROID: Checkpoint(config="pi0_fast_droid", dir="gs://openpi-assets/checkpoints/pi0_fast_droid"),
    EnvMode.LIBERO: Checkpoint(config="pi0_fast_libero", dir="gs://openpi-assets/checkpoints/pi0_fast_libero"),
}


# def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
#     if checkpoint := DEFAULT_CHECKPOINT.get(env):
#         return _policy_config.create_trained_policy(
#             _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
#         )
#     raise ValueError(f"Unsupported environment mode: {env}")


# def create_policy(args: Args) -> _policy.Policy:
#     match args.policy:
#         case Checkpoint():
#             return _policy_config.create_trained_policy(
#                 _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
#             )
#         case Default():
#             return create_default_policy(args.env, default_prompt=args.default_prompt)


def _make_transform(size):
    return v2.Compose([v2.Resize(size), v2.CenterCrop(size)])


def main(args: Args) -> None:
    # 1. 加载策略模型 (不变)
    logging.info("Loading policy...")
    # policy = create_policy(args)
    
    policy = MetaQueryVLA(
        model_id = "/data/yangyi/metaquery_action_refactoring/output_aloha_numbers/metaquery_image_action_language_aloha_numbers/checkpoint-11000",
        checkpoints_dir = "/data/yangyi/metaquery_action_refactoring/checkpoints_aloha_numbers/whole_model/epoch10_05_step6000"
    )

    # logging.info(
    #     f"Policy '{policy.metadata.get('model_name', 'N/A')}' loaded successfully.")

    # if args.record:
    #     policy = _policy.PolicyRecorder(policy, "policy_records")

    # 2. 创建 FastAPI 应用 (不变)
    app = FastAPI(title="Mantis MessagePack Policy Server")

    # 3. 【核心修改】定义使用MessagePack的API端点
    @app.post("/infer", summary="Run policy inference using MessagePack")
    async def infer(request: Request) -> Response:
        """
        接收MessagePack格式的二进制观测数据，并返回MessagePack格式的二进制动作数据。
        """
        # a. 从请求体中读取原始的二进制数据
        body_bytes = await request.body()

        # b. 使用msgpack解包，它会自动将数据重建为包含Numpy数组的字典
        observation = unpackb(body_bytes)

        # c. 调用模型进行推理 (不再需要数据重建，因为MsgPack已经完成了)
        # action = policy.infer(observation)
        
        unnorm_key = "aloha_numbers"
     
        img = observation['images']['cam_high']
        wrist_img = observation['images']['cam_right_wrist']

        img = img.copy()
        wrist_img = wrist_img.copy()
        
        img = img.transpose(1, 2, 0)
        wrist_img = wrist_img.transpose(1, 2, 0)
        
        img_tensor = to_tensor(img)
        wrist_img_tensor = to_tensor(wrist_img)
        
        primary_image_transform = _make_transform(512)
        wrist_image_transform = _make_transform(256)
        
        img_tensor = primary_image_transform(img_tensor)
        wrist_img_tensor = wrist_image_transform(wrist_img_tensor)

        action, _, _ = policy.inference(
            [img_tensor, wrist_img_tensor],
            observation['prompt'],            
            unnorm_key=unnorm_key,
            eval_mode="action_chunking",
            relevant_tokens_threshold=0.1,
        )
        
        print(action)
        print(action.shape)
        action = {"actions": action}

        # d. 使用msgpack打包返回的action字典
        response_bytes = packb(action)

        # e. 将二进制数据作为响应返回，并设置正确的Content-Type
        return Response(content=response_bytes, media_type="application/msgpack")

    @app.get("/", summary="Check server status")
    def root():
        return # {"message": "MessagePack Policy Server is running.", "policy_metadata": policy.metadata}

    # 4. 启动 Uvicorn 服务器 (不变)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
    

#  uv run /data/yangyi/metaquery_action_refactoring/experiments/aloha/serve_policy_fastapi_msgpack.py --port 8000