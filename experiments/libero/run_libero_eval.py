
import os
import math
from dataclasses import dataclass
from typing import Optional
from skimage.util import view_as_blocks
from PIL import Image

from metaquery_vla_utils import MetaQueryVLA

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

import torch
import random
import time
import torchvision

# Append current directory so that interpreter can find experiments.robot
# sys.path.append("../..")
from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "metaquery_vla"
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    long: bool=False 
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "/data/yangyi/metaquery_action_refactoring/experiments/libero_eval_logs"        

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    
    model_id: str = None
    checkpoints_dir: str = None
    action_dim: int = 7
    future_action_window_size: int = 4

    eval_mode: str = "action_chunking"
    dynamic_tokens_threshold: float = 0.1
    relevant_tokens_threshold: float = 0.1


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_gripper_action(action, binarize=True):
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


def calculate_patch_similarity(patches1, patches2):
    """
    Computes cosine similarity between two sets of patches.
    """
    flat1 = patches1.reshape(len(patches1), -1).astype(np.float32)
    flat2 = patches2.reshape(len(patches2), -1).astype(np.float32)
    
    norm1 = np.linalg.norm(flat1, axis=1)
    norm2 = np.linalg.norm(flat2, axis=1)
    
    dot = np.sum(flat1 * flat2, axis=1)
    cosine_sim = dot / (norm1 * norm2 + 1e-8)
    return cosine_sim


def patchify(image, patch_size):
    """
    Converts an image into non-overlapping patches.
    """
    image = np.array(image)
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0, "Image dimensions must be divisible by patch size."

    if image.ndim == 3:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size, image.shape[2]))
    else:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size))

    patches = blocks.reshape(-1, patch_size, patch_size, image.shape[2]) if image.ndim == 3 else blocks.reshape(-1, patch_size, patch_size)
    return patches


def find_most_different_patches(img_0, img_1, target_size, grid_size, top_percentage=0.1):
    patches1 = patchify(img_0, target_size // grid_size)
    patches2 = patchify(img_1, target_size // grid_size)
    similarity = calculate_patch_similarity(patches1, patches2)

    total_patches = grid_size * grid_size
    top_k = int(total_patches * top_percentage)

    patch_scores = [(idx, similarity[idx]) for idx in range(len(similarity))]
    patch_scores.sort(key=lambda x: x[1], reverse=False)
    top_different_ids = [idx for idx, _ in patch_scores[:top_k]]
    return top_different_ids


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    metaquery_vla = MetaQueryVLA(
        cfg.model_id,
        cfg.checkpoints_dir
    )

    # Initialize local logging
    DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = [512, 256]

    # Start evaluation
    total_episodes, total_successes, total_inference_count = 0, 0, 0
    total_start_time = time.perf_counter()
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=512)

        # Start episodes
        task_episodes, task_successes, task_inference_count = 0, 0, 0
        task_start_time = time.perf_counter()
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            if cfg.eval_mode in ["action_chunking", "action_chunking_dynamic_temporal_agg"]:
                query_frequency = cfg.future_action_window_size
            elif cfg.eval_mode in ["action_chunking_temporal_agg"]:
                query_frequency = 1
            all_time_actions = np.zeros(
                (max_steps + cfg.num_steps_wait, max_steps + cfg.num_steps_wait + cfg.future_action_window_size, cfg.action_dim),
                dtype=np.float64
            )
            
            t = 0
            previous_img = None
            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                img, save_img = get_libero_image(obs, resize_size)    
                # replay_images.append(save_img[0])
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }

                if (t - cfg.num_steps_wait) % query_frequency == 0:
                    actions, top_relation_indices, num_patches = metaquery_vla.inference(
                        observation["full_image"], 
                        task_description, 
                        unnorm_key=cfg.unnorm_key,
                        eval_mode=cfg.eval_mode,
                        relevant_tokens_threshold=cfg.relevant_tokens_threshold,
                    )
                    all_time_actions[t, t:t + cfg.future_action_window_size] = actions
                    task_inference_count += 1
                    total_inference_count += 1

                if cfg.eval_mode in ["action_chunking_dynamic_temporal_agg"] and previous_img is not None:
                    grid_size = int(math.sqrt(num_patches))
                    # target_size = ((img[0].shape[0] + grid_size - 1) // grid_size) * grid_size
                    # img_0 = Image.fromarray(previous_img).resize((target_size, target_size))
                    # img_1 = Image.fromarray(img[0]).resize((target_size, target_size))
                    
                    target_size = ((img[0].shape[1] + grid_size - 1) // grid_size) * grid_size
                    img_0 = torchvision.transforms.functional.to_pil_image(previous_img).resize((target_size, target_size))
                    img_1 = torchvision.transforms.functional.to_pil_image(img[0]).resize((target_size, target_size))
                    
                    top_different_ids = find_most_different_patches(img_0, img_1, target_size, grid_size, top_percentage=cfg.dynamic_tokens_threshold)
                    query_frequency = 1 if set(top_different_ids) & set(top_relation_indices) else cfg.future_action_window_size      

                actions_for_curr_step = all_time_actions[:, t]
                mask = np.any(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[mask]        
                if len(actions_for_curr_step) == 1:
                    raw_action = actions_for_curr_step.squeeze()
                else:
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights /= exp_weights.sum()
                    exp_weights = exp_weights[:, None]
                    raw_action = (actions_for_curr_step * exp_weights).sum(axis=0)

                obs, reward, done, info = env.step(raw_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                previous_img = img[0]

            task_episodes += 1
            total_episodes += 1


            # Save a replay video of the episode
            # save_rollout_video(
            #     replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            # )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()


        task_end_time = time.perf_counter()
        total_end_time = time.perf_counter()
        task_inference_time = task_end_time - task_start_time
        total_inference_time = total_end_time - total_start_time

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        print(f"Current task inference count: {float(task_inference_count) / float(task_episodes)}")
        print(f"Current total inference count: {float(total_inference_count) / float(total_episodes)}")
        print(f"Current task inference time: {float(task_inference_time) / float(task_episodes)}")
        print(f"Current total inference time: {float(total_inference_time) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.write(f"Current task inference count: {float(task_inference_count) / float(task_episodes)}\n")
        log_file.write(f"Current total inference count: {float(total_inference_count) / float(total_episodes)}\n")
        log_file.write(f"Current task inference time: {float(task_inference_time) / float(task_episodes)}\n")
        log_file.write(f"Current total inference time: {float(total_inference_time) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
