import tensorflow_datasets as tfds
import tensorflow as tf
import json


dataset = tfds.load(
    "droid", 
    # data_dir="/data/yangyi/datasets/droid_100", 
    data_dir="/data/yangyi/datasets/DROID", 
    split="train",
    download=False
)

success_indices = []
total_count = 0

for episode_index, episode in enumerate(dataset):
    total_count += 1

    tf_contains = tf.strings.regex_full_match(
        episode['episode_metadata']['file_path'],
        '.*/success/.*'
    )
    if bool(tf_contains.numpy()):
        success_indices.append(episode_index)

result = {
    "total_count": total_count,
    "success_indices": success_indices
}

with open("/data/yangyi/datasets/droid_lerobot/meta/episodes_success.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)