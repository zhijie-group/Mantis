# torchrun --nproc-per-node=8 train.py \
#     --run_name metaquery_action_droid_gap4_6actquery_3gapquery \
#     --config_file /data/yangyi/metaquery_action_refactoring/configs/action_training_droid.yaml \
#     --base_dir /data/yangyi/metaquery_action_refactoring \
#     --logging_dir /data/yangyi/metaquery_action_refactoring/log \
#     > /data/yangyi/metaquery_action_refactoring/log/metaquery_action.log 2>&1

# torchrun --nproc-per-node=8 train.py \
#     --run_name metaquery_action_droid_gap4_6actquery_3gapquery \
#     --config_file /data/yangyi/metaquery_action_refactoring/configs/image_action_training_libero.yaml \
#     --base_dir /data/yangyi/metaquery_action_refactoring \
#     --logging_dir /data/yangyi/metaquery_action_refactoring/log \
#     > /data/yangyi/metaquery_action_refactoring/log/metaquery_spatial_image_action.log 2>&1


# torchrun --nproc-per-node=8 train.py \
#     --run_name metaquery_image_action_language \
#     --config_file /data/yangyi/metaquery_action_refactoring/configs/image_action_language_training_droid_ddp.yaml \
#     --base_dir /data/yangyi/metaquery_action_refactoring \
#     --logging_dir /data/yangyi/metaquery_action_refactoring/log \
#     > /data/yangyi/metaquery_action_refactoring/log/metaquery_image_action_language_without_language.log 2>&1


torchrun --nproc-per-node=8 train.py \
    --run_name metaquery_image_action_language_aloha_numbers \
    --config_file /data/yangyi/metaquery_action_refactoring/configs/image_action_language_training_aloha_numbers.yaml \
    --base_dir /data/yangyi/metaquery_action_refactoring \
    --logging_dir /data/yangyi/metaquery_action_refactoring/log \
    > /data/yangyi/metaquery_action_refactoring/log/image_action_language_aloha_numbers.log 2>&1