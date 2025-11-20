#!/bin/bash

# init environment
# export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_GDR_LEVEL=4
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_IB_TC=186
# export NCCL_NVLS_ENABLE=0
# export NCCL_IB_GID_INDEX=3
# export GL00_SOCKET_IFNAME=bond0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_TIMEOUT=22
# export NCCL_IB_RETRY_CNT=7
# export NCCL_IB_HCA=^=mlx5_3,mlx5_4,mlx5_5,mlx5_bond_0
export NCCL_IB_DISABLE=1

ulimit -l unlimited

cd /data/yangyi/metaquery_action_refactoring
source /data/yangyi/miniconda3/bin/activate
conda activate metaquery_action

# torchrun \
#     --nnodes=${PET_NNODES} \
#     --node_rank=${PET_NODE_RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     /data/yangyi/metaquery_action_refactoring/train.py \
#     --run_name metaquery_droid_frz_bckbn \
#     --config_file image_action_training_droid_ddp.yaml \
#     --base_dir /data/yangyi/metaquery_action_refactoring \
#     --logging_dir /data/yangyi/metaquery_action_refactoring/log \
#     > /data/yangyi/metaquery_action_refactoring/log/metaquery_image_action_frz_bckbn_ddp_no_ib.log 2>&1


# torchrun \
#     --nnodes=${PET_NNODES} \
#     --node_rank=${PET_NODE_RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     /data/yangyi/metaquery_action_refactoring/train.py \
#     --run_name metaquery_image_action_language_queue_h200 \
#     --config_file /data/yangyi/metaquery_action_refactoring/configs/image_action_language_training_droid_ddp.yaml \
#     --base_dir /data/yangyi/metaquery_action_refactoring \
#     --logging_dir /data/yangyi/metaquery_action_refactoring/log \
#     > /data/yangyi/metaquery_action_refactoring/log/metaquery_image_action_language_queue_h200.log 2>&1




torchrun \
    --nnodes=${PET_NNODES} \
    --node_rank=${PET_NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    /data/yangyi/metaquery_action_refactoring/train.py \
    --run_name metaquery_image_action_language_aloha_goods_0_1 \
    --config_file /data/yangyi/metaquery_action_refactoring/configs/image_action_language_training_aloha_goods.yaml \
    --base_dir /data/yangyi/metaquery_action_refactoring \
    --logging_dir /data/yangyi/metaquery_action_refactoring/log \
    > /data/yangyi/metaquery_action_refactoring/log/image_action_language_aloha_goods_0_1_ddp.log 2>&1