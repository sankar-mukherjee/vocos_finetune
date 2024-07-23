#!/usr/bin/env bash

source /efs/smukherjee/.venv/bin/activate

: ${exp_dir:="/efs/smukherjee/vocos/logs/lightning_logs"}

# Define the log directories
log_directories=(

    "oliver_version_14:$exp_dir/oliver_version14"
    "oliver_finetune_version15_with_version17data:$exp_dir/version_11"
    "oliver_finetune_version15_with_version17data2:$exp_dir/version_13"

    # "daphne:$exp_dir/daphne"

)

# Combine log directories into ARGS
ARGS=$(IFS=,; echo "${log_directories[*]}")

# Run TensorBoard
tensorboard --bind_all --port 8882 --logdir_spec=$ARGS
