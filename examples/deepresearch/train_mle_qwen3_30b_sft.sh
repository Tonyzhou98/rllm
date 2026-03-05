#!/bin/bash

#SBATCH --chdir=/fsx/zyhang/verl/
#SBATCH --nodes 1 
#SBATCH --tasks-per-node 8 
#SBATCH --cpus-per-task 24 
#SBATCH --gpus-per-node 8
#SBATCH --mem 500G
#SBATCH --time=48:00:00
#SBATCH --job-name=mle_syn_qwen3_30b_sft_agent_single_node
#SBATCH --output=/fsx/zyhang/rllm/examples/deepresearch/slurm/mle_syn_qwen3_30b_sft_agent_single_node.stdout
#SBATCH --error=/fsx/zyhang/rllm/examples/deepresearch/slurm/mle_syn_qwen3_30b_sft_agent_single_node.stderr

set -x

CHECKPOINT_PATH=/checkpoints/zyhang

DATA_PATH=/fsx/zyhang/rllm/data/datasets
project_name="algoevolve"
experiment_name="algoevolve_qwen3_30b_mle_sft_single_node"


torchrun --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_PATH/mle_bench_sft/train.parquet \
    data.val_files=$DATA_PATH/mle_bench_sft/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=40000 \
    data.truncation=right \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    optim.lr=1e-5 \
    model.partial_pretrain=/fsx/zyhang/Qwen/Qwen3-30B-A3B-Thinking-2507 \
    trainer.default_local_dir=$CHECKPOINT_PATH/${project_name}/${experiment_name} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.logger=console \
    trainer.save_freq=20 \
    trainer.total_training_steps=200 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true