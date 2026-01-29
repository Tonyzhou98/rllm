#!/bin/bash

#SBATCH --chdir=/fsx/zyhang/rllm/
#SBATCH --nodes 1 
#SBATCH --tasks-per-node 8 
#SBATCH --cpus-per-task 24 
#SBATCH --gpus-per-node 8
#SBATCH --mem 700G
#SBATCH --time=48:00:00
#SBATCH --job-name=mle_syn_qwen3_8b_rl_grpo_agent_single_node_toy_traj_wise
#SBATCH --output=/fsx/zyhang/rllm/examples/deepresearch/slurm/mle_syn_qwen3_8b_rl_grpo_agent_single_node_toy_traj_wise.stdout
#SBATCH --error=/fsx/zyhang/rllm/examples/deepresearch/slurm/mle_syn_qwen3_8b_rl_grpo_agent_single_node_toy_traj_wise.stderr


set -x

# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

export SRUN_API_URL="http://10.136.114.209:9000"

# Find the directory where rllm package is located
CHECKPOINT_PATH=/checkpoints/zyhang
DATA_PATH=/fsx/zyhang/rllm/data/datasets
project_name="algoevolve"
experiment_name="algoevolve_qwen3_8b_mle_syn_single_node_toy_traj_wise"

run_root=/fsx/zyhang/rllm/examples/deepresearch/output
ts=$(date +%Y%m%d-%H%M%S)
export DEEPRESEARCH_OUTPUT_DIR=${run_root}/train-${ts}
mkdir -p "${DEEPRESEARCH_OUTPUT_DIR}"

PYTHONUNBUFFERED=1 bash -c "python3 -m examples.deepresearch.custom_train \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=32768 \
    data.train_files=$DATA_PATH/mle_bench_syn_toy/train.parquet \
    data.val_files=$DATA_PATH/mle_bench_syn_toy/test.parquet \
    actor_rollout_ref.model.path=/fsx/zyhang/Qwen/Qwen3-8B \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    rllm.compact_filtering.enable=False \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=False \
    rllm.compact_filtering.mask_max_response_length_exceeded=False \
    rllm.compact_filtering.mask_max_turns_exceeded=False \
    rllm.compact_filtering.mask_timeout=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=$CHECKPOINT_PATH/${project_name}/${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.workflow.use_workflow=True \
    rllm.workflow.n_parallel_tasks=64 \
    rllm.stepwise_advantage.enable=False \
    trainer.total_epochs=100 2>&1
      "

wait