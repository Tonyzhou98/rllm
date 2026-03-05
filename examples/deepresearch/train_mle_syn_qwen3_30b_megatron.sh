#!/bin/bash

#SBATCH --chdir=/fsx/zyhang/rllm/
#SBATCH --nodes 2
#SBATCH --tasks-per-node 8
#SBATCH --cpus-per-task 24
#SBATCH --gpus-per-node 8
#SBATCH --mem 1000G
#SBATCH --time=96:00:00
#SBATCH --job-name=mle_syn_qwen3_30b_megatron_rl_grpo_agent_two_nodes
#SBATCH --output=/fsx/zyhang/rllm/examples/deepresearch/slurm/mle_syn_qwen3_30b_megatron_rl_grpo_agent_single_node_filter_timeout.stdout
#SBATCH --error=/fsx/zyhang/rllm/examples/deepresearch/slurm/mle_syn_qwen3_30b_megatron_rl_grpo_agent_single_node_filter_timeout.stderr

set -x

export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_NCCL_SYMM_MEM=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1

export SRUN_API_URL="http://10.136.102.133:9000"

CHECKPOINT_PATH=/checkpoints/zyhang
DATA_PATH=/fsx/zyhang/rllm/data/datasets
project_name="algoevolve"
experiment_name="algoevolve_qwen3_30b_mle_syn_two_nodes_filter_timeout_megatron"

# 16 GPUs total (2 nodes x 8 GPUs)
# Keep TP/EP as before, increase PP to 2 to reduce per-rank memory.
gen_tp=4
train_tp=1
train_pp=2
train_ep=8
train_etp=1

max_prompt_len=8192
max_response_len=32768

total_seq_len=$((max_prompt_len + max_response_len))
optimizer_offload_fraction=1.0

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
fi

port=6379
ip_head=$head_node_ip:$port
export RAY_ADDRESS=$ip_head
echo "IP Head: $ip_head"

echo "Starting Ray HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
  ray start --head --node-ip-address="$head_node_ip" --port="$port" \
    --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_PER_NODE" --block &

sleep 20

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting Ray WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    ray start --address "$ip_head" \
      --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_PER_NODE" --block &
  sleep 5
done

sleep 30
expected_gpus="$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)).0 GPU"
for i in {1..30}; do
  worker_cnt=$(ray status | grep GPU | grep -o "[0-9.]\+/[0-9.]\+ GPU" | head -n 1 | cut -d/ -f2)
  if [[ "$worker_cnt" == "$expected_gpus" ]]; then
    echo "All workers connected: $worker_cnt"
    break
  fi
  echo "Current GPUs: ($worker_cnt), waiting... ($i)"
  sleep 5
done

ray status

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "
    run_root=/fsx/zyhang/rllm/examples/deepresearch/output
    ts=\$(date +%Y%m%d-%H%M%S)
    export DEEPRESEARCH_OUTPUT_DIR=\${run_root}/train-\${ts}
    mkdir -p \"\${DEEPRESEARCH_OUTPUT_DIR}\"

    python3 -m examples.deepresearch.custom_train_megatron \
      algorithm.adv_estimator=grpo \
      data.train_batch_size=4 \
      data.val_batch_size=64 \
      data.max_prompt_length=$max_prompt_len \
      data.max_response_length=$max_response_len \
      data.train_files=$DATA_PATH/mle_bench_syn/train.parquet \
      data.val_files=$DATA_PATH/mle_bench_syn/test.parquet \
      actor_rollout_ref.model.path=/fsx/zyhang/Qwen/Qwen3-30B-A3B-Thinking-2507 \
      actor_rollout_ref.hybrid_engine=True \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
      +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
      +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
      +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
      actor_rollout_ref.actor.ppo_mini_batch_size=4 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$total_seq_len \
      actor_rollout_ref.actor.use_dynamic_bsz=True \
      actor_rollout_ref.actor.use_kl_loss=False \
      actor_rollout_ref.actor.kl_loss_coef=0.001 \
      actor_rollout_ref.actor.clip_ratio_high=0.28 \
      actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$train_tp \
      actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$train_pp \
      actor_rollout_ref.actor.megatron.expert_model_parallel_size=$train_ep \
      actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$train_etp \
      actor_rollout_ref.actor.megatron.use_mbridge=True \
      actor_rollout_ref.ref.megatron.use_mbridge=True \
      actor_rollout_ref.model.use_fused_kernels=True \
      actor_rollout_ref.actor.megatron.param_offload=True \
      actor_rollout_ref.actor.megatron.grad_offload=True \
      actor_rollout_ref.actor.megatron.optimizer_offload=True \
      actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$train_tp \
      actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$train_pp \
      actor_rollout_ref.ref.megatron.expert_model_parallel_size=$train_ep \
      actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$train_etp \
      +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
      +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
      +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
      +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
      +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
      +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
      +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
      +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
      +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
      +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
      +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
      +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
      +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
      actor_rollout_ref.ref.megatron.param_offload=True \
      actor_rollout_ref.rollout.tensor_model_parallel_size=$gen_tp \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.mode=async \
      actor_rollout_ref.rollout.enforce_eager=False \
      actor_rollout_ref.rollout.enable_prefix_caching=True \
      actor_rollout_ref.rollout.temperature=1.0 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
      actor_rollout_ref.rollout.n=4 \
      actor_rollout_ref.rollout.val_kwargs.n=1 \
      actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
      actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$total_seq_len \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$total_seq_len \
      critic.megatron.tensor_model_parallel_size=$train_tp \
      critic.megatron.pipeline_model_parallel_size=$train_pp \
      critic.megatron.expert_model_parallel_size=$train_ep \
      critic.megatron.expert_tensor_parallel_size=$train_etp \
      critic.ppo_micro_batch_size_per_gpu=1 \
      critic.ppo_mini_batch_size=4 \
      rllm.compact_filtering.enable=True \
      rllm.compact_filtering.mask_max_prompt_length_exceeded=False \
      rllm.compact_filtering.mask_max_response_length_exceeded=False \
      rllm.compact_filtering.mask_max_turns_exceeded=False \
      rllm.compact_filtering.mask_timeout=True \
      rllm.filter_token_mismatch=False \
      actor_rollout_ref.actor.entropy_coeff=0 \
      rllm.mask_truncated_samples=False \
      trainer.critic_warmup=0 \
      trainer.val_before_train=False \
      trainer.logger=['console','wandb'] \
      trainer.project_name=${project_name} \
      trainer.experiment_name=${experiment_name} \
      trainer.default_local_dir=$CHECKPOINT_PATH/${project_name}/${experiment_name} \
      trainer.n_gpus_per_node=8 \
      trainer.nnodes=$SLURM_JOB_NUM_NODES \
      trainer.save_freq=5 \
      trainer.test_freq=5 \
      trainer.default_hdfs_dir=null \
      rllm.workflow.use_workflow=True \
      rllm.workflow.n_parallel_tasks=8 \
      rllm.stepwise_advantage.enable=False \
      rllm.stepwise_advantage.mode=broadcast \
      rllm.stepwise_advantage.normalize_by_steps=False \
      trainer.total_epochs=20 2>&1
  "

wait
