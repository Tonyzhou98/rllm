#!/bin/bash

#SBATCH --chdir=/fsx/zyhang/rllm/
#SBATCH --nodes 2 
#SBATCH --tasks-per-node 8 
#SBATCH --cpus-per-task 24 
#SBATCH --gpus-per-node 8
#SBATCH --mem 500G
#SBATCH --time=48:00:00
#SBATCH --job-name=mle_syn_qwen3_8b_rl_grpo_agent
#SBATCH --output=/fsx/zyhang/rllm/examples/deepresearch/slurm/mle_syn_qwen3_8b_rl_grpo_agent.stdout
#SBATCH --error=/fsx/zyhang/rllm/examples/deepresearch/slurm/mle_syn_qwen3_8b_rl_grpo_agent.stderr

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

# Find the directory where rllm package is located
CHECKPOINT_PATH=/checkpoints/zyhang
DATA_PATH=/fsx/zyhang/rllm/data/datasets
project_name="algoevolve"
experiment_name="algoevolve_qwen3_8b_mle_syn"
# max_token_per_gpu=$((40960 * 8))


export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping


mkdir -p logs/${project_name}
# rm -rf $CHECKPOINT_PATH/${project_name}/${experiment_name}

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 处理IPv6或多个IP的情况
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export RAY_ADDRESS=$ip_head
echo "IP Head: $ip_head"

# -----------start Ray Head ----------
echo "Starting Ray HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
  ray start --head --node-ip-address=$head_node_ip --port=$port \
    --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block &

sleep 20
# -----------start Ray Worker ----------
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting Ray WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    ray start --address $ip_head \
      --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block &
  sleep 5
done
sleep 30
for i in {1..20}; do
  worker_cnt=$(ray status | grep GPU | grep -o "[0-9.]\+/[0-9.]\+ GPU" | head -n 1 | cut -d/ -f2)
  if [[ "$worker_cnt" == "16.0 GPU" ]]; then
    echo "All workers connected!"
    break
  fi

  echo "current GPUs: ($worker_cnt) Waiting for workers... ($i)"
  sleep 5
done

ray status

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "
    run_root=/fsx/zyhang/rllm/examples/deepresearch/output
    ts=\$(date +%Y%m%d-%H%M%S)
    export DEEPRESEARCH_OUTPUT_DIR=\${run_root}/train-\${ts}
    mkdir -p \"\${DEEPRESEARCH_OUTPUT_DIR}\"
    export NCCL_SOCKET_IFNAME=eth0
    export GLOO_SOCKET_IFNAME=eth0
    python3 -m examples.deepresearch.custom_train \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.val_batch_size=64 \
    data.max_prompt_length=32768 \
    data.max_response_length=8192 \
    data.train_files=$DATA_PATH/mle_bench_syn/train.parquet \
    data.val_files=$DATA_PATH/mle_bench_syn/test.parquet \
    actor_rollout_ref.model.path=/fsx/zyhang/Qwen/Qwen3-8B \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    rllm.compact_filtering.enable=True \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=True \
    rllm.compact_filtering.mask_max_response_length_exceeded=True \
    rllm.compact_filtering.mask_max_turns_exceeded=False \
    rllm.compact_filtering.mask_timeout=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=$CHECKPOINT_PATH/${project_name}/${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.workflow.use_workflow=True \
    rllm.workflow.n_parallel_tasks=64 \
    rllm.stepwise_advantage.enable=False \
    trainer.total_epochs=20 2>&1
  "

wait
