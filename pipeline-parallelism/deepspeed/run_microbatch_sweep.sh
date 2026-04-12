#!/bin/bash
# Sweep microbatch counts: m = gradient_accumulation_steps
# global_batch = micro_batch_size × gradient_accum × data_parallel_size
# With micro_batch_size=2, DP=1: global_batch = 2 × m

for M in 4 8 16 32 64; do
    GLOBAL_BATCH=$((2 * M))

    echo "=========================================="
    echo "  Running PP with m=$M microbatches (global_batch=$GLOBAL_BATCH)"
    echo "=========================================="

    # Create per-run config
    cat > ds_config_m${M}.json << EOF
{
    "train_batch_size": ${GLOBAL_BATCH},
    "train_micro_batch_size_per_gpu": 4,
    "steps_per_print": 999,
    "optimizer": {
        "type": "Adam",
        "params": {"lr": 3e-4, "betas": [0.9, 0.999]}
    },
    "fp16": {"enabled": true, "initial_scale_power": 12},
    "pipeline": {"pipe_partitioned": true, "grad_partitioned": true},
    "wall_clock_breakdown": false
}
EOF

    deepspeed --num_gpus=4 train_pipeline.py \
        --deepspeed_config ds_config_m${M}.json \
        --seq_len 1024 \
        --num_steps 15 \
        --warmup_steps 5 \
        --tag pp4_m${M}

    echo ""
done
