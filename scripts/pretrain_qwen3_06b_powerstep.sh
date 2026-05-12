#!/bin/bash

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_SAVE_DIR=""
TOKENIZER_PATH=""

TP=1
PP=1
MBS=2
GBS=256
SEQ_LENGTH=2048
TRAIN_ITERS=20000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 5e-4 \
    --min-lr 5e-5 \
    --lr-warmup-iters 2000 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --lr-decay-style cosine \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --optimizer powerstep \
    --powerstep-gamma 0.9 \
    --powerstep-beta 0.1 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

GPT_ARGS="
    --use-mcore-models \
    --sequence-parallel \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --num-layers 28 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --ffn-hidden-size 3072 \
    --max-position-embeddings 32768 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --norm-epsilon 1e-6 \
    --no-gradient-accumulation-fusion \
    --attention-softmax-in-fp32 \
    --exit-on-missing-checkpoint \
    --group-query-attention \
    --num-query-groups 8 \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --bf16
"

DATA_ARGS="
    --data-path "" 
    --split 90,10,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000000 \
    --eval-interval 1000 \
    --eval-iters 100 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
    --log-throughput \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/train_mcore_qwen3_0point6b_powerstep.log
