#!/bin/bash

DATA_PATH_LIST=""

TP=1
PP=4
EP=32
CP=1
VPP=8
MBS=1
GBS=256
CP_TYPE='megatron_cp_algo'
SEQ_LENGTH=2048
TRAIN_ITERS=20000
ROUTER_BALANCING_TYPE='aux_loss'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

RECOMPUTE_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 8 \
"

MOE_ARGS="
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 1536 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --norm-topk-prob \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-aux-loss-coeff 0.001 \
    --moe-permutation-async-comm \
    --moe-alltoall-overlap-comm \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --gemm-gradient-accumulation-fusion \
    --recompute-activation-function \
    --moe-zero-memory level0 \
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    --sequence-parallel \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 2.0e-4 \
    --lr-decay-style cosine \
    --min-lr 2.0e-5 \
    --lr-warmup-iters 2000 \
    --weight-decay 1e-1 \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

GPT_ARGS="
    --kv-channels 128 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --qk-layernorm \
    --use-mcore-models \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --noop-layers 94,95 \
    --num-layers 96 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --group-query-attention \
    --num-query-groups 4 \
"

DATA_ARGS="
    --data-path ${DATA_PATH_LIST[*]} \
    --split 90,10,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --log-throughput \
    --save-interval 100000 \
    --eval-interval 1000 \
    --eval-iters 100 \
    --no-load-optim \
    --no-load-rng
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $RECOMPUTE_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
    | tee logs/train_mcore_qwen3_235b_adam.log
