batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.0
bias = False

max_iters = 50000
lr_decay_iters = 50000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 1
ckpt_interval = 1000

# optimizer
# learning_rate = 6e-4
# min_lr = 6e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000


comment = 'gpt2_medium' 
save_dir = 'log_gpt2/'+ comment
out_dir = 'out-gpt2/' + comment # save ckpt