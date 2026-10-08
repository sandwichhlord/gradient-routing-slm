# Tiny CPU smoke test: runs end-to-end in about a minute on synthetic data.
n_layer = 2
n_head = 4
n_embd = 64
block_size = 64
n_blocks = 8

batch_size = 8
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 10
max_iters = 100

eval_interval = 25
eval_iters = 4
log_interval = 10
ckpt_interval = 50
