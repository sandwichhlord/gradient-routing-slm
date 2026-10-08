# Baseline dense model for a Kaggle T4 (fp16 AMP). ~10.7M non-embedding params.
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
n_blocks = 8

batch_size = 32
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 200
max_iters = 5000

eval_interval = 250
eval_iters = 50       # 50 x 32 x 256 = ~410K tokens per val domain
log_interval = 50
ckpt_interval = 1000
