# Same as configs/baseline.py, plus activation-based gradient routing (model/routing.py).
# Run with a separate out dir so the baseline isn't overwritten: train.py --config configs/routed.py --out_dir out_routed
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

routing = True
route_k = 2          # blocks kept per layer (of n_blocks)
route_eps = 0.0      # gradient scale for non-selected blocks (0 = hard routing)
route_lambda = 0.01  # load-balancing aux loss weight
route_ema = 0.99     # EMA decay for selection frequencies
route_mask_aux = False  # True: aux-loss gradient is masked too (literal masking; blocks can die at lambda=0.01)
