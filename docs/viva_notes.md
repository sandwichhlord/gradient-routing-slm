# Viva notes: 5-minute mid-term demo

## Script (about 5 minutes)

**0:00 - The question (40 s).** Gradient Routing (Cloud et al., 2024) shows you can localize what a network learns by masking gradients, but a person has to supply the labels that say where each kind of data goes. Our question: can the model's own activations decide that, with no labels? If so, blocks specialise by domain and can be ablated cleanly.

**0:40 - Setup and baseline (60 s).** A 10.6M-parameter GPT (6 layers, 6 heads, 384 dim) on three domains: TinyStories, synthetic arithmetic and Python code, 41M tokens. Each MLP's hidden layer is split into 8 blocks. We ablate each of the 48 blocks in turn and compute a specialization score S (0 = damage spread evenly over domains, 1 = damage in one domain). The dense baseline trained 5000 iterations on a Kaggle T4 in 19 minutes. Final validation loss: story 2.12, math 1.47, code 2.50. **Its S is 0.051**, which is low, as expected: nothing pushes it to separate domains. Damage differs mainly by layer (layer 0 blocks matter 5-9x more), not by domain.

**1:40 - The method (50 s).** For each layer, rank blocks by mean activation norm and keep the top 2. Zero the gradient of the other 6 blocks' weights (their `c_fc` rows and `c_proj` columns). The forward pass stays dense. Add a load-balancing loss, borrowed from Switch Transformer, so blocks do not starve.

**2:30 - The engineering finding (70 s).** Our first version masked everything, including the balancing loss, and our own test failed: in CPU smoke runs 9 of 16 blocks were never selected after 100 iterations, and only 2 blocks per layer were alive after 300. A block that stops being selected gets no gradient that could bring it back, and raising the balancing weight only delayed the collapse. The fix was to compute the balancing loss on a separate graph whose gradient is not masked. We kept the collapse as a regression test. At full scale with the fix, no block died and selection was close to uniform.

**3:40 - The result (80 s).** The routed run, same seed and data: **mean S rose from 0.051 to 0.110**, and 8 blocks now have S above 0.2 where the dense model has none. Cost: validation loss 3-7% higher, training 7% slower. Be plain about the limits: the median block only moved from 0.04 to 0.07; the most specialised blocks are specialised toward math; ablating a block now costs about 10x more loss, so routing also concentrated function into fewer blocks; and this is **one seed per arm**.

**5:00 - Next (20 s).** Multiple seeds first, to see whether the gap is real. Then a label-routing upper bound, domain probes, and unlearning and continual-learning tests.

## Likely questions

**1. Isn't this just mixture of experts?**
No. In an MoE the router is a learned network, it picks experts per token, and the experts are separate modules, so the forward computation is sparse. Here all 8 blocks always compute (the forward pass is unchanged), the choice is made by activation energy with no router parameters, and it is made per micro-batch and layer. It only changes which weights receive gradient. We borrowed the top-2-of-8 shape and the balancing loss from MoE work (Mixtral, Switch), but we get no inference savings. The aim is localization of what is learned, for ablation and unlearning.

**2. Why does the balancing gradient bypass the mask?**
Because with the mask applied to it too, a block that loses selection receives no gradient at all, so nothing can restore it. We measured this at smoke scale: 9 of 16 blocks never selected after 100 iterations, and only 2 of 8 alive per layer after 300. With the aux gradient unmasked, idle blocks keep receiving a signal that raises their energy share. It reaches only the `c_fc` rows of its own layer (a side graph from detached inputs), and a test checks that unselected blocks receive exactly that gradient and nothing else. At full scale the routed run had no dead blocks.

**3. Why do masked blocks still change under AdamW?**
A zero gradient does not stop an Adam step. The first and second moment estimates are running averages, so the update keeps going for roughly 10 steps while it decays, and AdamW's decoupled weight decay shrinks every weight by a fixed factor each step regardless of its gradient. So routing controls what a block learns from, not whether its weights stay constant. If we need strict freezing, we would also mask the optimizer update and exclude those blocks from weight decay. We have not done that.

**4. Why activation energy as the routing signal?**
It is free (already computed), it needs no labels, and it says which blocks the current input drives strongly. The weakness is that it is self-reinforcing, since the blocks that learn get stronger activations and win more. We have no direct evidence that energy tracks domain: block activation norms were no more domain-dependent in the routed model (relative spread across domains 0.31 against 0.30 for dense). Alternatives to try are a Fisher or gradient-norm score (as in EWC).

**5. You got S = 0.11 against 0.05. Is that a real effect?**
Not established yet. It is one seed per arm, so we cannot separate it from run-to-run variation. The numbers also need care: S measures how damage is split across domains, so it cannot distinguish "more domain-specific" from "more critical overall", and ablation damage grew about 10x in the routed model. High-S blocks tend to be high-damage blocks (correlation 0.31). The effect is also mostly math versus the rest. The next step is several seeds per arm, plus linear domain probes as a measure that does not rely on ablation.

**6. Is ablation damage a valid measure of specialization?**
It has confounds. Zeroing a block puts the rest of the network off-distribution, and damage depends on how much a block matters overall, not only on domain. We partly control for this by comparing all conditions on identical validation windows, and S works on shares of damage. It is also why we plan linear domain probes: they ask whether a block encodes the domain, whether or not removing it hurts.

**7. What would count as success, and what if routing shows nothing?**
A routed model whose mean S is clearly above the dense model's across several seeds, at a similar validation loss, with ablating a block's domain hurting that domain much more than the others. The label-routed model is the upper bound. If the gap does not hold up across seeds, that is still a result: activations alone would be too weak a signal, and we would report it that way.

**8. What are the limits of this work so far?**
One seed per arm. A small model (10.6M non-embedding parameters) and narrow data: TinyStories is synthetic, and arithmetic is only partly learned (small sums right, large products and the prompt sum wrong). Code samples are poor for both models, and all three routed ones were whitespace. Routing costs 3-7% validation loss. We have no label-routed baseline yet, and 20 evaluation batches per ablation leave the small damages in the dense model noisy.
