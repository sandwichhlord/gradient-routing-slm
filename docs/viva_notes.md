# Viva notes: 5-minute mid-term demo

## Script (about 5 minutes)

**0:00 - The question (45 s).** Gradient Routing (Cloud et al., 2024) shows you can localize what a network learns by masking gradients, but a person has to supply the labels that say where each kind of data goes. Our question: can the model's own activations decide that, with no labels? If so, blocks specialise by domain and can be ablated cleanly.

**0:45 - Setup and baseline (75 s).** A 10.6M-parameter GPT (6 layers, 6 heads, 384 dim) on three domains: TinyStories, synthetic arithmetic and Python code, 41M tokens. Each MLP's hidden layer is split into 8 blocks. The dense baseline trained 5000 iterations on a Kaggle T4 in 19 minutes. Final validation loss: story 2.12, math 1.47, code 2.50. We ablate each of the 48 blocks in turn and compute a specialization score S (0 = damage spread evenly over domains, 1 = damage in one domain). **The dense baseline has S = 0.051.** That is low, as expected: nothing pushes the model to separate domains. The damage differs mainly by layer (layer 0 blocks matter 5-9x more), not by domain.

**2:00 - The method (60 s).** For each layer, rank blocks by mean activation norm and keep the top 2. Zero the gradient of the other 6 blocks' weights (their `c_fc` rows and `c_proj` columns). The forward pass stays dense. Add a load-balancing loss, borrowed from Switch Transformer, so blocks do not starve.

**3:00 - The engineering finding (90 s).** Our first version masked everything, including the balancing loss, and our own test failed: after 100 smoke-test iterations 9 of 16 blocks had never been selected, and after 300 iterations only 2 blocks per layer were alive. A block that stops being selected gets no gradient that could bring it back. Raising the balancing weight only delayed the collapse. The fix was to compute the balancing loss on a separate graph whose gradient is not masked, so idle blocks keep getting pushed back toward selection. We kept the collapse as a regression test. **Caveat to state:** this was on CPU smoke runs (16 blocks, one seed). Even with the fix, one block per layer won nearly every step, so the full-scale behaviour is unknown.

**4:30 - Status and next (30 s).** Data pipeline and dense baseline are done. The routing module is implemented and unit-tested but **there is no full-scale routed run yet**, so we have no claim about whether routing increases specialization. Next: the routed run, a label-routing upper bound, domain probes, and multiple seeds.

## Likely questions

**1. Isn't this just mixture of experts?**
No. In an MoE the router is a learned network, it picks experts per token, and the experts are separate modules, so the forward computation is sparse. Here all 8 blocks always compute (the forward pass is unchanged), the choice is made by activation energy with no router parameters, and it is made per micro-batch and layer. It only changes which weights receive gradient. We borrowed the top-2-of-8 shape and the balancing loss from MoE work (Mixtral, Switch), but we get no inference savings. The aim is localization of what is learned, for ablation and unlearning.

**2. Why does the balancing gradient bypass the mask?**
Because with the mask applied to it too, a block that loses selection receives no gradient at all, so nothing can restore it. We measured this: 9 of 16 blocks never selected after 100 iterations, and only 2 of 8 alive per layer after 300. With the aux gradient unmasked, idle blocks keep receiving a signal that raises their energy share. It reaches only the `c_fc` rows of its own layer (a side graph from detached inputs), and a test checks that unselected blocks receive exactly that gradient and nothing else.

**3. Why do masked blocks still change under AdamW?**
A zero gradient does not stop an Adam step. The first and second moment estimates are running averages, so the update keeps going for roughly 10 steps while it decays, and AdamW's decoupled weight decay shrinks every weight by a fixed factor each step regardless of its gradient. So routing controls what a block learns from, not whether its weights stay constant. If we need strict freezing, we would also mask the optimizer update and exclude those blocks from weight decay. We have not done that.

**4. Why activation energy as the routing signal?**
It is free (already computed), it needs no labels, and it says which blocks the current input drives strongly. The weakness is that it is self-reinforcing, since the blocks that learn get stronger activations and win more. We have no evidence yet that energy tracks domain. Alternatives to try are a Fisher or gradient-norm score (as in EWC).

**5. What does S = 0.05 mean, and is it low?**
S = 1 - H(p)/log 3, where p is each block's damage split across the three domains. 0 means damage is spread evenly, 1 means it all falls on one domain. For the dense baseline, the best single block reaches 0.19, and layer 0 matters most for all domains. That is a layer effect, not domain specialisation. Individual damages are small (under 0.1 nats), one seed, so small S differences between blocks are noise. The number is only meaningful relative to a routed run, and we do not have one yet.

**6. Is ablation damage a valid measure of specialization?**
It has confounds. Zeroing a block puts the rest of the network off-distribution, and damage depends on how much a block matters overall, not only on domain. We partly control for this by comparing all conditions on identical validation windows and normalising damage across domains in S. It is also why we plan linear domain probes as a second measure: they ask whether a block encodes the domain, whether or not removing it hurts.

**7. What would count as success, and what if routing shows nothing?**
A routed model whose mean S is clearly above the dense model's across several seeds, at a similar validation loss, with ablating a block's domain hurting that domain much more than the others. The label-routed model is the upper bound. If routing does not beat dense, that is still a result: it would say activations alone are not enough signal, and we would report it as such.

**8. What are the limits of this work so far?**
One dense run with one seed, and a routed run that has only been smoke-tested on CPU. The model is small (10.6M non-embedding parameters) and the data narrow: TinyStories is synthetic, and the arithmetic is only partly learned (the model gets small sums right and 3-digit products wrong). The code samples are poor. Routing cost about 0.1-0.24 nats at smoke scale, and we do not know the full-scale cost.
