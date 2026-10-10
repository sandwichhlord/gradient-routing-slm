# Architecture

Both diagrams are Mermaid. The sources are `block_diagram.mmd` and `pipeline.mmd`, and the rendered images are `block_diagram.png` and `pipeline.png`.
To re-render: `npx @mermaid-js/mermaid-cli -i docs/block_diagram.mmd -o docs/block_diagram.png -s 2 -b white`.

## (a) Transformer layer with the 8-block MLP and the router / gradient-mask path

![Transformer layer with block-split MLP and routing path](block_diagram.png)

```mermaid
flowchart TB
    subgraph L["Transformer layer l (pre-LN)"]
        X["x (residual stream)"] --> ATT["LayerNorm 1 + causal self-attention"] --> A1["+ residual"]
        X --> A1
        A1 --> FC["LayerNorm 2 + c_fc (up-projection, n_embd to 4 n_embd) + GELU"]
        FC --> HB["Hidden layer = 8 blocks of 4 n_embd / 8 units (192 at n_embd 384)<br/>h0 | h1 | h2 | h3 | h4 | h5 | h6 | h7<br/>(top-2 by energy are 'selected')"]
        HB --> PROJ["c_proj (down-projection, 4 n_embd to n_embd)"] --> A2["+ residual"]
        A1 --> A2
        A2 --> OUT["x to layer l+1"]
    end

    HB -. "forward: read h_b only<br/>(forward pass unchanged)" .-> E

    subgraph R["Router and gradient-mask path (training only, --routing)"]
        E["Energy a_b = mean over tokens of ||h_b||"] --> TK["Top-k (k = 2) blocks by a_b"]
        TK --> M["Gradient mask m_b = 1 if selected, else epsilon (0)"]
        TK --> F["EMA selection frequency f_b"] --> AUX["Aux loss = lambda * sum f_b * a_b / sum a"]
    end

    M == "backward: scale LM-loss gradient of block b<br/>(c_fc rows b, c_proj columns b) by m_b" ==> FC
    M ==> PROJ
    AUX -. "backward: unmasked gradient through c_fc<br/>(keeps idle blocks alive)" .-> FC
```

Notes:
- The router never changes the forward pass. It only reads each block's activations and changes which blocks receive gradient.
- Implementation: `model/model.py` (`BlockMLP`) and `model/routing.py` (`GradientRouter`).
- The aux-loss gradient bypassing the mask is the default (`route_mask_aux=False`). See the README "Method" section for why.

## (b) Pipeline flow

![Pipeline flow](pipeline.png)

```mermaid
flowchart LR
    PREP["1. Data prep (data/prepare.py)<br/>TinyStories + synthetic math + code<br/>train.bin and per-domain val_*.bin"]

    PREP --> DENSE["2. Dense baseline training<br/>configs/baseline.py, routing off"]
    PREP --> LOOP

    subgraph LOOP["3. Routed training step (train.py --routing, configs/routed.py)"]
        direction TB
        S["Activation scoring<br/>a_lb = mean token L2 norm per MLP block"]
        S --> T["Top-k gradient masking<br/>LM-loss gradient reaches only top-2 blocks per layer"]
        T --> B["Load balancing<br/>EMA selection frequency + aux loss, log routing_stats.csv"]
    end

    DENSE -. "planned: start routed run from dense checkpoint" .-> LOOP

    DENSE --> EVAL
    LOOP --> EVAL

    subgraph EVAL["4. Evaluation (same code for both arms)"]
        direction TB
        CURVES["Per-domain val loss curves and samples"]
        ABL["Ablation matrix (eval/ablate.py)<br/>zero each layer.block: loss increase per domain, specialization S"]
        PROBE["Domain probes (planned)"]
    end

    EVAL --> CMP["5. Compare dense vs routed<br/>mean S, damage concentration, per-domain loss"]
```

Notes:
- Activation scoring, masking and load balancing are not separate stages. They run inside every routed training step.
- Today the routed model trains from scratch. Initialising it from the dense checkpoint (the dashed edge) is planned (see `plan.md`).
- The domain probes (dashed box) are planned and do not exist yet.
