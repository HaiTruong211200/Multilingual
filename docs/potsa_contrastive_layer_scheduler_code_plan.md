# Code Plan — POTSA-Style Reward-Guided Contrastive Layer Scheduling

## 0. Mục tiêu

Triển khai cơ chế chọn động layer để áp **Contrastive Alignment** cho LLM theo cấu trúc scheduler của POTSA:

\[
\text{task-loss change}
\rightarrow
r_l
\rightarrow
Q_l\;(\text{EMA})
\rightarrow
u_l\;(\text{UCB})
\rightarrow
p_l\;(\text{softmax})
\rightarrow
\text{sample next layer}
\]

Trong phiên bản đầu tiên:

- **Không dùng** Gradient Conflict.
- **Không dùng** Hessian/JVP.
- **Không dùng** parameter-space oracle.
- **Không dùng** trend correction.
- **Không dùng** adaptive \(\lambda_l\).
- Chỉ thay alignment loss của POTSA bằng **Contrastive loss tại intermediate layer của LLM**.
- \(\lambda_{\text{align}}\) là **hằng số toàn cục**.

---

# 1. Training Objective

Với task loss:

\[
\mathcal L_T
\]

và contrastive loss tại layer được chọn \(l_t\):

\[
\mathcal L_{\mathrm{ctr}}^{(l_t)},
\]

training objective tại step \(t\):

\[
\boxed{
\mathcal L_{\mathrm{train}}^{(t)}
=
\mathcal L_T^{(t)}
+
\lambda_{\mathrm{align}}
\mathcal L_{\mathrm{ctr}}^{(l_t,t)}
}
\]

Trong code:

```python
loss = task_loss + lambda_align * contrastive_loss
```

---

# 2. Candidate Layers

Không schedule trên toàn bộ LLM ngay từ đầu.

Ví dụ:

```python
candidate_layers = [8, 10, 12, 14, 16, 18]
```

Nên hỗ trợ config:

```yaml
scheduler:
  candidate_layers: [8, 10, 12, 14, 16, 18]
```

Có thể ablate:

```text
lower-middle
middle
upper-middle
all candidate layers
```

---

# 3. State cần lưu cho mỗi layer

Theo POTSA, với mỗi layer \(l\), lưu:

\[
Q_l,\qquad
n_l,\qquad
L_{l,\mathrm{prev}}
\]

## 3.1. Ý nghĩa

### `Q_l`

EMA reward lịch sử:

\[
Q_l^{(t)}
=
(1-\rho)Q_l^{(t-1)}
+
\rho r_l^{(t)}
\]

### `n_l`

Số lần layer đã được kích hoạt.

### `prev_task_loss_l`

Task loss quan sát tại lần kích hoạt trước của layer đó.

---

## 3.2. Python state

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class LayerState:
    q_value: float = 0.0
    count: int = 0
    prev_task_loss: Optional[float] = None
```

Scheduler:

```python
self.states = {
    layer_idx: LayerState()
    for layer_idx in candidate_layers
}
```

---

# 4. Reward

POTSA chỉ mô tả reward là được tính từ **loss change** giữa hai lần activation của cùng layer, nhưng không đưa explicit equation cho \(r_l^{(t)}\).

Trong implementation đầu tiên dùng:

\[
\boxed{
r_l^{(t)}
=
L_{l,\mathrm{prev}}
-
L_T^{(t)}
}
\]

Trong code:

```python
reward = prev_task_loss - current_task_loss
```

## 4.1. Interpretation

Nếu:

```text
prev_task_loss = 1.25
current_task_loss = 1.10
```

thì:

```text
reward = +0.15
```

Nếu task loss xấu đi:

```text
1.25 -> 1.30
```

thì:

```text
reward = -0.05
```

---

# 5. First Activation

Nếu layer chưa từng được chọn:

```python
state.prev_task_loss is None
```

thì **không tính reward**.

Chỉ lưu:

```python
state.prev_task_loss = current_task_loss
state.count += 1
```

Giữ:

```python
state.q_value
```

không đổi.

Pseudo:

```python
if state.prev_task_loss is None:
    reward = None
else:
    reward = state.prev_task_loss - current_task_loss
```

---

# 6. EMA Reward Update

Sau khi có reward:

\[
\boxed{
Q_l
\leftarrow
(1-\rho)Q_l
+
\rho r_l
}
\]

Config:

```yaml
scheduler:
  reward_ema_rho: 0.1
```

Code:

```python
state.q_value = (
    (1.0 - rho) * state.q_value
    + rho * reward
)
```

---

# 7. UCB Utility

Sau khi cập nhật state, với mọi candidate layer \(l\):

\[
\boxed{
u_l^{(t)}
=
Q_l^{(t)}
+
\beta
\sqrt{
\frac{\log t}
{\max(1,n_l^{(t)})}
}
}
\]

Trong đó:

- \(Q_l\): exploitation.
- UCB bonus: exploration.
- \(\beta\): exploration strength.

Config:

```yaml
scheduler:
  ucb_beta: 0.5
```

Code:

```python
import math

utility = (
    state.q_value
    + beta * math.sqrt(
        math.log(max(global_step, 2))
        / max(1, state.count)
    )
)
```

---

# 8. Temperature-Controlled Softmax Sampling

Từ utility:

\[
\boxed{
p_l
=
\frac{
\exp(u_l/\tau)
}{
\sum_k \exp(u_k/\tau)
}
}
\]

Config:

```yaml
scheduler:
  temperature: 1.0
```

Code:

```python
utilities = torch.tensor(
    [utility_by_layer[l] for l in candidate_layers],
    dtype=torch.float32,
)

probs = torch.softmax(
    utilities / temperature,
    dim=0,
)
```

Sample:

```python
idx = torch.multinomial(
    probs,
    num_samples=1,
).item()

selected_layer = candidate_layers[idx]
```

---

# 9. Warm-up / Initialization

## Option A — Uniform warm-up

Khuyến nghị implementation đầu tiên:

```yaml
scheduler:
  warmup_steps: 100
```

Trong warm-up:

```python
selected_layer = random.choice(candidate_layers)
```

Mục đích:

- mỗi layer được thử ít nhất vài lần;
- có `prev_task_loss`;
- tránh UCB/softmax bị chi phối bởi state chưa đủ dữ liệu.

## Option B — Force each layer once

Có thể deterministic:

```python
warmup_order = candidate_layers
```

và activate mỗi layer ít nhất một lần trước khi bật UCB.

Khuyến nghị:

```text
force each layer once
+
short uniform warm-up
```

---

# 10. Contrastive Representation Extraction

Model forward cần trả hidden states:

```python
outputs = model(
    **batch,
    output_hidden_states=True,
)
```

Lấy:

```python
hidden = outputs.hidden_states[selected_layer]
```

Cần kiểm tra indexing:

```text
hidden_states[0] = embedding output
hidden_states[1] = block 1 output
...
```

Phải map `candidate_layers` đúng theo kiến trúc model.

Nên viết helper:

```python
def get_layer_hidden(hidden_states, layer_idx):
    ...
```

Không hard-code indexing rải rác trong training loop.

---

# 11. Pooling

Với hidden:

```text
[B, T, D]
```

tạo sentence representation:

\[
z^{(l)}
=
P(
\operatorname{Pool}(H^{(l)})
)
\]

Khuyến nghị ban đầu:

```text
masked mean pooling
```

Code:

```python
def masked_mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)

    return summed / denom
```

---

# 12. Projection Head

Nên dùng **shared projection head** cho mọi layer candidate:

```python
projection = nn.Sequential(
    nn.Linear(hidden_dim, proj_dim),
    nn.GELU(),
    nn.Linear(proj_dim, proj_dim),
)
```

Không tạo `projection_l` riêng cho từng layer trong baseline đầu tiên.

Lý do:

- giảm parameter;
- tránh projection riêng hấp thụ khác biệt layer;
- cross-layer comparison sạch hơn.

Config:

```yaml
contrastive:
  projection_dim: 256
  temperature: 0.07
```

---

# 13. Contrastive Loss

Với parallel pair:

```text
source sentence
target sentence
```

tạo:

```python
z_src
z_tgt
```

Normalize:

```python
z_src = F.normalize(z_src, dim=-1)
z_tgt = F.normalize(z_tgt, dim=-1)
```

Similarity:

```python
logits = z_src @ z_tgt.T
logits = logits / contrastive_temperature
```

Labels:

```python
labels = torch.arange(
    logits.size(0),
    device=logits.device,
)
```

Symmetric InfoNCE:

```python
loss_src_tgt = F.cross_entropy(
    logits,
    labels,
)

loss_tgt_src = F.cross_entropy(
    logits.T,
    labels,
)

contrastive_loss = 0.5 * (
    loss_src_tgt
    + loss_tgt_src
)
```

---

# 14. Batch Construction

Một batch phải cung cấp đủ dữ liệu cho:

```text
task loss
+
contrastive loss
```

Khuyến nghị baseline đầu tiên:

```text
same parallel batch
```

Ví dụ MT:

```text
src -> tgt
```

Task:

```text
translation CE
```

Alignment:

```text
src representation <-> tgt representation
```

Tránh dùng hai batch khác nhau trong baseline POTSA-style đầu tiên.

---

# 15. Training Step — Order of Operations

Điểm này phải cố định rõ.

## Step t

### 1. Scheduler đã có `selected_layer`

```python
selected_layer = scheduler.current_layer
```

### 2. Forward

```python
outputs = model(
    ...,
    output_hidden_states=True,
)
```

### 3. Compute task loss

```python
task_loss = ...
```

### 4. Compute reward cho selected layer

Reward dùng **task loss hiện tại**:

```python
reward = scheduler.observe(
    layer_idx=selected_layer,
    task_loss=task_loss.detach().item(),
    global_step=global_step,
)
```

### 5. Compute contrastive loss tại selected layer

```python
contrastive_loss = compute_contrastive_loss(
    outputs,
    layer_idx=selected_layer,
)
```

### 6. Total loss

```python
loss = (
    task_loss
    + lambda_align * contrastive_loss
)
```

### 7. Backward + optimizer

```python
optimizer.zero_grad()

loss.backward()

optimizer.step()

lr_scheduler.step()
```

### 8. Scheduler sample next layer

```python
next_layer = scheduler.sample_next(
    global_step=global_step,
)
```

---

# 16. Scheduler Class API

Khuyến nghị:

```python
class POTSAStyleLayerScheduler:

    def __init__(
        self,
        candidate_layers,
        ema_rho=0.1,
        ucb_beta=0.5,
        temperature=1.0,
        warmup_steps=0,
        seed=42,
    ):
        ...

    def observe(
        self,
        layer_idx,
        task_loss,
        global_step,
    ):
        # Update reward / Q / count / prev loss
        # for the currently activated layer.
        ...

    def get_utilities(
        self,
        global_step,
    ):
        ...

    def get_probabilities(
        self,
        global_step,
    ):
        ...

    def sample_next(
        self,
        global_step,
    ):
        ...

    def state_dict(self):
        ...

    def load_state_dict(self, state):
        ...
```

---

# 17. `observe()` Logic

Pseudo-code:

```python
def observe(
    self,
    layer_idx,
    task_loss,
    global_step,
):
    state = self.states[layer_idx]

    reward = None

    if state.prev_task_loss is not None:
        reward = (
            state.prev_task_loss
            - task_loss
        )

        state.q_value = (
            (1 - self.ema_rho)
            * state.q_value
            + self.ema_rho
            * reward
        )

    state.prev_task_loss = task_loss
    state.count += 1

    return reward
```

---

# 18. `get_utilities()` Logic

```python
def get_utilities(
    self,
    global_step,
):
    step = max(global_step, 2)

    result = {}

    for layer, state in self.states.items():
        explore = (
            self.ucb_beta
            * math.sqrt(
                math.log(step)
                / max(1, state.count)
            )
        )

        result[layer] = (
            state.q_value
            + explore
        )

    return result
```

---

# 19. `sample_next()` Logic

```python
def sample_next(
    self,
    global_step,
):
    utilities = self.get_utilities(
        global_step
    )

    layers = list(self.candidate_layers)

    values = torch.tensor(
        [utilities[l] for l in layers],
        dtype=torch.float32,
    )

    probs = torch.softmax(
        values / self.temperature,
        dim=0,
    )

    idx = torch.multinomial(
        probs,
        1,
        generator=self.generator,
    ).item()

    return layers[idx]
```

---

# 20. Checkpointing

Scheduler state phải save cùng model checkpoint.

Ví dụ:

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "lr_scheduler": lr_scheduler.state_dict(),
    "layer_scheduler": layer_scheduler.state_dict(),
    "global_step": global_step,
}
```

`state_dict()` của scheduler cần chứa:

```text
Q_l
n_l
prev_task_loss_l
temperature
rho
beta
candidate_layers
current_layer
RNG state
```

Resume training phải tái lập đúng sampling trajectory nếu seed/state giống nhau.

---

# 21. Logging

Mỗi step log:

```text
global_step
selected_layer
task_loss
contrastive_loss
total_loss
reward
Q_selected
```

Định kỳ log cho mọi layer:

```text
Q_l
n_l
UCB_l
p_l
```

Ví dụ WandB:

```python
wandb.log({
    "scheduler/selected_layer": selected_layer,
    "scheduler/reward": reward,
    "scheduler/q_value": q_value,
})
```

---

# 22. Heatmap cần lưu

## Selection frequency

```text
x-axis: training step / epoch
y-axis: candidate layer
value: selected/not selected or frequency
```

## Probability distribution

\[
p_l^{(t)}
\]

theo thời gian.

## EMA reward

\[
Q_l^{(t)}
\]

theo thời gian.

Ba plot này rất quan trọng để chứng minh:

```text
layer selection is dynamic
```

thay vì scheduler collapse thành fixed-layer.

---

# 23. Assertions / Safety Checks

Code cần check:

```python
assert selected_layer in candidate_layers
```

```python
assert temperature > 0
```

```python
assert 0 < ema_rho <= 1
```

```python
assert ucb_beta >= 0
```

```python
assert torch.isfinite(loss)
```

```python
assert torch.isfinite(probs).all()
```

và:

```python
assert abs(probs.sum().item() - 1.0) < 1e-5
```

---

# 24. Edge Cases

## NaN task loss

Không update scheduler:

```python
if not math.isfinite(task_loss):
    skip scheduler update
```

## Large reward spikes

Baseline đầu tiên nên log raw reward.

Có thể chuẩn bị optional:

```python
reward = np.clip(
    reward,
    -reward_clip,
    reward_clip,
)
```

nhưng **không bật mặc định** trước khi xem distribution.

## Layer never selected

UCB exploration phải dần ưu tiên layer có count thấp.

Warm-up giúp tránh trường hợp này.

---

# 25. Unit Tests

## Test 1 — EMA

Input:

```text
Q_old = 0.1
reward = 0.2
rho = 0.1
```

Expected:

```text
Q_new = 0.11
```

## Test 2 — UCB exploration

Hai layer cùng Q:

```text
Q1 = Q2
n1 = 100
n2 = 2
```

Expected:

```text
UCB2 > UCB1
```

## Test 3 — Softmax

Expected:

```text
sum(probs) == 1
all(probs >= 0)
```

## Test 4 — Reward

```text
prev_loss = 1.2
curr_loss = 1.0
```

Expected:

```text
reward = +0.2
```

## Test 5 — First activation

Expected:

```text
reward is None
count becomes 1
prev_task_loss initialized
```

## Test 6 — State save/load

Sau save/load:

```text
same Q
same count
same prev loss
same sampled sequence given same RNG state
```

---

# 26. Smoke Test trước full training

Chạy:

```text
50–200 steps
```

với:

```text
3 candidate layers
small batch
```

Kiểm tra:

- task loss finite;
- contrastive loss finite;
- tất cả layers được chọn;
- Q-values thay đổi;
- selection probabilities thay đổi;
- scheduler state restore được.

---

# 27. Baseline Experiments

## B0 — No alignment

```text
L = L_T
```

## B1 — Fixed layer

```text
L = L_T + lambda * L_ctr^(fixed)
```

Chạy từng candidate:

```text
L8
L10
L12
...
```

## B2 — Random layer

```python
selected_layer = random.choice(candidate_layers)
```

## B3 — POTSA-style

```text
reward
-> EMA
-> UCB
-> softmax
-> dynamic layer
```

---

# 28. Required Ablation

### Candidate range

```text
lower-middle only
middle only
all selected layers
```

### `rho`

```text
0.05
0.10
0.20
```

### `beta`

```text
0.1
0.5
1.0
```

### temperature

```text
0.2
0.5
1.0
```

### reward definition

Baseline:

\[
r_l
=
L_{\mathrm{prev}}
-
L_{\mathrm{current}}
\]

Later extension:

```text
relative loss change
trend-corrected reward
gradient-aware reward
```

Nhưng không trộn vào baseline POTSA-style đầu tiên.

---

# 29. Metrics

Downstream:

```text
BLEU
COMET
validation CE
```

Scheduler diagnostics:

```text
selection frequency per layer
mean reward per layer
Q_l
UCB utility
sampling probability
entropy of layer distribution
```

Efficiency:

```text
wall-clock time
samples/sec
peak VRAM
```

---

# 30. Recommended File Structure

```text
project/
│
├── configs/
│   └── potsa_scheduler.yaml
│
├── alignment/
│   ├── pooling.py
│   ├── projection.py
│   └── contrastive.py
│
├── scheduler/
│   ├── __init__.py
│   └── potsa_layer_scheduler.py
│
├── training/
│   ├── trainer.py
│   └── train_step.py
│
├── evaluation/
│   ├── metrics.py
│   └── scheduler_analysis.py
│
├── tests/
│   ├── test_potsa_scheduler.py
│   └── test_contrastive.py
│
└── train.py
```

---

# 31. Config đề xuất ban đầu

```yaml
alignment:
  lambda_align: 0.1
  projection_dim: 256
  contrastive_temperature: 0.07
  pooling: mean

scheduler:
  type: potsa
  candidate_layers:
    - 8
    - 10
    - 12
    - 14
    - 16
    - 18

  reward:
    type: absolute_task_loss_change

  reward_ema_rho: 0.1
  ucb_beta: 0.5
  temperature: 1.0

  warmup_steps: 100
  force_each_layer_once: true

logging:
  scheduler_log_every: 10
  full_scheduler_state_every: 100
```

---

# 32. Version 1 — Minimum Viable Implementation

Ưu tiên implement đúng thứ tự:

## V1.1

- fixed layer;
- output hidden states;
- contrastive loss;
- verify gradients.

## V1.2

- random layer selection;
- verify dynamic layer works.

## V1.3

- `LayerState`;
- reward;
- EMA.

## V1.4

- UCB;
- temperature softmax;
- sampling.

## V1.5

- checkpoint scheduler;
- logging;
- heatmaps.

## V1.6

- full comparison:

```text
fixed vs random vs POTSA-style
```

---

# 33. Version 2 — Sau khi POTSA-style baseline chạy ổn

Chỉ khi V1 ổn mới thử thay reward.

Các extension tách biệt:

```text
V2-A: relative task-loss reward
V2-B: trend-corrected reward
V2-C: gradient-conflict reward
V2-D: parameter-validated utility reward
```

Giữ:

```text
EMA + UCB + softmax
```

không đổi.

Như vậy có thể cô lập contribution:

\[
\boxed{
\text{same scheduler, different reward}
}
\]

---

# 34. Claim thực nghiệm của baseline

Baseline POTSA-style chỉ cần kiểm chứng:

### H1

Dynamic scheduling tốt hơn random/fixed layer.

### H2

Selection probability thay đổi theo training:

\[
p(l\mid t)
\]

không collapse ngay từ đầu.

### H3

Một số candidate ranges phù hợp hơn toàn bộ depth.

Không claim reward hiện tại là causal utility.

---

# 35. Lưu ý khi mô tả POTSA

Trong paper/code comments cần viết chính xác:

> POTSA maintains an EMA reward, selection count, and previous task loss for each candidate layer. When a layer is activated, a reward is derived from the observed loss change, followed by UCB-based exploration and temperature-controlled softmax sampling.

Không viết:

```text
POTSA explicitly defines:
r_l = L_prev - L_current
```

vì paper không công bố explicit equation này.

Trong implementation của mình có thể ghi:

```python
# Concrete instantiation of POTSA's
# "reward from the loss change".
reward = prev_task_loss - current_task_loss
```

---

# 36. Tóm tắt Implementation Core

Toàn bộ V1 có thể tóm lại bằng:

```python
# current selected layer
l = layer_scheduler.current_layer

# forward
task_loss, hidden_states = forward(...)

# update reward state for current layer
layer_scheduler.observe(
    layer_idx=l,
    task_loss=task_loss.detach().item(),
    global_step=global_step,
)

# alignment at selected layer
contrastive_loss = contrastive(
    hidden_states,
    layer_idx=l,
)

# train
loss = task_loss + lambda_align * contrastive_loss

optimizer.zero_grad()
loss.backward()
optimizer.step()

# sample next layer
layer_scheduler.current_layer = (
    layer_scheduler.sample_next(
        global_step=global_step + 1
    )
)
```

Core scheduler:

\[
\boxed{
r_l
\rightarrow
Q_l
\rightarrow
u_l
\rightarrow
p_l
\rightarrow
l_{t+1}
}
\]

Đây là phiên bản cần implement trước khi thử các reward phức tạp hơn.
