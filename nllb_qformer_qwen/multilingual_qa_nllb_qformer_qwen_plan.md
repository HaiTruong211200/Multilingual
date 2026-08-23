# Kế hoạch thí nghiệm: Multilingual QA với NLLB Encoder → Global Adapter → Q-Former → Qwen

## 1. Mục tiêu

Xây dựng một kiến trúc modular để tăng cường khả năng multilingual QA của một LLM hiện có mà **không cập nhật trọng số gốc của NLLB Encoder và Qwen**.

Kiến trúc mục tiêu:

\[
x^{(l)}
\rightarrow
\text{NLLB Encoder}_{\text{frozen}}
\rightarrow
\text{Global Adapter}_{\text{trainable}}
\rightarrow
\text{Q-Former}_{\text{trainable}}
\rightarrow
\text{Projector}_{\text{trainable}}
\rightarrow
\text{Qwen}_{\text{frozen}}
\rightarrow
y
\]

Trong đó:

- **NLLB Encoder**: trích xuất representation đa ngôn ngữ.
- **Global Adapter**: giảm sai lệch phân phối ở mức ngôn ngữ / sentence-level.
- **Q-Former**: học latent interface chung và local/fine-grained alignment.
- **Projector**: đưa output Q-Former sang hidden dimension của Qwen.
- **Qwen**: reasoning + answer generation.
- Chỉ train các khối mới thêm vào.

---

## 2. Giả thuyết nghiên cứu

### H1 — Global alignment trước local alignment

Nếu các ngôn ngữ còn lệch nhau mạnh ở mức distribution/global structure, local token/query alignment sẽ phải đồng thời xử lý:

- language-specific bias;
- semantic mismatch;
- tokenization mismatch;
- local correspondence.

Do đó nên thực hiện:

\[
\text{coarse/global alignment}
\rightarrow
\text{fine/local alignment}
\]

thay vì chỉ áp local OT trực tiếp.

### H2 — Shared latent interface

Các câu hỏi tương đương ngữ nghĩa ở nhiều ngôn ngữ nên được ánh xạ về một shared latent QA space:

\[
Z(q^{en})
\approx
Z(q^{vi})
\approx
Z(q^{km})
\approx
Z(q^{th})
\]

nhưng vẫn phải giữ đủ answer-relevant information để frozen Qwen trả lời đúng.

### H3 — Frozen-model interoperability

Một bridge đủ mạnh có thể cho phép hai pretrained models độc lập giao tiếp mà không cần fine-tune trọng số gốc:

\[
\text{Frozen multilingual encoder}
\rightarrow
\text{Trainable interface}
\rightarrow
\text{Frozen reasoner}
\]

---

## 3. Kiến trúc tổng thể

```text
Multilingual Question x^l
        │
        ▼
┌──────────────────────┐
│ NLLB Encoder         │
│ FROZEN               │
└──────────┬───────────┘
           │
           ▼
      H_l ∈ R^(T × d_N)
           │
           ▼
┌──────────────────────┐
│ Global Adapter       │
│ TRAINABLE            │
└──────────┬───────────┘
           │
           ▼
      H~_l ∈ R^(T × d_N)
           │
           ├──────────────► Global Alignment
           │
           ▼
┌──────────────────────┐
│ Q-Former             │
│ TRAINABLE            │
└──────────┬───────────┘
           │
           ▼
      Z_l ∈ R^(M × d_B)
           │
           ├──────────────► Local Alignment
           │
           ▼
┌──────────────────────┐
│ Projector            │
│ TRAINABLE            │
└──────────┬───────────┘
           │
           ▼
      Z_Q ∈ R^(M × d_Q)
           │
           ▼
 [Z_Q ; Prompt Embeddings]
           │
           ▼
┌──────────────────────┐
│ Qwen                 │
│ FROZEN               │
└──────────┬───────────┘
           │
           ▼
        Answer y
```

---

## 4. Global Adapter

### 4.1. Baseline đơn giản nhất

Residual bottleneck MLP:

\[
U =
W_{\text{down}}
\operatorname{LN}(H)
\]

\[
U =
\operatorname{GELU}(U)
\]

\[
\Delta H =
W_{\text{up}}U
\]

\[
\tilde H =
H + \alpha \Delta H
\]

Ví dụ nếu NLLB hidden size là 1024:

```text
1024
 ↓
LayerNorm
 ↓
Linear 1024 → 256
 ↓
GELU
 ↓
Linear 256 → 1024
 ↓
Residual Add
```

### 4.2. Mục đích

Global Adapter không nên làm reasoning hoặc token mixing phức tạp.

Nó chỉ cần:

1. giảm language-specific shift;
2. đưa các language distributions gần nhau hơn;
3. giữ nguyên token-level information để Q-Former xử lý tiếp.

---

## 5. Global Alignment

### 5.1. Input

Với một parallel QA pair:

\[
(q_i^{l_1}, q_i^{l_2})
\]

ta có:

\[
\tilde H_i^{l_1},
\tilde H_i^{l_2}
\]

sau Global Adapter.

Pool:

\[
g_i^{l}
=
Pool(\tilde H_i^{l})
\]

Có thể thử:

- mean pooling;
- attention pooling;
- learned pooling token.

### 5.2. Baseline 1 — InfoNCE

\[
L_{\text{global}}
=
-\frac{1}{B}
\sum_i
\log
\frac{
\exp(sim(g_i^{l_1},g_i^{l_2})/\tau)
}{
\sum_j
\exp(sim(g_i^{l_1},g_j^{l_2})/\tau)
}
\]

Mục tiêu:

\[
\text{same semantics}
\rightarrow
\text{close}
\]

\[
\text{different semantics}
\rightarrow
\text{separated}
\]

### 5.3. Baseline 2 — Relational alignment

Tạo similarity matrix:

\[
S^l_{ij}
=
\cos(g_i^l,g_j^l)
\]

và:

\[
L_{\text{rel}}
=
\|S^{l_1}-S^{l_2}\|_F^2
\]

Mục tiêu là giữ semantic geometry giữa các ngôn ngữ.

### 5.4. Baseline 3 — CKA

\[
L_{\text{CKA}}
=
1-CKA(G^{l_1},G^{l_2})
\]

Dùng chủ yếu để đánh giá và có thể thử làm regularizer.

### 5.5. Baseline 4 — Bias Compensation kiểu POTSA

Không trainable:

\[
b_l
=
\frac{1}{N_l}
\sum_n Pool(H_l^{(n)})
\]

\[
\tilde H_l
=
H_l-b_l
\]

Dùng như một baseline để kiểm tra global language mismatch có gần tuyến tính/offset hay không.

---

## 6. Q-Former

### 6.1. Learnable queries

Khởi tạo:

\[
Q_0
\in
\mathbb{R}^{M\times d_B}
\]

với:

\[
M\in\{16,32,64\}
\]

Mỗi Q-Former block gồm:

\[
Q' =
Q + SelfAttn(Q)
\]

\[
Q'' =
Q' + CrossAttn(Q',\tilde H,\tilde H)
\]

\[
Q_{\text{next}}
=
Q'' + FFN(Q'')
\]

Output:

\[
Z^l
=
QFormer(Q_0,\tilde H^l)
\]

### 6.2. Cấu hình khởi đầu

Đề xuất:

- số query: 32;
- số Q-Former layers: 6 hoặc 8;
- hidden size: 768 hoặc 1024;
- attention heads: 8 hoặc 16.

---

## 7. Local Alignment

### 7.1. Mục tiêu

Sau global alignment, Q-Former học local correspondence giữa latent query tokens.

Với:

\[
Z^{l_1}
=
[z_1^{l_1},...,z_M^{l_1}]
\]

\[
Z^{l_2}
=
[z_1^{l_2},...,z_M^{l_2}]
\]

không giả định:

\[
z_i^{l_1}
\leftrightarrow
z_i^{l_2}
\]

nên dùng soft matching.

### 7.2. OT loss

Cost:

\[
C_{ij}
=
1-\cos(z_i^{l_1},z_j^{l_2})
\]

Optimal transport:

\[
L_{\text{OT}}
=
\min_T
\sum_{ij}
T_{ij}C_{ij}
\]

Có thể dùng entropy-regularized Sinkhorn.

### 7.3. Layer placement

Không áp OT ở tất cả Q-Former layers ngay từ đầu.

Thử:

- middle layers: 3–5;
- late layers: 6–8;
- reward-guided layer selection;
- một layer duy nhất;
- nhiều layer có trọng số.

Ví dụ:

\[
L_{\text{local}}
=
\frac{1}{|I|}
\sum_{\ell\in I}
L_{\text{OT}}^{(\ell)}
\]

---

## 8. Đưa latent vào Qwen

Project:

\[
Z_Q
=
Z W_p
\]

với:

\[
W_p:
d_B\rightarrow d_{Qwen}
\]

Sau đó concatenate với prompt embeddings:

\[
E_{\text{input}}
=
[Z_Q;E(\text{prompt})]
\]

Ví dụ prompt:

```text
Answer the question based on the provided representation.
```

hoặc ngắn hơn:

```text
Answer:
```

Qwen nhận `inputs_embeds`, không cần biến latent thành token text.

---

## 9. QA Task Loss

Với answer target:

\[
a=(a_1,...,a_T)
\]

dùng teacher forcing:

\[
L_{QA}
=
-\sum_t
\log
p_{\text{Qwen}}
(
a_t
\mid
Z,
prompt,
a_{<t}
)
\]

Mask toàn bộ latent và prompt tokens khỏi label loss.

Chỉ answer tokens có label thật.

---

## 10. Loss tổng

### Stage đầu

Bắt đầu đơn giản:

\[
L
=
L_{QA}
\]

### Thêm global alignment

\[
L
=
L_{QA}
+
\lambda_gL_{\text{global}}
\]

### Thêm local alignment

\[
L
=
L_{QA}
+
\lambda_gL_{\text{global}}
+
\lambda_lL_{\text{local}}
\]

Cấu hình mặc định:

\[
L
=
L_{QA}
+
\lambda_gL_{\text{InfoNCE}}
+
\lambda_lL_{\text{OT}}
\]

---

## 11. Training policy

### Frozen

- NLLB Encoder;
- Qwen;
- embedding layer của Qwen;
- LM head của Qwen.

### Trainable

- Global Adapter;
- Q-Former;
- Projector;
- learnable query embeddings.

Gradient vẫn được phép đi xuyên qua Qwen để cập nhật bridge, nhưng:

\[
\nabla_{\theta_{Qwen}}=0
\]

---

## 12. Dữ liệu cần chuẩn bị

Mỗi semantic item nên có:

```json
{
  "question_id": "...",
  "language": "vi",
  "question": "...",
  "answer": "...",
  "semantic_group_id": "..."
}
```

Các câu hỏi thuộc cùng `semantic_group_id` là parallel / semantic-equivalent questions.

Ví dụ:

```text
Group 001

EN: What is the capital of France?
VI: Thủ đô của Pháp là gì?
KM: ...
TH: ...

Answer semantics:
Paris
```

Không nhất thiết answer text giống hệt giữa các ngôn ngữ.

---

## 13. Pair sampling

Không nên luôn dùng English làm anchor.

Thử:

### English pivot

\[
(en,vi),
(en,km),
(en,th)
\]

### Random multilingual pairs

\[
(vi,km),
(km,th),
(th,en),
(vi,en)
\]

### Multi-positive

Một sample có thể có nhiều positive languages trong cùng batch.

---

## 14. Thí nghiệm chính

### E0 — Frozen baseline

\[
NLLBEnc
\rightarrow Linear
\rightarrow Qwen
\]

chỉ QA loss.

### E1 — Q-Former baseline

\[
NLLBEnc
\rightarrow QFormer
\rightarrow Qwen
\]

chỉ QA loss.

### E2 — Global Adapter

\[
NLLBEnc
\rightarrow GlobalAdapter
\rightarrow QFormer
\rightarrow Qwen
\]

chỉ QA loss.

### E3 — Global alignment

\[
L=
L_{QA}
+
\lambda_gL_{\text{InfoNCE}}
\]

### E4 — Local OT

\[
L=
L_{QA}
+
\lambda_lL_{OT}
\]

### E5 — Global + Local

\[
L=
L_{QA}
+
\lambda_gL_{\text{InfoNCE}}
+
\lambda_lL_{OT}
\]

---

## 15. Ablation

### Global alignment method

- None
- Bias Compensation
- Cosine
- InfoNCE
- VICReg
- Relational
- CKA
- CORAL

### Local alignment method

- None
- MSE
- cosine
- OT
- span/query-level contrastive

### Q-Former queries

\[
M\in\{8,16,32,64,80\}
\]

### Q-Former depth

\[
L\in\{2,4,6,8\}
\]

### Alignment layer

Thử:

```text
OT @ layer 2
OT @ layer 4
OT @ layer 6
OT @ layer 8
OT @ {4,6}
OT @ {4,6,8}
```

### Language pairing

- English pivot
- random pairwise
- family-based
- all-pairs

---

## 16. Representation analysis

### Language ID probe

Train linear classifier:

\[
Z^{(\ell)}\rightarrow language
\]

Nếu accuracy giảm sau alignment:

\[
I(Z;Language)\downarrow
\]

### Semantic retrieval

Với \(q_i^{vi}\), tìm nearest neighbor trong English/KH/TH latent set.

Metric:

- Recall@1
- Recall@5
- MRR

### CKA

Đo:

\[
CKA(Z^{en},Z^{vi})
\]

theo layer.

### OT cost

Theo dõi:

\[
OT(Z^{l_1},Z^{l_2})
\]

theo layer và epoch.

### QA information probe

Dùng pooled latent để dự đoán answer class khi dataset cho phép.

Mục tiêu là đảm bảo alignment không làm mất answer-relevant information.

---

## 17. Evaluation QA

### In-language QA

Train/eval cùng ngôn ngữ.

### Cross-lingual QA

Question ngôn ngữ A, answer ngôn ngữ B hoặc language-specific output.

### Zero-shot language QA

Không dùng một số ngôn ngữ trong alignment training nhưng đánh giá ở test.

### Consistency

Với cùng semantic question:

\[
q^{en},q^{vi},q^{km}
\]

đo consistency của answer semantics.

---

## 18. Metrics

### QA

- Exact Match
- F1
- Accuracy
- LLM-as-judge nếu task open-ended, chỉ dùng bổ sung

### Representation

- CKA
- cross-lingual retrieval Recall@K
- language-ID probe accuracy
- OT cost
- intra-class/inter-class distance

### Efficiency

- trainable parameters
- VRAM
- training throughput
- inference latency
- number of latent query tokens

---

## 19. Experimental progression

### Phase 1 — Khả năng nối model

Chỉ thử:

\[
NLLBEnc
\rightarrow
QFormer
\rightarrow
Qwen
\]

với:

\[
L=L_{QA}
\]

Mục tiêu: xác minh frozen Qwen có thể sử dụng latent từ NLLB hay không.

### Phase 2 — Global alignment

Thêm:

\[
GlobalAdapter + InfoNCE
\]

Mục tiêu: giảm multilingual distribution gap.

### Phase 3 — Local alignment

Thêm OT trên Q-Former outputs.

Mục tiêu: tăng fine-grained semantic consistency.

### Phase 4 — Layer study

Xác định layer tốt nhất cho local OT và representation invariance.

### Phase 5 — Generalization

Đánh giá:

- unseen languages;
- unseen language pairs;
- low-resource languages;
- cross-domain QA.

---

## 20. Research questions thực nghiệm

### RQ-A

Việc học một shared latent interface giữa frozen multilingual encoder và frozen LLM có cải thiện multilingual QA hay không?

### RQ-B

Global alignment trước local alignment có giúp cross-lingual generalization tốt hơn chỉ dùng task supervision hoặc chỉ local alignment không?

### RQ-C

Mức độ language invariance của latent space có tương quan với QA performance và zero-shot transfer hay không?

### RQ-D

Local OT nên được áp ở layer nào của Q-Former để cân bằng semantic alignment và task-relevant information?

---

## 21. Baseline bắt buộc

- Qwen trực tiếp với multilingual text.
- NLLB translation-to-English → Qwen.
- Linear projector.
- MLP adapter.
- Q-Former không alignment.
- Q-Former + global only.
- Q-Former + local only.
- Q-Former + global + local.

---

## 22. Cấu hình MVP đề xuất

Bắt đầu với:

```text
NLLB Encoder: frozen
Global Adapter:
    LN
    Linear d → d/4
    GELU
    Linear d/4 → d
    Residual

Q-Former:
    4 layers
    32 queries
    hidden 768

Projector:
    Linear 768 → d_Qwen

Qwen:
    frozen
```

Loss:

\[
L=
L_{QA}
+
0.1L_{\text{InfoNCE}}
+
0.05L_{\text{OT}}
\]

Các hệ số trên chỉ là **điểm khởi đầu để tune**, không phải giá trị mặc định có cơ sở lý thuyết.

---

## 23. Tiêu chí thành công

Kiến trúc được coi là có tín hiệu tốt nếu:

1. Q-Former-only vượt Linear/MLP bridge.
2. Global alignment tăng cross-lingual retrieval và giảm language-ID probe.
3. Global + Local tăng QA accuracy trên low-resource languages.
4. Zero-shot languages được cải thiện dù không xuất hiện trong alignment training.
5. Alignment không làm giảm mạnh QA accuracy ở high-resource languages.
6. Chỉ cần một tỷ lệ nhỏ trainable parameters so với Qwen + NLLB.
