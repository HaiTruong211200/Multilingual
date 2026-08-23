# Frozen NLLB -> Global Adapter -> Q-Former -> Frozen Qwen

Đây là implementation độc lập của MVP trong `multilingual_qa_nllb_qformer_qwen_plan.md`.
Không file nào trong `src/` hoặc `scripts/` cũ bị thay đổi.

## Thành phần

- NLLB encoder và Qwen causal LM được freeze hoàn toàn.
- Global Adapter residual bottleneck, bridge và projector là các phần duy nhất được train.
- Bridge chọn được giữa Q-Former (`qformer`) và Transformer Encoder token-preserving (`transformer`).
- QA teacher forcing dùng `inputs_embeds`; latent và prompt đều được mask khỏi labels.
- Global InfoNCE áp dụng trên mean-pooled output của adapter.
- Local Sinkhorn OT áp dụng trên output hidden state của các Q-Former layer cấu hình được; mask được xử lý riêng cho từng mẫu trong batch nên padding không nhận transport mass.
- Dataset lấy ngẫu nhiên hai ngôn ngữ trong cùng `semantic_group_id`, không ép English làm anchor.

## Dữ liệu

JSON hoặc JSONL, mỗi dòng:

```json
{"question_id":"1-vi","language":"vi","question":"Thủ đô của Pháp là gì?","answer":"Paris","semantic_group_id":"1"}
```

Mỗi group phải có ít nhất hai ngôn ngữ.

## Chạy

Từ repository root:

```bash
pip install -r nllb_qformer_qwen/requirements.txt
python -m nllb_qformer_qwen.train \
  --train_file path/to/train.jsonl \
  --output_dir outputs/nllb-qformer-qwen \
  --batch_size 4 --epochs 3 --bf16
```

Thay Q-Former bằng Transformer Encoder:

```bash
python -m nllb_qformer_qwen.train \
  --train_file path/to/train.jsonl \
  --output_dir outputs/nllb-transformer-qwen \
  --bridge_type transformer --batch_size 4 --epochs 3 --bf16
```

Nhánh `transformer` giữ chuỗi token và mask của NLLB. Nhánh `qformer` nén chuỗi
thành số lượng query cố định. Cả hai đều trả hidden state từng layer để tính OT.

Checkpoint chỉ lưu ba khối trainable trong `bridge.pt`, tránh nhân bản hai backbone lớn.
Đặt `--contrastive_weight 0` hoặc `--ot_weight 0` để chạy ablation tương ứng.

Kiểm thử nhẹ, không tải model:

```bash
pytest -q nllb_qformer_qwen/tests
```
