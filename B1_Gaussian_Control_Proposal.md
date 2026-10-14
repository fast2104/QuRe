# Đề xuất thực nghiệm B1 — Gaussian Control cho QuRe

## 1. Phạm vi

Thực nghiệm này được thực hiện trên repo [QuRe](.), không sử dụng
[QuRe_new_3Qformer](../QuRe_new_3Qformer) vì đó là một hướng thí nghiệm khác.

Mục tiêu của B1 là kiểm tra riêng câu hỏi:

> Chỉ thêm stochastic Gaussian noise vào query tokens có cải thiện hiệu năng
> Composed Image Retrieval hay không?

Nếu Gaussian noise đã giúp tăng hiệu năng, chưa thể kết luận rằng Prompt Flow
đã học được một representation tốt hơn. Vì vậy B1 phải được xem như một
control experiment cho Prompt Flow.

---

## 2. Trạng thái B0 hiện tại

B0 đã được train trong repo này với:

- Batch size: `8`.
- Loss: KL-divergence trên hai score positive/negative.
- Hard-negative sampling bằng cơ chế reranking.
- Dataset split và preprocessing hiện tại.
- Optimizer, learning rate, scheduler, số epoch và random seed hiện tại.

Các điều kiện trên phải được giữ nguyên khi chạy B1.

### 2.1. Loss hiện tại

Trong [train_qure.py](train_qure.py), model tạo hai score:

- `score_1`: score của target image đúng.
- `score_2`: score của negative image.

Sau đó code áp dụng:

```python
log_softmax_scores = F.log_softmax(scores, dim=1)

target_probs = torch.zeros_like(log_softmax_scores)
target_probs[:, 0] = 1.0

loss = F.kl_div(
    log_softmax_scores,
    target_probs,
    reduction="batchmean",
)
```

Target distribution là `[1, 0]`. Do đó loss yêu cầu score của target đúng cao
hơn score của negative.

B1 không được thay đổi loss này và không được thêm một KL regularization mới.

### 2.2. Hard-negative sampling hiện tại

Config hiện tại sử dụng cơ chế:

```json
"negative": "random_rerank",
"use_rerank": true,
"rerank_warmup": true
```

Luồng xử lý là:

```text
model hiện tại
    -> rerank toàn bộ target images
    -> xây hard-negative pool
    -> lấy negative từ hard-negative pool
    -> train bằng KL-divergence
```

Trong [data/fashionIQ.py](data/fashionIQ.py) và [data/cirr.py](data/cirr.py),
`rerank_score()` dùng query features và target features để xây danh sách
`hard_images`. Khi train, negative được lấy từ danh sách này.

B1 phải giữ cơ chế hard-negative sampling và tần suất cập nhật reranking như
B0, trừ khi chọn phương án dùng chung hard-negative pool được mô tả ở phần 6.

---

## 3. Thiết kế B1 đề xuất

B1 chỉ thêm một Gaussian residual vào query tokens:

```text
Q_new = Q_pre + gamma * epsilon
epsilon ~ N(0, I)
```

Trong đó:

- `Q_pre` là learned query tokens hiện tại của QuRe.
- `epsilon` là Gaussian noise có cùng shape với `Q_pre`.
- `gamma = 0.1` trong thí nghiệm đầu tiên.
- `num_samples = 1`.

Implementation conceptually:

```python
noise = torch.randn_like(query_tokens)
query_tokens = query_tokens + gamma * noise
```

B1 không được có:

- Linear layer học được để sinh residual.
- Decoder hoặc encoder mới.
- Planar flow hoặc normalizing flow.
- Image-conditioned generator.
- Prompt Flow parameter.
- KL regularization mới.
- Thay đổi loss.
- Thay đổi số lượng query tokens.

Như vậy B1 đo tác động của stochastic perturbation thuần túy, không đo khả
năng học của một module Prompt Flow.

---

## 4. Vị trí áp dụng noise

Noise nên chỉ được áp dụng ở reference/query branch:

```text
Reference/query branch:
    Q_ref = Q_pre + gamma * epsilon

Target/index branch:
    Q_target = Q_pre
```

Không nên thêm noise vào target/index branch. Target features tạo gallery dùng
cho retrieval; nếu gallery cũng bị nhiễu thì cả query và index đều thay đổi
ngẫu nhiên, làm kết quả khó ổn định và khó quy attribution cho riêng query
perturbation.

Vì vậy B1 nên được hiểu là:

```text
query representation bị perturb
target/index representation giữ nguyên
```

---

## 5. Ý nghĩa của `freeze_base_query_tokens`

Config:

```json
"freeze_base_query_tokens": true
```

chỉ nên freeze riêng `Q_pre`/`query_tokens`, không freeze toàn bộ model.

Các thành phần khác vẫn giữ cách train giống B0 nếu chúng đang có
`requires_grad=True`, ví dụ:

- Q-Former.
- Projection layers.
- Temperature nếu đang được train.
- Các parameter trainable khác của QuRe.

Khi đó B1 kiểm tra tác động của Gaussian perturbation trong cùng pipeline
training, trong khi query-token base được giữ cố định.

Lưu ý: vì noise không có parameter học được, B1 không học một representation
Gaussian. Model chỉ được train với query tokens có stochastic perturbation.
Đây là chủ đích của control experiment.

---

## 6. Hard-negative pool: hai phương án

Khi B1 có noise, query features dùng trong `rerank_score()` có thể khác B0.
Nếu B1 tạo lại hard-negative pool bằng query features có noise thì B1 đồng
thời thay đổi:

1. Query-token input.
2. Hard-negative set.

Điều này không làm B1 sai, nhưng làm việc kết luận nguyên nhân của
improvement kém chặt chẽ hơn.

### Phương án A — Giữ pipeline thực tế hiện tại

Mỗi run tự rerank bằng model tương ứng:

```text
B0 -> hard-negative pool riêng
B1 -> hard-negative pool riêng, có tác động của Gaussian noise
```

Ưu điểm:

- Giữ gần với pipeline train thực tế.
- Dễ triển khai hơn.

Nhược điểm:

- B1 không chỉ thay query tokens.
- Nếu B1 tốt hơn, chưa thể biết improvement đến từ noise hay hard-negative
  pool mới.

### Phương án B — Dùng chung hard-negative pool với B0

Đây là phương án được khuyến nghị nếu mục tiêu là ablation sạch:

```text
Tạo hard-negative pool bằng B0 một lần
    ├── dùng cho B0
    └── dùng lại cho B1
```

Khi đó khác biệt chính giữa B0 và B1 là Gaussian perturbation.

Nếu chọn phương án này, cần lưu pool theo từng sample/query và đảm bảo B0/B1
dùng cùng:

- Candidate hard negatives.
- Cách chọn negative.
- Random seed nếu việc lấy mẫu vẫn ngẫu nhiên.
- Tần suất hoặc lịch cập nhật tương ứng.

Nếu chưa triển khai cơ chế lưu và nạp hard-negative pool, có thể chạy theo
Phương án A, nhưng báo cáo phải ghi rõ rằng hard-negative pool của B1 được
rerank lại.

---

## 7. Noise trong training và evaluation

Có hai cách diễn giải khác nhau.

### 7.1. Noise chỉ trong training

```text
Train: Q_pre + gamma * epsilon
Eval:  Q_pre
```

Thí nghiệm này kiểm tra Gaussian noise như một dạng stochastic
regularization trong training.

Nếu B1 tốt hơn B0, kết luận phù hợp là:

> Gaussian perturbation trong quá trình train có thể giúp model học tốt hơn.

Không nên kết luận rằng query representation stochastic trực tiếp tốt hơn
trong inference.

### 7.2. Noise trong cả training và evaluation

```text
Train: Q_pre + gamma * epsilon
Eval:  Q_pre + gamma * epsilon
```

Thí nghiệm này kiểm tra tác động trực tiếp của stochastic query tokens trong
retrieval. Tuy nhiên evaluation sẽ không deterministic: hai lần chạy có thể
cho metric khác nhau do noise.

Nếu dùng cách này, cần một trong hai cách:

- Chạy nhiều evaluation seeds và báo cáo `mean ± standard deviation`.
- Dùng noise cố định theo từng query/sample để evaluation tái lập.

Không nên chỉ chạy một realization ngẫu nhiên rồi xem đó là kết quả cuối cùng
mà không ghi seed hoặc cách kiểm soát noise.

---

## 8. Config B1

Thiết kế theo file mô tả có thể biểu diễn bằng:

```json
"prompt_flow": {
    "enabled": true,
    "mode": "residual",
    "num_samples": 1,
    "gamma": 0.1,
    "freeze_base_query_tokens": true
}
```

Giữ `mode: "residual"` nếu đây là cấu hình được chỉ định trong kế hoạch
thực nghiệm. Về mặt diễn giải, B1 vẫn phải được ghi rõ là:

```text
Gaussian residual control
```

không phải Prompt Flow học được.

Các điều kiện bắt buộc ngoài config:

```text
batch_size = 8
loss = KL-divergence hiện tại
hard-negative sampling = giữ nguyên B0
num_samples = 1
gamma = 0.1
freeze riêng Q_pre
noise chỉ ở reference/query branch
target/index branch dùng Q_pre sạch
```

---

## 9. Các điều kiện phải giữ nguyên giữa B0 và B1

Để kết quả có thể so sánh:

- Dataset split.
- Preprocessing.
- Pretrained checkpoint.
- Batch size `8`.
- Optimizer.
- Learning rate.
- Scheduler.
- Weight decay.
- Số epoch.
- Random seed chính.
- Hard-negative sampling.
- Tần suất cập nhật hard-negative set.
- KL-divergence loss.
- Evaluation code.
- Metric.
- Số lượng query tokens.
- Temperature và cách sử dụng temperature.

Không nên đổi batch size của B1 sang giá trị mặc định khác trong config.

---

## 10. Cách diễn giải kết quả

### Trường hợp B1 tốt hơn B0

```text
B1 > B0
```

Kết luận được phép:

> Gaussian stochastic perturbation tự thân có thể đã đóng góp vào
> improvement.

Kết luận chưa được phép:

> Prompt Flow đã học được representation tốt hơn.

Để đưa ra kết luận về Prompt Flow, cần so sánh thêm B2 với B1.

### Trường hợp B1 tương đương hoặc kém B0

```text
B1 <= B0
```

Nếu B2 tốt hơn B1, bằng chứng sẽ thuyết phục hơn rằng improvement của B2
không chỉ đến từ noise ngẫu nhiên.

### Kết luận logic cho ba baseline

```text
B1 > B0:
    stochastic noise có lợi;
    chưa thể quy improvement cho Prompt Flow học được.

B1 ~= B0 hoặc B1 < B0, và B2 > B1:
    Prompt Flow có dấu hiệu đóng góp ngoài Gaussian noise.

B2 không tốt hơn B1:
    chưa có bằng chứng Prompt Flow học được lợi ích ngoài control.
```

---

## 11. Đề xuất quy trình chạy

### B0

- Sử dụng checkpoint/pretrained initialization giống run hiện tại.
- Batch size `8`.
- Giữ KL-divergence.
- Giữ hard-negative sampling.
- Nếu có thể, lưu hard-negative pool và lịch chọn negative.
- Lưu checkpoint, seed và metric đầy đủ.

### B1

- Bắt đầu từ cùng điều kiện với B0.
- Thêm đúng một Gaussian residual:

  ```text
  Q_ref = Q_pre + 0.1 * epsilon
  ```

- Dùng một sample mỗi forward.
- Freeze riêng `Q_pre`.
- Giữ target branch sạch.
- Giữ batch size `8`.
- Giữ loss và hard-negative pipeline.
- Ghi rõ noise được bật ở train, eval hay cả hai.
- Nếu hard-negative pool được tạo mới, ghi rõ điều này trong kết quả.

### Sau B1

Chỉ khi B0 và B1 đã có kết quả ổn định mới chạy B2 Prompt Flow. Khi đó B1
đóng vai trò control để đánh giá liệu Prompt Flow có tạo lợi ích ngoài
stochastic Gaussian noise hay không.

---

## 12. Tóm tắt quyết định

Thiết kế B1 phù hợp nhất với mục tiêu hiện tại:

```text
Repo:
    QuRe

Batch size:
    8

Loss:
    KL-divergence hiện tại của QuRe

Negative:
    Hard-negative sampling hiện tại

Query tokens:
    Q_ref = Q_pre + 0.1 * Gaussian noise

Target tokens:
    Q_target = Q_pre

Base query tokens:
    Freeze riêng Q_pre

Samples:
    1

Trainable Gaussian module:
    Không có

Prompt Flow/normalizing flow:
    Không có trong B1

Mục đích:
    Control để xác định stochastic noise có tự cải thiện performance hay không
```

Nếu ưu tiên kiểm soát nhân quả, dùng chung hard-negative pool với B0. Nếu ưu
tiên giữ nguyên pipeline train thực tế, cho phép B1 rerank lại hard negatives
nhưng phải ghi rõ hạn chế này khi phân tích kết quả.
