# BÁO CÁO TRIỂN KHAI THỰC NGHIỆM B2: PROMPT FLOW CHO QuRe

Tài liệu này tổng hợp toàn bộ cơ sở lý thuyết, kiến trúc module, các thay đổi mã nguồn và giải thích chi tiết về việc chuyển đổi query token trong thực nghiệm **B2 (Prompt Flow, $\gamma = 0.1$)** theo tài liệu [Mo_ta_thuc_nghiem_Prompt_Flow_QuRe (1).md](file:///c:/code/QuRe/Mo_ta_thuc_nghiem_Prompt_Flow_QuRe%20%281%29.md) và tham khảo từ bài báo **Bayes-PFL** (CVPR 2025).

---

## 1. Trả lời câu hỏi về kích thước Vector Query Token

* **Kích thước ban đầu:** Chính xác là **$32 \times 768$** (với $32$ là số lượng query tokens của Q-Former trong kiến trúc BLIP-2, và $768$ là chiều ẩn `hidden_size` của BERT). Khi đi vào mô hình với batch size $B$, kích thước mở rộng là `(B, 32, 768)`.
* **Kích thước sau khi đi qua Prompt Flow Module:** **VẪN GIỮ NGUYÊN HOÀN TOÀN LÀ $32 \times 768$ (hoặc `(B, 32, 768)`)**.
* **Bản chất của sự thay đổi:**
  * **Kích thước hình học không đổi:** Q-Former vẫn nhận đúng $32$ token, mỗi token chiều $768$, đảm bảo tương thích $100\%$ với các khối Self-Attention và Cross-Attention hiện có.
  * **Phân phối biểu diễn (Distribution) thay đổi hoàn toàn:**
    * **B0 (Baseline):** Query tokens là một ma trận trọng số tĩnh, tất cả các mẫu trong batch đều dùng chung một bộ $32$ vector cố định:
      $$Q = Q_{pre}$$
    * **B1 (Gaussian Control):** Query tokens bị rung lắc bởi nhiễu Gauss đẳng hướng ngẫu nhiên:
      $$Q = Q_{pre} + \gamma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
    * **B2 (Prompt Flow từ Bayes-PFL):** Biến thiên $\Delta Q_\phi$ được sinh ra từ một **Normalizing Flow** (chuỗi biến đổi Planar Flow). Do Normalizing Flow là một phép biến đổi song ánh khả vi ($f: \mathbb{R}^{768} \to \mathbb{R}^{768}$), nó uốn nắn phân phối Gaussian cơ bản ban đầu thành một phân phối học được **rất phức tạp, phi tuyến và có thể có nhiều cực trị (multi-modal)**:
      $$Q_{new} = Q_{pre} + \gamma \cdot \Delta Q_\phi$$

---

## 2. Thiết kế chi tiết của Prompt Flow Module (Trích xuất từ Bayes-PFL)

Module được triển khai tại file: [lavis/models/blip2_models/prompt_flow_module.py](file:///c:/code/QuRe/lavis/models/blip2_models/prompt_flow_module.py)

### 2.1. Các thành phần chính

1. **`InferenceBlock` (từ Bayes-PFL):**
   * Mạng nơ-ron đa tầng: `Linear(768, 384) -> Softplus() -> Linear(384, 384) -> Softplus() -> Linear(384, 768)`.
   * Sử dụng hàm kích hoạt mềm `Softplus` giúp gradient ổn định và liên tục.
2. **`Encoder`:**
   * Gồm 2 nhánh `InferenceBlock` dự đoán trung bình $\mu(Q_{pre})$ và phương sai $\log \sigma^2(Q_{pre})$ cho phân phối cơ sở Gaussian $q(z_0 \mid Q_{pre})$.
3. **`Planar` (Planar Normalizing Flow từ Bayes-PFL `models/flows.py`):**
   * Thực hiện phép biến đổi không gian:
     $$z_{k+1} = z_k + u_k \tanh(w_k^T z_k + b_k)$$
   * Kèm tính toán log determinant của ma trận Jacobian:
     $$\psi = w_k (1 - \tanh^2(w_k^T z_k + b_k))$$
     $$\log |\det J| = \log |1 + \psi^T u_k|$$
4. **Tham số dòng Amortized ($u, w, b$):**
   * Sinh trực tiếp từ $Q_{pre}$ qua các tầng chiếu tuyến tính (`amor_u`, `amor_w`, `amor_b`), giúp mỗi query token có bộ tham số dòng riêng phù hợp với ngữ nghĩa của nó.
5. **`Decoder`:**
   * Ánh xạ latent sau dòng $z_K$ thành phần biến thiên $\Delta Q_\phi \in \mathbb{R}^{B \times 32 \times 768}$.
6. **Residual Injection:**
   $$Q_{new} = Q_{pre} + \gamma \cdot \Delta Q_\phi \quad (\gamma = 0.1)$$

### 2.2. Cơ chế Train vs. Eval
* **Khi Train:** Lấy mẫu ngẫu nhiên thông qua reparameterization trick $z_0 = \mu + \sigma \odot \epsilon$ ($\epsilon \sim \mathcal{N}(0, I)$) để khám phá không gian phân phối.
* **Khi Eval:** 
  * Chế độ mặc định `eval_mode = "mean"`: Sử dụng kỳ vọng $z_0 = \mu$ để triệt tiêu phương sai ngẫu nhiên trong đánh giá Recall, đảm bảo kết quả đánh giá ổn định và nhất quán giữa các truy vấn và ứng viên target.
  * Hỗ trợ thêm chế độ `fixed` (lấy mẫu 1 lần và cache như Bayes-PFL) hoặc `sample`.

---

## 3. Danh sách các file đã chỉnh sửa & Nội dung thay đổi

### 3.1. [lavis/models/blip2_models/prompt_flow_module.py](file:///c:/code/QuRe/lavis/models/blip2_models/prompt_flow_module.py)
* **Tác vụ:** Tạo mới và hoàn thiện module.
* **Nội dung:** 
  * Cung cấp các lớp `InferenceBlock`, `Encoder`, `Decoder`, `Planar`, `PromptFlow`.
  * Lưu giữ các lớp `PlanarPFL`, `PlanarPFL_state`, `PFL` của bài Bayes-PFL gốc để tham chiếu.
  * Bổ sung 2 hàm tính độ đa dạng theo Mục 7 của mô tả thực nghiệm:
    * `compute_query_token_diversity(query_tokens)`: Tính $D_Q$.
    * `compute_feature_diversity(features)`: Tính $D_Z$.

### 3.2. [lavis/models/blip2_models/blip2_qure.py](file:///c:/code/QuRe/lavis/models/blip2_models/blip2_qure.py)
* **Import:** Import `PromptFlow`, `compute_query_token_diversity`, `compute_feature_diversity`.
* **Khởi tạo:** Thêm cờ `self.prompt_flow_enabled`, `self.prompt_flow = None`, và các biến theo dõi độ đa dạng `last_query_diversity`, `last_feature_diversity`.
* **Cấu hình:** Viết hàm `configure_prompt_flow(config)`:
  * Đọc config `prompt_flow` với `"mode": "residual"`, `"gamma": 0.1`, `"freeze_base_query_tokens": true`.
  * Khởi tạo `PromptFlow` module trên cùng device của model.
  * Đóng băng `self.query_tokens.requires_grad_(False)`.
  * Tương thích ngược với hàm `configure_gaussian_control(config)`.
* **Sinh Query Tokens:** Cập nhật hàm `_expand_query_tokens(batch_size)`:
  * Nếu B2 bật: Gọi `self.prompt_flow(self.query_tokens, batch_size=batch_size)` để sinh $Q_{new}$.
  * Nếu B1 bật: Giữ logic thêm Gaussian noise cũ.
  * Nếu B0: Trả về $Q_{pre}$ mở rộng theo batch.
* **Đồng bộ hóa trong trích xuất Target Features:**
  * Tại `extract_target_features` và `extract_target_image_features_fiq`, thay thế việc gọi `self.query_tokens.expand(...)` bằng `self._expand_query_tokens(...)`. Điều này đảm bảo trong B2, đặc trưng của hình ảnh target và reference query nằm trong cùng không gian phân phối học được.
* **Ghi nhận Feature Diversity:** Trong hàm `forward()`, tự động tính $D_Z$ của `projected_query` khi đang train.

### 3.3. [models/\_\_init\_\_.py](file:///c:/code/QuRe/models/__init__.py)
* **Tác vụ:** Cập nhật hàm `create_qure_models`:
* **Nội dung:** Gọi `model.configure_prompt_flow(configs)` nếu model hỗ trợ, sau đó mới fallback về `configure_gaussian_control`.

### 3.4. [train_qure.py](file:///c:/code/QuRe/train_qure.py)
* **Tác vụ:** Logging & Bảo toàn điều kiện thực nghiệm.
* **Nội dung:**
  * Giữ nguyên $100\%$ Preference Loss, Optimizer (`SGD`), LR schedule, warmup, batch size, epochs theo Mục 5.
  * Bổ sung in ra màn hình và log lên WandB hai độ đo theo Mục 7:
    * `diversity_q` ($D_Q$)
    * `diversity_z` ($D_Z$)

### 3.5. [configs/fashionIQ/train.json](file:///c:/code/QuRe/configs/fashionIQ/train.json) & [configs/cirr/train.json](file:///c:/code/QuRe/configs/cirr/train.json)
* Đã cấu hình đúng định dạng đề xuất của B2:
```json
"prompt_flow": {
    "enabled": true,
    "mode": "residual",
    "num_samples": 1,
    "gamma": 0.1,
    "freeze_base_query_tokens": true
}
```

---

## 4. Kết quả kiểm tra (Unit Test)

Đã chạy kiểm tra tự động trên môi trường:
1. `q_new` có kích thước chính xác là `(B, 32, 768)`.
2. `base_query_tokens.grad is None`: Query token ban đầu được đóng băng hoàn toàn.
3. Tất cả các tham số của `PromptFlow` (Encoder, Amortized layers, Planar layers, Decoder) nhận đầy đủ gradient và sẵn sàng được tối ưu bằng SGD của `train_qure.py`.
4. Các hàm tính toán độ đa dạng $D_Q$ và $D_Z$ chạy mượt mà, không gây crash hay tràn bộ nhớ.
