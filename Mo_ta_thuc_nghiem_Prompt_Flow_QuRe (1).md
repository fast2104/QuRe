# **MÔ TẢ THỰC NGHIỆM PROMPT FLOW CHO QuRe** 

## **1. Mục tiêu thực nghiệm** 

Triển khai một vòng thực nghiệm mới tập trung riêng vào Prompt Flow cho query tokens. Trong vòng này tạm thời không dùng LoRA, không thêm MoE và chưa thay loss của QuRe. 

Mục tiêu duy nhất là kiểm tra độc lập xem việc mô hình hóa query tokens bằng một phân phối học được có cải thiện hiệu năng Composed Image Retrieval (CIR) hay không. 

## **2. Thiết kế Prompt Flow** 

Baseline dùng QuRe gốc với query tokens hiện tại. Prompt Flow được bổ sung để sinh phần biến thiên cho query tokens. 

Lấy latent: 



Đưa qua Prompt Flow: 

ΔQ_φ = F_φ(z) 

Sinh query tokens mới theo dạng residual: 



Trong vòng thực nghiệm đầu tiên, đề nghị freeze Q_pre và chỉ train Prompt Flow để tách rõ đóng góp của module mới. 

## **3. Cấu hình đề xuất** 

Nhờ bổ sung config: 

"prompt_flow": { 

"enabled": true, "mode": "residual", "num_samples": 1, 

"gamma": 0.1, 

"freeze_base_query_tokens": true } 

Phiên bản đầu tiên chưa cần normalizing flow phức tạp. Có thể dùng một generator nhỏ Linear → GELU → Linear để kiểm tra principle trước. Nếu có tín hiệu tốt mới thay bằng normalizing flow thực sự. 

## **4. Các cấu hình cần chạy** 

### **B0 — QuRe baseline** 

- Không Prompt Flow. 

- Không LoRA. 

- Query tokens giữ nguyên như code hiện tại. 

- Giữ nguyên hard-negative sampling và QuRe preference loss. 

### **B1 — Gaussian control** 

Dùng: 



Mục đích: kiểm tra xem chỉ thêm stochastic noise có tạo improvement hay không. Nếu Gaussian noise đã tăng performance thì chưa thể kết luận Prompt Flow học được representation tốt hơn. 

**B2 — Prompt Flow, γ = 0.1** 







## **5. Các yếu tố phải giữ nguyên giữa các run** 

- Dataset split và preprocessing. 

- Pretrained checkpoint. 

- Optimizer, learning rate và scheduler. 

- Batch size và số epoch. 

- Random seed. 

- Hard-negative sampling và tần suất cập nhật hard-negative set. 

- QuRe preference loss. 

- Evaluation code và metric. 

Nguyên tắc: ở vòng này chỉ thay phần sinh query tokens; không thay đồng thời Prompt Flow và loss. 

## **6. Sampling trong vòng đầu** 

Vòng đầu chỉ dùng 1 sample query-token set cho mỗi forward. Chưa chạy 5 hoặc 10 samples và chưa dùng mean/weighted aggregation. 

Nếu Prompt Flow tốt hơn baseline thì mới chạy tiếp: 



Khi S > 1, aggregation đầu tiên nên dùng mean: 



## **7. Metric cần log** 

Ngoài Recall, cần log thêm độ đa dạng của query tokens và feature sau Q-Former. 

Query-token diversity: 



Feature diversity: 

D_Z = 1 - [2 / (S(S-1))] Σ_{i<j} cos(z_q^(i), z_q^(j)) 

Nếu D_Q hoặc D_Z gần 0, các samples gần như giống nhau và Prompt Flow có nguy cơ bị collapse. 

## **8. Bảng kết quả cần tổng hợp** 

|**Method**|**Dress**<br>**R@10**|**Dress**<br>**R@50**|**Shirt**<br>**R@10**|**Shirt**<br>**R@50**|**Toptee**<br>**R@10**|**Toptee**<br>**R@50**|**Avg**<br>**R@10**|**Avg**<br>**R@50**|**Avg**|
|---|---|---|---|---|---|---|---|---|---|
|QuRe<br>baseline||||||||||
|Gaussian<br>control||||||||||
|Prompt<br>Flowγ=0.1||||||||||
|Prompt<br>Flowγ=0.5||||||||||



