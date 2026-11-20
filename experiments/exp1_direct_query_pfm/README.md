# QuRe Exp1 · PFM sinh trực tiếp query token reference

Mở [notebook Kaggle](Exp1_PFM_05_Direct_Query_Generation_Kaggle.ipynb), gắn FashionIQ đã giải nén, bật GPU và Internet, rồi chọn **Run All**. Notebook tải checkpoint BLIP-2 ViT-L hoặc dùng bản đã gắn, chuẩn bị backbone, kiểm tra gradient trên dữ liệu thật rồi bắt đầu train. Có thể đặt đường dẫn trong `DATA_ROOT` và `CHECKPOINT` ở cell đầu.

[DirectPromptFlow và PromptFlowQure](pfm_variants.py) sinh 32 × 768 query token từ riêng ảnh reference. Caption được tokenize riêng. Q-Former đọc query token mới, caption và toàn bộ visual memory. [PositiveQure](model.py) mã hóa ảnh target bằng query token pretrained và dùng chung trọng số Q-Former với nhánh reference.

[Trainer](train_pfm.py) chạy contrastive loss với các target trong batch và KL của flow. [Cấu hình](config.json) dùng seed 1000, batch 16, 30 epoch, latent 64 chiều, 10 planar flow và hệ số KL tăng đến 0,001 trong 5 epoch. Notebook ghi checkpoint, cấu hình thực tế và kết quả từng epoch vào thư mục output.

Code Exp1 và trình đọc FashionIQ lấy từ [Qure-Lora ở b36607d](https://github.com/MaiLinhLinh/Qure-Lora/tree/b36607df9d7813167df9cf2ef1d1de605938f1f2/Code_KT). Backbone dùng [QuRe ở c1c7f6b](https://github.com/fast2104/QuRe/tree/c1c7f6b62fe6d556b9554b6c7cae111f71f67917). Notebook cố định phiên bản source qua `RUNNER_SHA`. Khi thử một commit mới, cập nhật giá trị đó và dùng `RUN_TAG` mới.
