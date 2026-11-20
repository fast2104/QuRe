# QuRe Exp1 · Gaussian query tokens qua prompt flow

Chạy [notebook Kaggle](Exp1_Gaussian_Query_PFM_Kaggle.ipynb) với GPU, Internet và FashionIQ đã giải nén. Notebook chứa source của biến thể và lấy các file Exp1 dùng chung từ commit đã ghim. Đọc [GaussianQueryFlow](pfm_variants.py) để xem phân phối đầu vào và phép biến đổi từng token.

Mỗi query có 10 mẫu. Mỗi mẫu là một bộ 32 vector 768 chiều. Caption điều kiện hóa Gaussian và planar flow thông qua embedding token của Q-Former. Reference image và caption cùng vào một Q-Former cho từng bộ token. Nhánh target dùng query token pretrained. Trainer tính contrastive loss trên điểm trung bình của các mẫu. Mỗi điểm mẫu là dot product lớn nhất trên 32 token của target.

Đặt `NUM_FLOWS = 0` để chạy đối chứng Gaussian hoặc `NUM_FLOWS = 10` để bật flow. Giữ `NUM_SAMPLES = 10`, batch size, seed và epoch giống nhau. Trong `history.json`, so sánh `val_pre_flow_spread`, `val_post_flow_spread`, `val_feature_spread` và FashionIQ recall. Hai chỉ số đầu có cùng đơn vị. Chỉ số thứ ba đo độ phân tán giữa các feature retrieval đã chuẩn hóa của cùng một truy vấn. Khi eval, các mẫu dùng nhiễu cố định và dropout tắt.
