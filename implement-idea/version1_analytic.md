# PalmPrint Recognition System - Version 1 Analytic

Tài liệu này mô tả chi tiết toàn bộ kiến trúc và các module của hệ thống nhận diện sinh trắc học PalmPrint dựa trên nguyên lý Generative Model (VAE) kết hợp với Test-Time Optimization (để trích xuất vector đặc trưng `r`), theo đúng thiết kế trong `pipeline-general.jpg`.

---

## 1. Cấu trúc Thư mục (Directory Structure)

Dự án được phân chia theo module hóa chuẩn của PyTorch:

- **`config/`**: Chứa các file YAML cấu hình tham số cho mô hình, quá trình huấn luyện và test.
- **`data/`**: Nơi lưu trữ dataset (PolyU, MNIST giả lập, v.v.).
- **`implement-idea/`**: Nơi chứa các hình ảnh concept, tài liệu phân tích ý tưởng (`pipeline-general.jpg`, `version1_analytic.md`).
- **`src/`**: Thư mục lõi chứa mã nguồn chính.
  - **`datasets/`**: Định nghĩa Dataset và Sampler để load ảnh.
  - **`engine/`**: Chứa các hàm cốt lõi cho Train/Test/Optimization. Đặc biệt là `represent.py` quản lý logic tìm `r`.
  - **`losses/`**: Các loss function tùy biến (Ví dụ: Contrastive Loss, KL Divergence).
  - **`models/`**: Cấu trúc mạng nơ-ron: `encoder.py`, `decoder.py`, `palm_model.py` (tích hợp VAE), và `verifier.py` (Global MLP Verifier).
- **`tests/`**: Các bộ Unittest độc lập cho từng module đảm bảo logic vững chắc.
- **`tools/`**: Các script điều khiển luồng chính (Train, Test, Inference).
  - `train_generative_model.py`: Script huấn luyện VAE Model.
  - `test_pipeline.py`: Script đánh giá End-to-End Pipeline (Build Database -> Inference Query).

---

## 2. Kiến trúc Hệ thống

### Giai đoạn 1: Training Generative Model (VAE + Contrastive)
- **Đầu vào**: Ảnh vân tay (Palmprint).
- **Encoder**: Đưa ảnh thành dạng phân phối xác suất Không gian ẩn (Latent Space) đặc trưng bởi trung bình $\mu$ (Mu) và phương sai $\sigma$ (Sigma).
- **Reparameterization Trick**: Lấy mẫu $z = \mu + \sigma \times \epsilon$ (với $\epsilon \sim \mathcal{N}(0, I)$).
- **Decoder**: Tái tạo lại ảnh từ vector $z$.
- **Projector (Light MLP)**: Xử lý $\mu$ qua một MLP nhỏ để học Contrastive Loss (Push/Pull) giúp tách biệt các ID khác nhau.
- **Loss Function**: Kết hợp Lỗi Tái tạo (Reconstruction) + Lỗi KL Divergence + Lỗi Contrastive (SupConLoss).

### Giai đoạn 2: Test-time Optimization (Finding Representation `r`)
Được thiết kế dựa trên quá trình **Finding r** của sơ đồ.
Thay vì dùng trực tiếp $\mu$ làm đặc trưng, hệ thống sẽ đi tìm một vector `r` đại diện duy nhất.
- **Hàm xử lý**: `optimize_r_from_latent()` nằm trong `src/engine/represent.py`.
- **Cách thức hoạt động**:
  1. Trích xuất $\mu_q, \sigma_q$ của ảnh bằng Encoder (frozen).
  2. Dùng $\mu_q, \sigma_q$ để tạo ra một tập hợp giả lập (Sample) $X_{new}$ chứa các vector Positive (nhãn 1) và sinh ra các vector Negative (nhãn 0) ngẫu nhiên.
  3. Vector `r` được khởi tạo bằng $\mu_q$.
  4. Đưa $X_{new}$ và $r$ vào một mạng nơ-ron **TestTimeVerifier** (MLP Verifier).
  5. Backpropagation để tối ưu và dịch chuyển `r` sao cho Verifier phân biệt được Positive và Negative tốt nhất (BCE Loss + L2 Penalty).

### Giai đoạn 3: Attend (So khớp và Xác thực)
Đây là quy trình diễn ra thực tế trong file `tools/test_pipeline.py`.

#### Bước 3.1: Build Database (Enrollment)
- Từng ảnh gốc (hoặc trung bình các ảnh) của mỗi Person được đưa qua Encoder lấy $\mu_c, \sigma_c$.
- Gọi quá trình **Finding r** để tìm `r_c` đại diện lưu vào database (Gallery).
- **Global Verifier Concept**:
  - Tại Person đầu tiên trong tập Database, mạng **TestTimeVerifier** sẽ được phép cập nhật trọng số (*freeze_net=False*).
  - Tại tất cả các Person từ thứ 2 trở đi, mạng Verifier này bị khóa (*freeze_net=True*), bộ tối ưu chỉ còn nhiệm vụ dịch chuyển `r_c`. 
  - Điều này giúp hệ thống học được một không gian Metric chung (Global Metric) từ người đầu tiên, và mọi đối tượng còn lại đều được chiếu về không gian đó!

#### Bước 3.2: Query Inference
- Đưa ảnh truy vấn qua Encoder lấy $\mu_q, \sigma_q$.
- **Tìm `r_q`**: Truyền mỏ neo mạng Global Verifier (đang ở trạng thái *freeze_net=True*) vào hàm tối ưu để đi tìm vector đại diện `r_q` cho ảnh truy vấn. Đồng thời trích xuất luôn bộ mẫu giả lập $X_{new}$ ($z_{pos}$).
- **Retrieval (Find Similar)**: Dùng khoảng cách L2 (Euclid) tính toán giữa `r_q` và các `r_c` trong Database để lọc ra **Top-K** ứng viên giống nhất.
- **Verification (Xác thực)**:
  - Lấy các mẫu $X_{new}$ của Query và `r_c` của ứng viên.
  - Tái sử dụng lại chính mạng **Global Verifier** đã train từ vòng Build Database, truyền cặp dữ liệu này vào để chấm điểm.
  - Kết quả ra một giá trị `score_prob`. Dựa vào Threshold và Margin mà đưa ra quyết định ACCEPT (chấp nhận) hay REJECT (từ chối).

---

## 3. Tổng kết
Kiến trúc này đã phân tách hoàn toàn Bài toán Nhận diện (Verification/Identification) ra khỏi việc phải học một mô hình Phân loại (Classification).
Hệ thống linh hoạt cực kỳ cao vì mạng **Verifier** có thể học ngay cả khi tập Gallery có biến động. Nó tuân thủ chặt chẽ theo sơ đồ `pipeline-general.jpg` đã thiết kế.
