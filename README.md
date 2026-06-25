# Uncertainty-Aware Probabilistic PalmPrint Verification for Open-Set Attendance

Dự án nghiên cứu và phát triển hệ thống điểm danh bằng lòng bàn tay (**PalmPrint Attendance**) hướng tới bối cảnh mở (**Open-Set Biometric Verification**). Hệ thống không chỉ nhận diện chính xác danh tính các thành viên đã đăng ký (Closed-set), mà còn tối ưu hóa khả năng từ chối người dùng chưa hợp lệ (Unknown), giảm tỷ lệ nhận nhầm (**False Accept Rate - FAR**), và tự động ước lượng độ tin cậy của ảnh chụp thông qua cơ chế phân phối không gian ẩn xác suất (**Probabilistic Latent Space**).

---

## 📌 Các Tính Năng Cốt Lõi (Key Features)

1. **Định Vị & Trích Chọn ROI Tự Động**: Tích hợp giải pháp **MediaPipe Hand Landmarker** (`hand_landmarker.task`) để định vị các điểm mốc bàn tay chuẩn xác, căn chỉnh khung hình và cắt vùng lòng bàn tay (ROI) ổn định dưới các điều kiện góc chụp khác nhau.
2. **Biểu Diễn Không Gian Ẩn Xác Suất (Probabilistic Embedding)**: Mỗi ảnh lòng bàn tay không bị ép vào một vector cố định, mà được mô hình hóa bằng một phân phối Gaussian:
   $$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
   - $\boldsymbol{\mu}$: Đại diện cho đặc trưng định danh (Identity signal).
   - $\boldsymbol{\sigma}$: Biểu thị độ bất định (Uncertainty) hoặc độ mơ hồ của mẫu ảnh đầu vào (chất lượng ảnh kém, mờ, nhiễu).
3. **Chiến Lược Học Tương Phản Có Giám Sát (Supervised Contrastive Learning)**: Tối ưu không gian đặc trưng bằng cách kéo gần các mẫu cùng danh tính và đẩy xa các mẫu khác danh tính, đặc biệt tập trung khai thác các mẫu khó phân biệt (**Hard Negatives**).
4. **Cơ Chế Xác Thực Hai Tầng (Two-Stage Verification)**:
   - **Tầng 1 (Retrieval)**: Truy vấn nhanh Top-K ứng viên có khoảng cách embedding gần nhất trong Database.
   - **Tầng 2 (Verification & Reject)**: Sử dụng mạng MLP Verifier kết hợp thông tin đặc trưng $\boldsymbol{\mu}$ và độ bất định $\boldsymbol{\sigma}$ để đưa ra quyết định cuối cùng dựa trên Threshold (Ngưỡng điểm) + Margin (Khoảng cách an toàn) + Uncertainty Bound (Ngưỡng bất định tối đa).

---

## 📂 Cấu Trúc Thư Mục Dự Án (Project Structure)

Dự án được tổ chức chuẩn hóa, phân tách rõ ràng giữa cấu hình, mã nguồn xử lý, thực nghiệm và kiểm thử:

```text
PALM/
├── .vscode/                 # Cấu hình môi trường làm việc trên VS Code
├── config/                  # Quản lý cấu hình hệ thống và siêu tham số
│   ├── default.yaml         # File cấu hình chung cho Pipeline (Augmentation, Learning Rate,...)
│   └── hand_landmarker.task # Model pre-trained MediaPipe cho tác vụ Hand Landmarking
├── data/                    # Thư mục lưu trữ dữ liệu ảnh thô và dữ liệu đã tiền xử lý
├── implement-idea/          # Nơi lưu trữ tài liệu phân tích kiến trúc và sơ đồ pipeline
│   ├── Drawing...excalidraw # Bản vẽ thiết kế luồng hệ thống
│   └── pipeline-general.jpg # Sơ đồ tổng quan kiến trúc mô hình
├── logs/                    # Lưu vết quá trình huấn luyện và kiểm thử mô hình
├── notebooks/               # Thư mục chứa các file Jupyter Notebook để thực nghiệm nhanh
│   └── EDA.ipynb            # Phân tích khám phá dữ liệu (Exploratory Data Analysis)
├── papers/                  # Tài liệu tham khảo và các bài báo khoa học liên quan
├── src/                     # Mã nguồn chính của dự án (Source Code)
│   ├── processing/          # Chương trình nhanh cho việc tạo và xử lí dữ liệu
│   └── script/              # Tập hợp các kịch bản tạo dữ liệu huấn luyện
│       ├── __init__.py
│       ├── script1.py       
│       ├── script2.py       
│       ├── script3.py       
│       └── script4.py       
├── tests/                   # Bộ mã nguồn kiểm thử tự động (Unit Test)
├── .gitignore               # Chỉ định các tệp tin và thư mục không đẩy lên Git
├── README.md                # Tài liệu hướng dẫn dự án này
└── requirements.txt         # Danh sách thư viện và dependencies cần thiết
```

---

## ⚙️ Cấu Hình Hệ Thống (YAML Configuration)

Dự án sử dụng các file `.yaml` trong thư mục `config/` (VD: `default.yaml`, `mnist_unet.yaml`) để quản lý linh hoạt toàn bộ siêu tham số. Dưới đây là giải thích chi tiết các cấu hình khả dĩ:

### 1. `dataset` (Cấu hình dữ liệu)
- `data_dir`: Đường dẫn đến thư mục chứa dữ liệu gốc (VD: `data/MNIST` hoặc `data/Palmprint`).
- `image_size`: Kích thước ảnh đầu vào của mạng, ví dụ `[128, 128]` (Palmprint) hoặc `[32, 32]` (MNIST).
- `batch_size`: Kích thước batch huấn luyện thông thường.
- `num_workers`: Số lượng luồng xử lý song song để tải dữ liệu (Dataloader workers).

### 2. `sampler` (Cấu hình lấy mẫu dữ liệu)
- `type`: Chiến lược bốc dữ liệu (VD: `pk_sampler` đặc biệt tối ưu cho Contrastive Learning).
- `p`: Số lượng danh tính (Identities / Classes) trong một batch.
- `k`: Số lượng mẫu (Samples) của mỗi danh tính trong batch.
 *(Ví dụ: p=16, k=4 nghĩa là mỗi batch có 16 người, mỗi người 4 ảnh -> Tổng Batch Size = 64)*

### 3. `model` (Cấu hình kiến trúc mạng)
- `type`: Lõi kiến trúc (`default` cho Probabilistic VAE cơ bản, hoặc `unet` cho Probabilistic U-Net với Skip-connections).
- `encoder`:
  - `backbone`: Kiến trúc trích xuất đặc trưng (VD: `resnet18`, `resnet50`, hoặc `mock` để test nhẹ).
  - `latent_dim`: Số chiều không gian ẩn $\mathbf{z}$ (Ví dụ: `128`).
  - `pretrained`: Sử dụng trọng số ImageNet hay không (`true` / `false`).
- `decoder`:
  - `use_decoder`: Kích hoạt nhánh giải mã để tái tạo ảnh (Reconstruction) nhằm giữ chi tiết không gian cục bộ (`true` / `false`).
- `projector`:
  - `proj_dim`: Số chiều nén của nhánh Light MLP dùng riêng để tính Supervised Contrastive Loss (Ví dụ: `64`).

### 4. `losses` (Trọng số cân bằng hàm mất mát)
- `lambda_rec`: Trọng số của lỗi tái tạo ảnh (Reconstruction Loss).
- `lambda_con`: Trọng số ép cụm danh tính (Supervised Contrastive Loss).
- `beta_kl`: Trọng số ép không gian ẩn tuân theo phân phối chuẩn N(0, 1) (KL Divergence).
- `lambda_unc`: Trọng số kiểm soát độ bất định (Uncertainty Penalty) không quá lớn/nhỏ.

### 5. `training` (Cấu hình vòng đời huấn luyện)
- `epochs`: Tổng số vòng lặp huấn luyện.
- `learning_rate`: Tốc độ học của Optimizer (VD: `1e-3` hoặc `0.001`).
- `weight_decay`: Hệ số tiêu biến trọng số, chống Overfitting (VD: `1e-4`).
- `log_interval`: Chu kỳ in log ra Terminal và TensorBoard (tính theo số batch).
- `save_dir`: Thư mục tự động lưu lại các file trọng số `.pth` (Checkpoint).
