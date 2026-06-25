# Uncertainty-Aware Probabilistic PalmPrint Verification for Open-Set Attendance

Dự án nghiên cứu và phát triển hệ thống điểm danh bằng lòng bàn tay (**PalmPrint Attendance**) hướng tới bối cảnh mở (**Open-Set Biometric Verification**). Hệ thống không chỉ nhận diện chính xác danh tính các thành viên đã đăng ký (Closed-set), mà còn tối ưu hóa khả năng từ chối người dùng chưa hợp lệ (Unknown), giảm tỷ lệ nhận nhầm (**False Accept Rate - FAR**), và tự động ước lượng độ tin cậy của ảnh thông qua cơ chế phân phối không gian ẩn xác suất (**Probabilistic Latent Space**).

---

## 📌 Các Tính Năng Cốt Lõi (Key Features)

1. **Định Vị & Trích Chọn ROI Tự Động**: Tích hợp giải pháp **MediaPipe Hand Landmarker** để cắt vùng lòng bàn tay (ROI) ổn định dưới các góc chụp khác nhau.
2. **Biểu Diễn Không Gian Ẩn Xác Suất (Probabilistic Embedding)**: Mỗi ảnh không bị ép vào một vector cố định, mà được mô hình hóa bằng một phân phối Gaussian:
   $$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
   - $\boldsymbol{\mu}$: Đặc trưng định danh (Identity signal).
   - $\boldsymbol{\sigma}$: Độ bất định (Uncertainty) hoặc độ mơ hồ của mẫu ảnh đầu vào.
3. **Chiến Lược Học Tương Phản (Supervised Contrastive Learning)**: Tối ưu không gian đặc trưng bằng cách kéo gần các mẫu cùng danh tính và đẩy xa mẫu khác danh tính.
4. **Hydra Configuration**: Toàn bộ hệ thống được quản lý cấu hình bằng siêu khung `Hydra`, cho phép thay đổi cấu hình linh hoạt qua file YAML và tham số dòng lệnh mà không cần sửa code.
5. **Auto Logging & Versioning**: Tích hợp chặt chẽ với PyTorch Lightning, mọi kết quả huấn luyện, test, sinh ảnh, file checkpoint và ảnh preview sau mỗi Epoch đều được quản lý sạch sẽ và tự động trong thư mục `logs/version_X`.

---

## 📂 Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
PALM/
├── config/                  # Quản lý cấu hình bằng Hydra (cấu trúc module)
│   ├── dataset/             # Các kịch bản dataset (mnist, own_split_hand, own_split_ratio)
│   ├── model/               # Các kiến trúc model (unet, default, unet_mock)
│   ├── training/            # Cấu hình siêu tham số, optimizer, epochs
│   └── config.yaml          # File gốc điều phối toàn bộ cấu hình
├── data/                    # Nơi chứa ảnh gốc hoặc đã qua tiền xử lý
├── logs/                    # Thư mục hệ sinh thái Tracking
│   ├── version_0/           # Toàn bộ thông tin (config, checkpoints, test_results) của đợt chạy 0
│   ├── version_1/           # Đợt chạy 1...
│   └── unversioned_results/ # Nơi chứa kết quả các đoạn test chạy mù (không dùng model đã train)
├── src/                     # Mã nguồn lõi (Core Engine)
│   ├── datasets/            # Logic load ảnh, dataloader, sampler
│   ├── engine/              # LightningModule, các file core dùng để train/test
│   ├── losses/              # Các hàm mục tiêu (KL, Contrastive, Reconstruction)
│   ├── models/              # Kiến trúc mạng lưới (UNet, Backbone, Verifier)
│   └── processing/          # Script xử lý trích xuất điểm mốc bàn tay
├── tools/                   # Các công cụ tiện ích có thể chạy trực tiếp
│   ├── train_lightning.py   # Lệnh chính để huấn luyện mô hình
│   ├── test_pipeline.py     # Lệnh đánh giá mô hình (Metrics, Thresholding)
│   ├── generate_images.py   # Sinh/tái tạo ảnh từ mô hình (tạo biến thể, sample...)
│   ├── visualize_gradients.py # Trực quan hóa dòng chảy gradient của mạng
│   └── finding_represent.py # Mô phỏng tìm vector r chuẩn tối ưu cho Open-set (Test-Time Opt)
├── requirements.txt         # File cấu hình thư viện tinh gọn
└── README.md                # Tài liệu hướng dẫn bạn đang đọc
```

---

## 🚀 Hướng Dẫn Cài Đặt (Installation)

1. Cài đặt Python (khuyên dùng Python 3.9+).
2. Tải mã nguồn về và cài đặt các thư viện lõi:
```bash
pip install -r requirements.txt
```

---

## ⚡ Hướng Dẫn Sử Dụng (Usage)

Dự án sử dụng Hydra làm framework cấu hình chính. Cú pháp chạy chung là: 
`python tools/script.py [tham_số=giá_trị]`

### 1. Huấn Luyện Mô Hình (Training)
Lệnh mặc định sẽ tải toàn bộ cấu hình từ `config/config.yaml`.
```bash
python tools/train_lightning.py
```

Bạn có thể thay đổi ngay lập tức Dataset và Model mà không cần mở file:
```bash
# Huấn luyện trên OwnDataset (Chế độ chia theo Hand: tay trái train, tay phải val) với kiến trúc UNet
python tools/train_lightning.py dataset=own_split_hand model=unet

# Huấn luyện theo tỷ lệ random gộp tay (Ratio) với backbone dạng mock test
python tools/train_lightning.py dataset=own_split_ratio model=unet_mock
```

Khi chạy, hệ thống sẽ tự động sinh ra thư mục `logs/version_X`. Thư mục này chứa:
- `config_backup.yaml`: Lưu lại chính xác thông số bạn đã dùng để train.
- `checkpoints/best.ckpt` và `last.ckpt`: Các trọng số mô hình lưu tự động.
- `epoch_samples/`: Tự động sinh ảnh đối chiếu sau mỗi Epoch.
- Hệ sinh thái tracking của TensorBoard.

### 2. Sinh và Tái Tạo Ảnh (Image Generation)
Lệnh sinh ảnh có thể lấy trực tiếp checkpoint tương ứng trong `version_X`. Hệ thống sẽ xả kết quả ảnh sinh ra vào thẳng thư mục `logs/version_X/generated/`.

```bash
python tools/generate_images.py checkpoint="logs/version_0/checkpoints/best.ckpt" generation.mode="reconstruct"
```
*(Các chế độ sinh ảnh: `reconstruct`, `variations`, `contrastive`, `latent_sampling`)*

### 3. Đánh Giá Đường Ống (Test Pipeline)
Công cụ phân tích và đo lường hệ thống tự động, chạy ra các file logs log và `.csv` để đánh giá Threshold.
```bash
python tools/test_pipeline.py checkpoint="logs/version_0/checkpoints/best.ckpt"
```
Kết quả CSV và Text Logging sẽ nằm rải trong thư mục `logs/version_0/test_results/`.

### 4. Tìm vector đại diện tối ưu (TTO - Test Time Optimization)
Để test thuật toán tối ưu hóa vector đại diện trên tập mù:
```bash
python tools/finding_represent.py checkpoint="logs/version_0/checkpoints/best.ckpt" steps=100
```

### 5. Phân Tích Dòng Chảy Gradient
Xem cấu trúc mạng và luồng đi của gradient để debug hiện tượng mất mát (vanishing gradient):
```bash
python tools/visualize_gradients.py
```
Kết quả sơ đồ mạn nhện gradient sẽ xuất hiện dưới dạng `.png`.
