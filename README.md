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
