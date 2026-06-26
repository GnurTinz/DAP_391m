# Báo cáo Tổng quan Hệ thống: Palm Generative & Representation Framework

## 1. Tổng quan Kiến trúc
Dự án này là một framework Deep Learning toàn diện, được thiết kế chuyên biệt cho bài toán Phân tích, Rút trích đặc trưng (Representation / Metric Learning) và Sinh ảnh (Generative Modeling) cho dữ liệu bàn tay / vân tay.
Hệ thống được thiết kế theo hướng module hoá (Modular Design) sử dụng `Hydra` để quản lý config linh hoạt.

## 2. Các Thành phần Cốt lõi

### A. Xử lý Dữ liệu (Data Pipeline)
- **Tổ chức Module:** Nằm tại `src/datasets/`, quản lý tập trung thông qua `DatasetFactory` với lớp cha `BaseDataset`.
- **Hỗ trợ Đa Dataset:** Tích hợp sẵn các public datasets như `PalmPrintDataset` (PolyU), `TongjiDataset`, `IITDDataset`, và `MNISTDataset` (để dev/debug).
- **Custom Dataset:** Xây dựng `OwnDataset` và `OwnOriginalDataset` để tự động load cấu trúc dữ liệu tự thu thập từ `data/collect`. Có các tính năng mạnh mẽ thông qua YAML config:
  - Lọc dữ liệu theo tay trái/phải (`hand_filter`).
  - Chia tập train/val linh hoạt (`split_mode: ratio`, `train_ratio: 0.9`).
  - Tích hợp Transform/Augmentation tự động.

### B. Kiến trúc Mô hình (Models)
Hệ thống linh hoạt hỗ trợ hai trường phái chính:
1. **ProbabilisticPalmModel (VAE tiêu chuẩn):** Phù hợp để học phân bố liên tục và sinh ảnh vô điều kiện (unconditional generation).
2. **UNetPalmModel (Conditional U-Net):** Mô hình tái tạo mạnh mẽ với các *skip-connections*. Điểm sáng là việc sử dụng cơ chế **FiLM (Feature-wise Linear Modulation)** để tiêm thông tin từ không gian tiềm ẩn (Latent Vector Z) vào các nhánh upsampling (Decoder) thông qua việc điều chỉnh các tham số `gamma` và `beta`.

### C. Cơ chế Loss & Tối ưu hoá
- **Custom Losses (`src/losses/`):**
  - Khôi phục ảnh: `ReconstructionLoss` (L1/MSE).
  - Học phân bố: `KLDivLoss`.
  - Phạt bất định: `UncertaintyLoss` giúp tự động giữ cho phương sai (variance) không quá nhỏ (chống mất khả năng sinh ảnh) và không quá lớn (chống nhiễu).
- **Chiến lược huấn luyện nhiều giai đoạn (Loss Schedules):**
  - File `contrastive_first.yaml` cho thấy tư duy đào tạo xuất sắc: **(1)** Định hình không gian bằng Contrastive Learning trước $\rightarrow$ **(2)** Mở khóa dần Reconstruction $\rightarrow$ **(3)** Kéo phân bố lại bằng KL. 
- **Sampling:** Hỗ trợ `PK Sampler` để lấy mẫu có chủ đích, rất quan trọng để đảm bảo luôn có cặp Positive/Negative trong batch cho Contrastive Loss.

### D. Tiện ích Sinh ảnh (Image Generator)
Nằm tại `src/utils/generator.py` và `tools/generate_images.py`. Công cụ này cực kỳ mạnh mẽ, hỗ trợ hàng loạt cơ chế lấy mẫu từ Latent Space:
1. **Reconstruct:** Tái tạo ảnh gốc.
2. **Variations:** Lấy mẫu ngẫu nhiên quanh giá trị kỳ vọng ($\mu$) theo nhiệt độ (`temperature`) để tạo biến thể.
3. **Contrastive:** Trực quan hóa quan hệ Anchor - Positive - Negative.
4. **Average:** Lai tạo 2 ảnh bằng cách tính trung bình Latent và sinh các biến thể từ ảnh lai.
5. **Interpolate:** Nội suy tuyến tính (Linear Interpolation) tạo dải ảnh chuyển hóa mượt mà từ Ảnh 1 sang Ảnh 2.

## 3. Đánh giá nhanh
- **Tính học thuật cao:** Ứng dụng rất chuẩn các khái niệm như FiLM, Metric Learning schedule, PK Sampling, và Latent Space Interpolation.
- **Tính mở rộng:** Dễ dàng gắn thêm model mới hoặc dataset mới chỉ bằng việc thêm class và file `.yaml`.
- **Tiềm năng:** Mã nguồn hiện tại hoàn toàn sẵn sàng cho các thí nghiệm nghiên cứu sâu hơn về kiểm chứng sinh trắc học (Biometric Verification) hoặc sinh dữ liệu giả (Data Augmentation) cho vân tay/bàn tay.
