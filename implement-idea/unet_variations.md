# Các Biến Tấu (Variations) Của Mạng Probabilistic U-Net 

Tài liệu này ghi chú lại cách kiến trúc U-Net truyền thống đã được biến tấu, tuỳ chỉnh để phù hợp với bài toán của chúng ta (Contrastive Learning kết hợp Generative Model).

## Sơ đồ Tổng quan (Architecture Flow)

```mermaid
graph TD
    %% Inputs
    X[Input Image x]
    
    %% Encoders
    subgraph Encoders
        UE[U-Net Encoder]
        LE[Latent Encoder / PalmEncoder]
    end
    
    X --> UE
    X --> LE
    
    %% Latent Space
    subgraph Latent Space
        LE --> Mu["mu (μ)"]
        LE --> Logvar["logvar (log σ²)"]
        Mu & Logvar -->|Reparameterization| Z["Latent Vector (z)"]
    end
    
    %% Contrastive Branch
    subgraph Contrastive Branch
        Mu -->|Projector MLP / Identity| P[Projected Vector]
        P -->|Push / Pull| LossC((Contrastive Loss))
    end
    
    %% Decoder & Modulation
    subgraph Decoding Process
        UE -->|Skip Connections| FiLM["FiLM Layers (γ, β)"]
        Z -->|Modulation Signal| FiLM
        Z --> UD[U-Net Decoder]
        FiLM -->|Modulated Skips| UD
        UD --> Y[Reconstructed Image x']
    end
    
    %% Reconstruction Loss
    Y -->|Compare with Original| LossR((Reconstruction Loss))
    X -.-> LossR
```

## 1. Tách Biệt Bottleneck và Latent Encoder (Thay Giả Bằng Identity)
* **Vấn đề của U-Net truyền thống:** U-Net gốc đi từ ảnh $\to$ thu nhỏ liên tục (Encoder) $\to$ Đáy mạng (Bottleneck) $\to$ phóng to lên (Decoder).
* **Biến tấu:** 
  * Thay vì sử dụng chính phần đáy (bottleneck) của U-Net làm không gian biểu diễn, mạng của chúng ta dùng một `Latent Encoder` **độc lập hoàn toàn** (`PalmEncoder`) để trích xuất ra $\mu$ và $\sigma$.
  * Bottleneck gốc của U-Net (`down3` output) bị vô hiệu hoá hoặc "thay bằng identity" để chừa chỗ cho việc giải mã từ một vector tiềm ẩn ngẫu nhiên $z$.
  * Điều này cho phép không gian latent tập trung hoàn toàn vào việc biểu diễn đặc trưng (như ID người dùng) mà không bị vướng bận việc phải lưu giữ cả những feature rác (như background).

## 2. Lớp Projector Đa Năng Cho Contrastive Loss
Để bắt không gian $\mu$ học được cấu trúc phân cụm (cùng người thì gần nhau, khác người thì xa nhau), $\mu$ được cho đi qua một Projector (phục vụ Loss Push/Pull).

* **Biến tấu Cấu Hình:** Cấu trúc Projector không bị fix cứng, mà được điều khiển thông qua file `yaml` một cách linh hoạt:
  * Khả năng sử dụng/không sử dụng MLP (`use_mlp: True/False`). Khi set False, lớp Projector sẽ tự biến thành `nn.Identity()`.
  * Khả năng tuỳ chỉnh độ sâu và số node của từng lớp bằng mảng `hidden_dims` (vd: `[256, 128]`).
  * Khả năng tuỳ chỉnh **hàm kích hoạt (Activation)** tự động khởi tạo bằng `getattr(nn, act_name)`. Có thể dễ dàng đổi từ `ReLU` sang `GELU`, `SiLU`, hoặc `LeakyReLU` chỉ bằng việc sửa chữ trong yaml.

## 3. Điều Chế Skip Connections Bằng FiLM Layers
* **Vấn đề:** Khi tái tạo ảnh từ vector $z$ ở đáy U-Net, $z$ có thể mất mát những chi tiết cấu trúc hạt/vân tay cục bộ. U-Net giải quyết việc này bằng Skip Connections.
* **Biến tấu:** 
  * Thay vì chỉ đơn thuần là `torch.cat` (nối feature map) hoặc cộng feature map từ Encoder sang Decoder, vector không gian tiềm ẩn $z$ của chúng ta **can thiệp** trực tiếp vào các skip connections.
  * Thông qua kỹ thuật **FiLM** (Feature-wise Linear Modulation), vector $z$ sinh ra 2 ma trận $\gamma$ (Scale) và $\beta$ (Shift).
  * Feature map đi qua skip-connection sẽ được biến đổi: `x_modulated = (1 + gamma) * x + beta`.
  * Nhờ vậy, $z$ không chỉ định hướng toàn cục cho Decoder mà còn định hướng cho từng mảng chi tiết từ Encoder truyền qua.

## 4. Cơ Chế Lấy Mẫu (Sampling Modes) Đa Dạng
Việc tái tham số hoá (Reparameterization Trick) $\mu + \sigma \times \epsilon$ cũng được mở rộng để phục vụ nhiều luồng Inference:
* `stochastic`: Trạng thái huấn luyện VAE mặc định, cho phép khám phá (explore) không gian cục bộ. Có thể truyền thêm hệ số `temperature` để kiểm soát bán kính dao động.
* `deterministic`: Ép $\sigma \times \epsilon = 0 \to z = \mu$. Luôn sinh ra $z$ ổn định nhất (phục vụ cho tính toán độ đo khoảng cách, evaluate gallery/probe).
* `symmetric`: Trả về vector đối xứng qua tâm của $\mu$ (cụ thể: $-\mu + \sigma \times \epsilon$). Một variation thú vị cho các bài toán phân tích tính phản xạ của không gian tiềm ẩn.
* Công tắc `decode=True/False`: Giúp ngắt mạch không cho Decoder chạy nhằm tiết kiệm tài nguyên bộ nhớ khi chỉ cần lấy $z$ hoặc $\mu$ (Ví dụ như lúc matching).

## 5. Tương Thích Gradient Liền Mạch (Gradient Flow)
Mặc dù có nhiều nhánh tách biệt (Latent Encoder $\to \mu, \sigma$, U-Net Encoder $\to$ FiLM $\to$ Decoder, và Projector), nhưng kiến trúc bảo đảm **đạo hàm (gradient) chảy ngược liên tục** qua tất cả các khối kể cả các khối phụ như FiLM hay MLP Projector mà không bị đứt gãy. Đảm bảo huấn luyện Loss tổng ($Loss_{Recon} + Loss_{Contrastive} + Loss_{KL}$) có thể tối ưu hoá tất cả các trọng số đồng thời.
