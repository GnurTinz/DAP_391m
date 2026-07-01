# Đặc điểm của FiLM (Feature-wise Linear Modulation)

FiLM (Feature-wise Linear Modulation) là một phương pháp điều kiện hóa (conditioning) linh hoạt và hiệu quả, thường được sử dụng trong các mạng nơ-ron (đặc biệt là CNNs, ResNets) để kết hợp thông tin từ một nguồn điều kiện (ví dụ: văn bản, nhãn lớp, hoặc bước thời gian timestep trong Diffusion Models) vào quá trình xử lý đặc trưng (features).

## Các đặc điểm chính

1. **Phép biến đổi Affine đơn giản:** FiLM áp dụng một phép biến đổi tuyến tính (co giãn - scale và dịch chuyển - shift) lên từng kênh (channel) của bản đồ đặc trưng (feature map).
   - Công thức: `FiLM(F_c) = γ_c * F_c + β_c`
   - Trong đó: 
     - `F_c` là đặc trưng đầu vào ở kênh thứ `c`. 
     - `γ_c` (gamma) và `β_c` (beta) là các tham số điều chế được dự đoán từ nguồn điều kiện thông qua một mạng nơ-ron nhỏ (Conditioning Network).

2. **Tính hiệu quả về mặt tính toán:** Mạng tạo ra `γ` và `β` (thường là các lớp MLP/Linear) rất nhỏ và tính toán nhanh. Phép tính nhân (`γ`) và cộng (`β`) (element-wise) lên feature map có chi phí tính toán không đáng kể so với các phép tích chập (convolution) hay attention.

3. **Mức độ tác động linh hoạt cao:** Nhờ điều chế độc lập trên từng kênh, FiLM có thể:
   - "Tắt" một kênh đặc trưng không cần thiết (đặt `γ = 0, β = 0`).
   - Giữ nguyên kênh đặc trưng (đặt `γ = 1, β = 0`).
   - Khuyếch đại hoặc đảo ngược hoàn toàn tín hiệu của kênh tùy thuộc vào điều kiện.

4. **Dễ dàng tích hợp (Plug-and-play):** Rất dễ chèn FiLM vào các cấu trúc mạng hiện có. Vị trí phổ biến nhất là đặt FiLM ngay sau lớp chuẩn hóa (Normalization, ví dụ GroupNorm) và trước lớp kích hoạt (Activation, ví dụ SiLU/ReLU) bên trong các khối ResNet.

## Sơ đồ hoạt động của FiLM

Sơ đồ dưới đây minh họa cách luồng dữ liệu chính bị can thiệp bởi luồng điều kiện (Conditioning stream).

```mermaid
graph TD
    subgraph Luồng Điều Kiện (Conditioning Stream)
        Cond[Conditioning Input<br>e.g., Timestep, Text] --> CondNet[Conditioning Network<br>Linear / MLP]
        CondNet -->|Tạo ra| Gamma("\gamma (Hệ số Scale)")
        CondNet -->|Tạo ra| Beta("\beta (Hệ số Shift)")
    end
    
    subgraph Luồng Dữ Liệu Chính (Main Feature Stream)
        Input[Input Features<br>F] --> Conv[Convolution Layer]
        Conv --> Norm[Normalization<br>e.g., GroupNorm]
        
        Norm --> Mul(( "×" ))
        Mul --> Add(( "+" ))
        
        Add --> Act[Activation Layer<br>e.g., SiLU]
        Act --> Output[Output Features<br>F']
    end
    
    %% Modulation Connections
    Gamma -->|Nhân element-wise| Mul
    Beta -->|Cộng element-wise| Add
    
    %% Styling
    classDef cond fill:#fde6eb,stroke:#e83e8c,stroke-width:2px;
    classDef main fill:#e0f7fa,stroke:#00bcd4,stroke-width:2px;
    classDef op fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,shape:circle;
    
    class Cond,CondNet,Gamma,Beta cond;
    class Input,Conv,Norm,Act,Output main;
    class Mul,Add op;
```
