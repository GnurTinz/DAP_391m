# Data Science Project Report: Open-Set Palmprint Recognition

*Note: Theo yêu cầu của slide, báo cáo được thiết kế bằng tiếng Anh nhưng kèm giải thích ngắn gọn bằng tiếng Việt để bạn dễ dàng chuẩn bị nội dung.*

---

## PART 1 — Problem & Data Understanding (Steps 1-2-3)

### 1. Problem Understanding
*   **Context:** Palmprint recognition is a reliable biometric authentication method. However, traditional models often struggle in "open-set" scenarios (where test identities are unseen during training) because they tend to overfit to background or lighting rather than the actual physical ridge structure.
*   **Objective:** Develop a robust open-set palmprint verification system using a novel approach: Generative Regularization (U-Net) combined with Probabilistic Contrastive Learning (ArcFace).
*   *Tiếng Việt: Trình bày bài toán nhận dạng vân tay mở, nhược điểm của mô hình cũ và mục tiêu dùng mô hình sinh (Generative) làm điều chuẩn (Regularizer) để ép mạng học đúng cấu trúc vân tay.*

### 2. Data Understanding
*   **Datasets:** 
    *   **IITD Palmprint V1:** Collected under constrained environments (Left and Right hands).
    *   **Tongji Palmprint:** Collected in multiple sessions, providing cross-session challenges.
*   **Data Types:** Pre-processed ROI (Region of Interest) grayscale images (e.g., 128x128) representing the center of the palm.
*   *Tiếng Việt: Giới thiệu các bộ dữ liệu IITD và Tongji, đặc điểm ảnh đầu vào (ảnh ROI xám 128x128).*

### 3. Feature Understanding (EDA)
*   **Visual Characteristics:** Palmprints consist of principal lines, wrinkles, and ridges.
*   **Latent Distribution Hypothesis:** Standard CNNs map these to points, but palmprints have inherent ambiguity (blur, noise). Thus, mapping them to a probabilistic distribution $\mathcal{N}(\mu, \Sigma)$ is more appropriate.
*   *Tiếng Việt: Phân tích đặc trưng vân tay (đường nét, nếp nhăn). Đặt giả thuyết về việc dùng phân phối xác suất thay vì vector điểm thông thường.*

---

## PART 2 — Feature Engineering & Visualization (Steps 3-4)

### 1. Feature Engineering (Deep Representation Learning)
*   Instead of traditional manual feature engineering, we utilize an **Encoder-Decoder (U-Net)** architecture to automatically extract features.
*   **Latent Space ($\mu, \Sigma$):** Preserves physical structure via Reconstruction Loss.
*   **Projected Space:** Preserves identity separation via ArcFace Loss.
*   *Tiếng Việt: Giải thích cách dùng Deep Learning để trích xuất đặc trưng. Sự tách biệt giữa Không gian tiềm ẩn (chứa vật lý) và Không gian chiếu (chứa danh tính).*

### 2. EDA Advanced Charts (Latent Space Visualization)
*   **t-SNE / PCA Plots:** Visualizing the clustering of identities in the projected space vs latent space.
*   **Reconstruction Maps:** Showing original image vs. reconstructed image $\hat{X}$ to prove the model understands physical structures.
*   *Tiếng Việt: Đưa các biểu đồ PCA/t-SNE (từ file `pca_latent.py`) để chứng minh cụm danh tính. Đưa ảnh khôi phục từ Decoder.*

### 3. Interactive Dashboard (Optional)
*   Mention the development of a command-line interface or interactive script (`register.py`, `attendance.py`) for real-time inference and evaluation.
*   *Tiếng Việt: Giới thiệu hệ thống pipeline Điểm danh (`attendance.py`) và Đăng ký (`register.py`) như một AI App cơ bản.*

---

## PART 3 — Modeling, Evaluation & AI Application (Steps 5-6-7-8-9)

### 1. Dataset Partition
*   **Ratio Split:** Standard train/test split.
*   **Hand Split:** Train on left hands, test on right hands (Cross-domain challenge).
*   **Session Split:** Train on session 1, test on session 2 (Data drift challenge).
*   *Tiếng Việt: Trình bày các kịch bản chia data tử thần để test độ bền của mô hình.*

### 2. Data Modelling
*   **Backbone:** ResNet18, PalmNet (Gabor-initialized), CCNet.
*   **Loss Functions:** ArcFace (Identity separation) + MSE (Reconstruction) + KL Divergence (Probabilistic mapping).
*   **Novelty (Decoder-Loop Negative Sampling):** Generating hard negatives directly on the data manifold by perturbing the latent vector and passing it through the decoder.
*   *Tiếng Việt: Giới thiệu backbone, hàm loss tổng hợp và đặc biệt là kỹ thuật lấy mẫu âm bản bằng Decoder.*

### 3. Evaluation
*   **Metrics:** Equal Error Rate (EER), Rank-1 Accuracy, False Acceptance Rate (FAR), False Rejection Rate (FRR).
*   *Tiếng Việt: Đưa bảng kết quả đánh giá, so sánh giữa các chiến lược (ví dụ: có decoder vs không có decoder).*

### 4. Hyper-parameter Tuning
*   Tuning the balance weight ($\lambda$) between Contrastive Loss and Reconstruction Loss.
*   Tuning ArcFace margin and sampling temperature.
*   *Tiếng Việt: Giải thích cách tinh chỉnh trọng số giữa các hàm loss và nhiệt độ lấy mẫu.*

### 5. Pipeline & AI App
*   **Training Pipeline:** Contrastive-first curriculum learning.
*   **Inference App (Attendance System):** Discard the heavy decoder; only use Encoder + Projector for lightweight, real-time identity matching.
*   *Tiếng Việt: Trình bày pipeline thực tế: lúc train thì nặng (kèm Decoder), lúc chạy app điểm danh thì nhẹ (chỉ Encoder).*

---

## PART 4 — Conclusion & AI Reflection (Step 10 + Q&A)

### 1. Conclusion
*   Generative regularization significantly improves discriminative models in open-set biometrics.
*   Decoder-driven negative sampling successfully mines valid, hard negatives, boosting the verification margin.
*   *Tiếng Việt: Kết luận về sự thành công của phương pháp điều chuẩn sinh và lấy mẫu qua decoder.*

### 2. AI Audit Log
*   Experiment tracking using PyTorch Lightning and TensorBoard/Wandb to monitor validation metrics across diverse splits.
*   *Tiếng Việt: Theo dõi lịch sử huấn luyện qua Lightning logs.*

### 3. Human Delta (Sự can thiệp của con người)
*   The transition from a standard classifier to an Encoder-Decoder architecture required strong human intuition regarding the biological structure of palmprints.
*   *Tiếng Việt: Giá trị con người nằm ở việc thiết kế kiến trúc ép AI phải nhớ hình học vật lý của vân tay thay vì học vẹt.*

### 4. Hallucination Detection (Uncertainty Rejection)
*   Using the latent variance ($\Sigma$) to detect "uncertain" inputs. If an image is too blurry or an imposter attacks the system, the high uncertainty score allows the system to gracefully reject it (Open-set rejection).
*   *Tiếng Việt: Cách dùng logvar để đánh giá độ "bất định", từ đó từ chối người lạ hoặc ảnh nhiễu, chống nhận diện sai.*

### 5. Self-Assessment (Hạn chế & Hướng phát triển)
*   **Limitation:** The lightweight models (e.g., PalmNet) heavily rely on perfect ROI alignment. Unaligned palms may degrade performance.
*   **Future work:** Integrating Spatial Transformer Networks (STN) for auto-alignment before the encoder.
*   *Tiếng Việt: Đánh giá điểm yếu (phụ thuộc vào khâu crop ảnh) và hướng giải quyết tương lai.*
