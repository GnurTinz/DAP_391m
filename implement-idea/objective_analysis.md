# PHÂN TÍCH KHÁCH QUAN VỀ CÁCH TIẾP CẬN VÀ CHIẾN LƯỢC HUẤN LUYỆN (OBJECTIVE ANALYSIS)

Tài liệu này cung cấp một góc nhìn phân tích, đánh giá khách quan về thiết kế hệ thống, các quyết định cấu hình và các thí nghiệm (experiments) đang được triển khai trong dự án Palmprint Recognition.

---

## 1. PHÂN TÍCH CÁCH TIẾP CẬN (THE APPROACH)

Cách tiếp cận cốt lõi của dự án là một sự kết hợp độc đáo: **Sử dụng kiến trúc sinh (Generative - U-Net) làm nền tảng ép buộc để phục vụ bài toán Phân biệt (Discriminative - Metric Learning/ArcFace).**

*   **Tính đột phá:** Trong các bài toán sinh trắc học thông thường, người ta chỉ dùng CNN -> Linear -> ArcFace. Điểm yếu của cách truyền thống là mạng dễ bị "học vẹt" bối cảnh (như ánh sáng, phông nền, nếp nhăn tay không ổn định). Bằng cách chèn thêm một nhánh U-Net Decoder và tính hàm Loss khôi phục (Reconstruction Loss), dự án ép mạng phải duy trì "tính vật lý" của đường vân tay trong không gian Vector (Latent Space).
*   **Sự hợp lý của khối Projector:** Việc tách biệt `latent_dim` (không gian lưu cấu trúc vật lý) và `proj_dim` (không gian lưu danh tính bằng Projector) là một thiết kế giải quyết triệt để sự xung đột (gradient conflict) giữa hai hàm loss.
*   **Tính thực tiễn (Deploy):** Cực kỳ xuất sắc. Khi đem ra chạy thực tế (Inference), toàn bộ nhánh Decoder (rất nặng) được vứt bỏ, chỉ giữ lại Encoder -> Projector. Điều này có nghĩa là lúc train thì "khổ luyện", nhưng lúc chạy thì siêu nhẹ.

## 2. ĐÁNH GIÁ CÁC CẤU HÌNH KIẾN TRÚC (MODELS)

Dự án cung cấp một phổ (spectrum) cực rộng các backbone từ siêu nhẹ đến siêu nặng, phục vụ cho nhiều mục đích khác nhau:

*   **ResNet18 (`unet_resnet`) - The Heavyweight:**
    *   *Ưu điểm:* Tận dụng bộ trọng số ImageNet, năng lực biểu diễn không gian cực mạnh. Độ chính xác nhận diện chắc chắn sẽ đạt ngưỡng SOTA (State-of-The-Art). Chống chịu tốt với ảnh bị mờ, rung, lệch góc.
    *   *Nhược điểm:* Quá dư thừa tham số cho một bài toán vân tay vốn dĩ chỉ có đường nét đơn giản.
*   **PalmNet Gabor (`unet_palmnet_gabor`) - The Lightweight:**
    *   *Ưu điểm:* Một quyết định thiết kế thông minh. Dùng 3 lớp Conv nhưng mồi sẵn trọng số Gabor Filter. Gabor sinh ra là để trích xuất vân (texture). Việc này giúp mạng cực nhẹ, có thể chạy real-time trên Raspberry Pi.
    *   *Rủi ro:* Vì quá nông, Receptive Field (tầm nhìn) bị hạn chế. Rất dễ thất bại nếu ảnh đầu vào chưa được căn chỉnh (align/crop) chuẩn xác.
*   **CCNet (`unet_ccnet`) - The Experimental:**
    *   *Đánh giá:* Ý tưởng sử dụng Criss-Cross Attention để bắt cấu trúc toàn cục (Global Context) của lòng bàn tay rất thú vị về mặt học thuật. Tuy nhiên, vân tay thường mang tính cục bộ (local texture), do đó CCNet có thể gây tốn kém tính toán (RAM/VRAM) mà không mang lại sự đột phá về Accuracy.

## 3. CHIẾN LƯỢC HUẤN LUYỆN & HÀM LOSS (LOSS SCHEDULES & LOSSES)

*   **Hàm Loss ArcFace vs. Tái tạo:**
    *   Việc sử dụng **ArcFace** cho bài toán tập dữ liệu mở (Open-set) là tiêu chuẩn vàng hiện tại. Nó đẩy lùi các danh tính ra xa nhau trên một mặt cầu đa chiều.
    *   Sự kết hợp giữa ArcFace và Reconstruction Loss tạo ra thế gọng kìm: ArcFace chia cắt danh tính, Reconstruction giữ lại bản chất vật lý.
*   **Chiến lược `contrastive_first` (Huấn luyện hai giai đoạn):**
    *   *Cách làm:* Cho ArcFace chạy trước với Learning Rate (LR) cao để phân cụm thô. Sau đó hạ LR và mở khóa nhánh Decoder để tinh chỉnh bề mặt không gian vật lý.
    *   *Đánh giá khách quan:* Đây là một chiến thuật rất "nghệ thuật" và khó tinh chỉnh (heuristic). Rủi ro là nếu Giai đoạn 1 (Contrastive) làm hỏng không gian quá nặng, Giai đoạn 2 (Reconstruction) sẽ bị quá tải (Loss nổ tung). Do đó, cần cực kỳ cẩn thận với trọng số (weight) của từng hàm loss.

## 4. EXPERIMENTS (THÍ NGHIỆM) TRÊN DATASET

Dự án thiết kế các kịch bản kiểm thử (benchmark) rất nghiêm ngặt và toàn diện:

*   **Chia tỷ lệ (Ratio Split - IITD / Tongji_Mixed):** Kiểm tra khả năng ghi nhớ cơ bản. Kịch bản này thường sẽ cho kết quả > 98%, chủ yếu dùng để khẳng định pipeline hoạt động.
*   **Chia theo Tay Trái/Phải (Hand Split):** Đây mới là bài test tử thần. Việc train trên tay trái và bắt mạng phải suy diễn danh tính khi nhìn thấy tay phải đo lường khả năng tổng quát hóa tuyệt đối của mô hình.
*   **Chia theo Session (Tongji_Session):** Train trên lần thu thập 1, test trên lần thu thập 2 (cách nhau nhiều ngày). Kịch bản này kiểm tra khả năng chống "trôi dạt dữ liệu" (Data Drift) - ví dụ độ ẩm tay thay đổi, ánh sáng phòng đổi.

## 5. TỔNG KẾT VÀ KIẾN NGHỊ (RISKS & RECOMMENDATIONS)

**Điểm mạnh nhất:** Hệ thống module hóa (Hydra Config) cho phép tráo đổi nhanh chóng giữa "Backbone - Đầu máy" và "Decoder/Projector - Toa xe". Triết lý dùng Decoder làm Regularizer là một vũ khí hạng nặng.

**Điểm rủi ro (Rủi ro kỹ thuật cần chú ý):**
1.  **Cân bằng Loss:** `L_total = L_arcface + lambda * L_recon`. Việc tìm ra chỉ số `lambda` hoàn hảo là cực kỳ khó. Nếu `lambda` quá lớn, mạng sẽ biến thành Autoencoder thông thường (Accuracy thấp). Nếu quá nhỏ, Decoder trở nên vô dụng. Kiến nghị dùng công cụ Hyperparameter Optimization (như Optuna) hoặc Adaptive Loss Weighting (ví dụ: PCGrad).
2.  **Dataset Alignment:** Kiến trúc PalmNet Gabor phụ thuộc sống còn vào việc ảnh đầu vào phải được Crop (cắt) vùng ROI trung tâm chuẩn xác. Nếu dataset thu thập bị lệch ROI quá nhiều, PalmNet sẽ "chết" trước ResNet.

*Về tổng thể, framework này là một cơ sở dữ liệu vững chắc cho một bài báo khoa học (Paper) chuẩn mực mức Q1/Q2, vì nó vừa có phần thực nghiệm diện rộng (các kiến trúc), vừa có đóng góp về mặt lý thuyết (Decoder làm Regularizer cho bài toán Sinh trắc).*
