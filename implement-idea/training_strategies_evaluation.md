# ĐÁNH GIÁ CÁC CHIẾN LƯỢC HUẤN LUYỆN (TRAINING STRATEGIES)

Tài liệu này tổng hợp và đánh giá chuyên sâu các chuỗi lệnh train (combinations) được quy định trong file `commands.txt`, dựa trên triết lý **"Mục tiêu là Điểm Danh, U-Net Decoder chỉ là công cụ chuẩn hóa (Regularizer)"**.

---

## 1. BỘ TỨ SOTA (State-of-The-Art) - 👑 MẠNH NHẤT
**Chuỗi cấu hình:** `model=unet_resnet` + `losses=arcface` + `loss_schedules=contrastive_first` + `training=arcface` + `dataset=own_original`
- **Backbone (ResNet18):** Mạng cực sâu, trích xuất đặc trưng không gian đa mức độ siêu việt nhờ trọng số Pretrained trên ImageNet. Chống chịu cực tốt với nhiễu (xoay, lệch sáng, mờ).
- **Decoder Regularization:** ResNet18 rất dễ bị overfit nếu chỉ dùng cho phân loại. Khi gắn Decoder vào, nó bị ép phải giữ lại bản đồ không gian vân tay, giúp Vector 128D trở nên chân thực và chứa đầy đủ cấu trúc sinh trắc học.
- **ArcFace & Contrastive:** Sắp xếp lại vector 128D của ResNet18 sao cho các góc độ giữa các người khác nhau bị đẩy ra xa tối đa.
- **Đánh giá:** 
  - **Ưu điểm:** Kết quả độ chính xác Rank-1 chắc chắn sẽ cao nhất. Biểu diễn (Latent Space) đẹp nhất, điểm Silhouette lúc vẽ t-SNE sẽ tiệm cận 1.
  - **Nhược điểm:** Tốn dung lượng bộ nhớ lớn, inference chậm hơn một chút so với CNN nông, không tối ưu cho thiết bị nhúng quá yếu.

---

## 2. EDGE-AI LITE (NHẸ & NHANH) - 🚀 CHUYÊN GIA THỰC CHIẾN
**Chuỗi cấu hình:** `model=unet_palmnet_gabor` + `losses=arcface` + `loss_schedules=contrastive_first` + `training=arcface` + `dataset=own_original`
- **Backbone (PalmNet Gabor):** Mạng siêu nhẹ chỉ có 3 lớp chập (Conv). Việc mồi sẵn công thức toán học Gabor vào Layer 1 giúp nó lập tức nhìn thấy "đường vân" ngay từ Epoch 1 mà không cần học nhiều.
- **Decoder Regularization:** Đóng vai trò sinh tử! Vì PalmNet quá nông, nếu không có Decoder ép buộc tính loss khôi phục, PalmNet sẽ gãy gọn và vứt bỏ toàn bộ thông tin không gian, khiến ArcFace vô dụng.
- **Đánh giá:**
  - **Ưu điểm:** Siêu nhẹ (vài ngàn tham số), huấn luyện cực nhanh, Inference thời gian thực tuyệt đỉnh. Rất phù hợp nếu muốn deploy hệ thống điểm danh lên Raspberry Pi hoặc hệ thống nhúng rẻ tiền (Chỉ cần bẻ phần Decoder vứt đi lúc deploy).
  - **Nhược điểm:** Khả năng chống chịu với ảnh chất lượng kém (lóa sáng, nhòe) không thể bằng ResNet18.

---

## 3. MẠNG KIỂM THỬ CCNET - 🔬 EXPERIMENTAL
**Chuỗi cấu hình:** `model=unet_ccnet` + `losses=arcface` + `loss_schedules=contrastive_first` + `training=arcface` + `dataset=own_original`
- **Đánh giá:** CCNet (Criss-Cross Network) nổi tiếng trong việc bắt thông tin ngữ cảnh diện rộng (Global Context) thông qua sự chú ý chéo. Việc này về lý thuyết rất tốt để thấy được toàn bộ cấu trúc bàn tay.
  - **Ưu điểm:** Bắt được các cấu trúc vĩ mô của bàn tay (hình dáng bàn tay, các cụm vân tay lớn).
  - **Nhược điểm:** Chi phí tính toán cực cao do Attention Matrix. Có thể hơi thừa thãi vì bài toán Palmprint thiên về đặc trưng cục bộ (đường chỉ tay nhỏ) hơn là ngữ cảnh vĩ mô.

---

## 4. CHUẨN HÓA CƠ BẢN (KHÔNG GABOR) - ⚖️ ĐỐI CHỨNG
**Chuỗi cấu hình:** `model=unet_palmnet` + `losses=arcface` ...
- **Đánh giá:** Thay vì khởi tạo Gabor, mạng tự khởi tạo ngẫu nhiên (Kaiming). Mục đích duy nhất của combo này là chạy để so sánh đối chứng (Ablation Study) nhằm chứng minh luận điểm: *"Việc chèn Gabor Filter có thực sự làm mạng tốt lên không?"*.

---

## TỔNG KẾT TRIẾT LÝ
Các chuỗi kết hợp trên minh chứng cho một triết lý thiết kế mạng tuyệt vời: **Bất kể bạn lắp cái đầu máy nào (ResNet, PalmNet, CCNet)**, hệ thống toa tàu đằng sau (Gồm U-Net Decoder làm chuẩn hóa vật lý + Projector ArcFace làm chia cắt danh tính) đều sẽ ép cái đầu máy đó phải hoạt động ở công suất và độ tinh xảo cao nhất cho bài toán Điểm danh.
