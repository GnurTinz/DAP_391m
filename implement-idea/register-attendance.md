Logic cài đặt, ràng buộc cho tạo luồng thực thi

Các bước chung:
- Lấy model đã tối ưu (chỉ lấy khối PalmEncoder và Projected)
- Testing ở chế độ open-set
- Tiến hành load bộ dữ liệu được chỉ định

- Quá trình so sánh, tìm người giống nhất trong cơ sở dữ liệu được lưu đều hoạt động sau khi projected

Các cấu hình register:
+ Cho phép chạy tối ưu biểu diễn của person thay vì trích xuất trực tiếp biến đổi từ Projected 
+ Việc tối ưu biểu diễn của person có 2 chế độ. Là biểu diễn r được nối hoặc cộng vào latent vector mu hoặc latent vector z, sau đó qua Projected. Tối ưu dùng BCE loss như thường Hoặc là r được định nghĩa trong Projected, tối ưu dùng BCE Loss như thường
+ Nếu không chạy tối ưu biểu diễn thì mô hình lưu lại trích xuất trực tiếp (kết quả của projected mu). Tức là Projected nhận đầu vào là mu và r=0.

Cấu hình attendance:
+ Tuân theo nguyên tắc của tạo luồng thực thi: lấy model đã tối ưu, testing ở chế độ open-set, tiến hành load bộ dữ liệu được chỉ định.
+ Các bước so sánh, tìm người giống nhất trong cơ sở dữ liệu được lưu đều hoạt động sau khi projected 
+ Khi chạy attendance thì mặc định r luôn bằng 0. Tức là Projected nhận đầu vào là mu và r=0. Không chạy tối ưu biểu diễn.

Mong muốn:
+ Có tiến hành đánh giá Acc, rank-1, rank-5 và eer
+ Chạy xong phải ghi log (file log có format là attendance_<bộ dữ liệu>.log). trong file có thông tin của cấu hình sử dụng, Acc, rank-1, rank-5 và eer, khoảng cách trung bình, sai số chuẩn. 
+ Chạy xong phải ghi ra biểu đồ biểu diễn của r, và biểu đồ biểu diễn của person trong db trên mặt phẳng PCA.
