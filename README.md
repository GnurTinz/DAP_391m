markdown_content = """# CCNet for Evaluation (ccnet_for_eval)

Dự án này triển khai và đánh giá mô hình CCNet. Dưới đây là hướng dẫn chi tiết về cấu trúc dữ liệu và cách chạy mã nguồn huấn luyện (train) trên các thiết bị cá nhân (local) có cấu hình thông thường.

## Lưu ý về Dữ liệu (Data)

Để chạy được mô hình, bạn cần tải dữ liệu về và sắp xếp cấu trúc thư mục theo đúng định dạng mà mô hình yêu cầu.
* Các file dữ liệu dạng `.txt` dùng cho quá trình huấn luyện (train) và kiểm thử (test) đã được tạo sẵn (gen sẵn) trong thư mục dữ liệu.

## Hướng dẫn Huấn luyện (Training)

Để huấn luyện mô hình trên các thiết bị local có cấu hình bình thường, lệnh dưới đây đã được tối ưu hóa cấu hình (ví dụ như `batch_size`, `lr`, `epoch_num`) để chạy ổn định và phù hợp nhất:

python train.py --id_num 231 --train_set_file ./data/train_script2.txt --test_set_file ./data/test_script2.txt --des_path ./checkpoints/script2 --path_rst ./results/script2 --batch_size 128 --epoch_num 200 --lr 0.0005 --temp 0.07 --weight1 0.8 --weight2 0.2 --com_weight 0.8 --redstep 500 --gpu_id 0