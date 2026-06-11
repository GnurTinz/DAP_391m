# 5 ảnh mỗi bàn tay, ảnh xám

import sys
sys.path.append("../")

import os
import cv2
import numpy as np

from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Đã thêm lại tqdm ở đây nhe bạn
from processing.similar_image import k_medoids_pam_robust, compare_image

def script1(x_path):
    """Đọc ảnh ở dạng grayscale và resize về kích thước 224x224."""
    img = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc file ảnh: {x_path}")
    img = cv2.resize(img, (224, 224))
    return img


def read_full_folder(input_folder, output_folder, k=5, random_state=42):
    """Xử lý một thư mục cụ thể: tính ma trận tương đồng và trích chọn k-medoids."""
    os.makedirs(output_folder, exist_ok=True)

    tempo = os.listdir(input_folder)
    N = len(tempo)
    
    if N == 0:
        return
        
    # Khởi tạo mảng NumPy chuẩn (tránh lỗi shallow copy)
    array = np.zeros((N, N))

    # Tính toán khoảng cách giữa các cặp ảnh (đã bỏ tqdm ở đây để không bị loạn màn hình)
    pairs = combinations(range(N), 2)
    for i, j in tqdm(pairs, desc="Ghép cặp", total=N * (N - 1) // 2):
        x1_path = os.path.join(input_folder, tempo[i])
        x2_path = os.path.join(input_folder, tempo[j])
        
        dist = compare_image(x1_path, x2_path)
        array[i][j] = dist
        array[j][i] = dist
    
    selected_index = k_medoids_pam_robust(array, k, random_state=random_state)
    
    for index in selected_index:
        img_root = os.path.join(input_folder, tempo[index])
        img_path = os.path.join(output_folder, tempo[index])
        cv2.imwrite(img_path, script1(img_root))


def process_single_folder(folder, input_folder, output_folder):
    """
    Hàm xử lý độc lập cho từng luồng (Thread).
    """
    _input_folder = os.path.normpath(os.path.join(input_folder, folder))
    _output_folder = os.path.normpath(os.path.join(output_folder, folder))
    
    if not os.path.exists(_input_folder):
        return f"Lỗi: Thư mục không tồn tại -> {folder}"
        
    os.makedirs(_output_folder, exist_ok=True)
    
    if len(os.listdir(_output_folder)) != 0:
        return f"Bỏ qua (Đã xử lý): {folder}"

    read_full_folder(_input_folder, _output_folder, random_state=1310)
    return f"Xong: {folder}"


def main_parallel(input_folders, input_folder, output_folder, max_workers=4):
    """
    Bộ điều khiển trung tâm sử dụng ThreadPool kết hợp với TQDM quản lý tiến độ tổng thể.
    """
    print(f"=== Bắt đầu chạy song song với {max_workers} Workers ===")
    
    total_folders = len(input_folders)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit toàn bộ tác vụ vào luồng xử lý
        future_to_folder = {
            executor.submit(process_single_folder, folder, input_folder, output_folder): folder 
            for folder in input_folders
        }
        
        # Bọc tqdm quanh vòng lặp as_completed để hiển thị thanh tiến trình tổng thể
        # Sử dụng thuộc tính `desc` để đặt tên hiển thị cho thanh tiến trình
        with tqdm(as_completed(future_to_folder), total=total_folders, desc="Xử lý thư mục") as pbar:
            for future in pbar:
                folder = future_to_folder[future]
                try:
                    result = future.result()
                    # Cập nhật thông tin thư mục vừa xử lý xong vào bên cạnh thanh tiến trình cho đẹp nhe
                    pbar.set_postfix_str(f"Vừa xong: {folder[:15]}...") 
                except Exception as exc:
                    # Nếu có lỗi nghiêm trọng, ta in ra phía trên thanh tiến trình để không làm vỡ giao diện
                    tqdm.write(f"[LỖI SẬP LUỒNG] Thư mục '{folder}' bị lỗi: {exc}")

    print("=== TOÀN BỘ CHƯƠNG TRÌNH ĐÃ HOÀN THÀNH XỬ LÝ ===")


if __name__ == "__main__":
    input_folder = "E:/palm/data-collection/mother_dataset"
    output_folder = "E:/palm/data-collection/sub_dataset/script1"

    try:
        input_tempo = os.listdir(input_folder)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục mẹ tại '{input_folder}'")
        sys.exit(1)

    input_folders = [i + '/left' for i in input_tempo] + [i + '/right' for i in input_tempo]

    # Bạn có thể tăng số workers lên 4 hoặc 8 tuỳ thuộc vào tốc độ đọc ghi của ổ cứng nhe
    MAX_WORKERS = 8
    
    main_parallel(input_folders, input_folder, output_folder, max_workers=MAX_WORKERS)