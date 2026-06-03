import cv2
import os
import glob
import argparse

def crop_image(input_folder, output_folder, discard_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(discard_folder, exist_ok=True)

    image_paths = [input_folder + '/' + tmp for tmp in os.listdir(input_folder)]

    if not image_paths:
        print("Không tìm thấy ảnh nào trong thư mục đầu vào!")
        return

    print(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu xử lý...")
    print("\n--- HƯỚNG DẪN THAO TÁC ---")
    print("BƯỚC 1: Dùng chuột TRÁI kéo giữ để chọn vùng trên ảnh. Nhấn ENTER hoặc SPACE để CHỐT VÙNG.")
    print("BƯỚC 2: Nhấn một trong các phím sau để thực hiện lệnh:")
    print("  - Nhấn ENTER (Phím mã 13): XÁC NHẬN lưu ẢNH GỐC (không cắt).")
    print("  - Nhấn SPACE (Phím cách): XÁC NHẬN CẮT ẢNH theo vùng đã chọn.")
    print("  - Nhấn phím 'd' hoặc 'D' : LOẠI BỎ (DISCARD) ảnh này.")
    print("  - Nhấn phím ESC          : THOÁT chương trình hoàn toàn.")
    print("---------------------------\n")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Không thể đọc ảnh: {filename}, tự động bỏ qua.")
            continue
        
        window_name = f"Dang xu ly: {filename}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
        x, y, w, h = roi

        print(f"-> Đang chờ lệnh cho ảnh {filename} (Bấm SPACE để cắt, ENTER để lưu gốc, D để loại)...")
        key = cv2.waitKey(0) & 0xFF

        try:
            path_parts = input_folder.replace('\\', '/').split('/')
            saved_filename = f"{path_parts[3]}_{path_parts[4]}_{path_parts[5]}_{filename}"
        except IndexError:
            saved_filename = f"processed_{filename}"

        if key == 27: 
            print("Đã chủ động thoát chương trình.")
            cv2.destroyAllWindows()
            break

        elif key == ord('d') or key == ord('D'):
            print(f"[DISCARD] Đã loại bỏ ảnh: {filename}")
            cv2.imwrite(os.path.join(discard_folder, saved_filename), image)

        elif key == 32:
            if w > 0 and h > 0:
                # Tiến hành cắt mảng (Matrix Slicing)
                cropped_image = image[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(output_folder, filename), cropped_image)
                print(f"[SUCCESS] Đã CẮT và lưu thành công: {filename}")
            else:
                # Nếu bấm SPACE mà trước đó chưa kéo chuột chọn vùng
                print(f"[WARNING] Chưa chọn vùng cắt! Tự động đưa {filename} vào discard.")
                cv2.imwrite(os.path.join(discard_folder, saved_filename), image)

        elif key == 13:
            cv2.imwrite(os.path.join(output_folder, filename), image)
            print(f"[SUCCESS] Đã lưu ẢNH GỐC (Không cắt): {filename}")

        else:
            print(f"[UNKNOWN KEY] Phím không hợp lệ. Đưa vào discard: {filename}")
            cv2.imwrite(os.path.join(discard_folder, saved_filename), image)

        cv2.destroyWindow(window_name)
    
    cv2.destroyAllWindows()

def generate_input_folder(input_path, output_path):
    # Đường dẫn trỏ tới data-collection/dataset/c1
    # Đường dẫn lưu trữ là: data-collection/processed/c1
    os.makedirs(output_path, exist_ok=True)

    folders = os.listdir(input_path)

    outputs = []
    for folder in folders:
        tmp_input_folder = input_path + '/' + folder
        tmp_output_folder = output_path + '/' + folder

        outputs.append(
            (tmp_input_folder + '/left', tmp_output_folder + '/left')
        )

        outputs.append(
            (tmp_input_folder + '/right', tmp_output_folder + '/right')
        )

    return outputs

def process_all_path():
    input_path = '../data-collection/dataset'
    output_path = '../data-collection/preprocessed'
    result = []
    
    for code in os.listdir(input_path):
        result.extend(
            generate_input_folder(input_path + '/' + code, output_path + '/' + code)
        )
    
    with open('../data-collection/paths.txt', mode='w', encoding='utf-8') as f:
        for i, o in result:
            tmp = f'{i},{o}\n'
            print("Handling:", tmp)
            f.write(tmp)

def main():
    # Bước 1: Khởi tạo bộ phân tích tham số (Argument Parser)
    parser = argparse.ArgumentParser(
        description="Chương trình cắt ảnh hàng loạt bằng OpenCV và phân loại ảnh."
    )

    # Bước 2: Định nghĩa các tham số đầu vào (Add Arguments)
    
    # Tham số BẮT BUỘC (Positional Argument): Không có dấu gạch ngang trước tên
    parser.add_argument(
        "input_folder", 
        type=str, 
        help="Đường dẫn đến thư mục chứa ảnh gốc cần xử lý."
    )

    # Tham số TÙY CHỌN (Optional Argument): Có dấu gạch ngang (-o hoặc --output)
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="output_folder", 
        help="Thư mục lưu ảnh đã cắt (Mặc định: output_folder)."
    )

    args = parser.parse_args()
    output_folder = str(args.input_folder).replace('dataset', 'preprocessed')
    crop_image(args.input_folder, output_folder, discard_folder="../data-collection/discard")

if __name__ == '__main__':
    # process_all_path()
    # crop_image("../data-collection/dataset/c1/person_1/left", "../data-collection/preprocessed/c1/person_1/left", discard_folder="../data-collection/discard")
    main()