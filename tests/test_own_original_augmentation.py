import unittest
import torch
from src.datasets.own_original_dataset import OwnOriginalDataset

class TestOwnOriginalAugmentation(unittest.TestCase):
    def test_grayscale_augmentation_is_active(self):
        """
        Kiểm chứng luồng chương trình tích hợp RandomGrayscale vào OwnOriginalDataset.
        Sử dụng cấu hình mặc định (tự động tiêm random_grayscale=0.5 khi is_train=True).
        """
        # 1. Khởi tạo dataset (is_train=True)
        # Giả sử thư mục 'data/collect' có tồn tại (hoặc BaseDataset xử lý dry-run)
        dataset = OwnOriginalDataset(is_train=True)
        
        # 2. Phân tích Transform Pipeline
        transforms_list = dataset.transform.transforms
        
        # Tìm xem có RandomGrayscale trong danh sách hay không
        has_grayscale = any(type(t).__name__ == 'RandomGrayscale' for t in transforms_list)
        self.assertTrue(has_grayscale, "Không tìm thấy RandomGrayscale trong pipeline augmentation!")
        
        # Lấy ra đối tượng RandomGrayscale để kiểm tra xác suất p
        grayscale_transform = next(t for t in transforms_list if type(t).__name__ == 'RandomGrayscale')
        self.assertEqual(grayscale_transform.p, 0.5, "Xác suất random_grayscale không bằng 0.5 như thiết lập!")
        print(f"\n[OK] Transform Pipeline đã tích hợp: {dataset.transform}")

    def test_grayscale_tensor_output(self):
        """
        Kiểm chứng bằng dữ liệu thật xem Tensor trả về có đảm bảo:
        - Giữ nguyên 3 channels
        - Khi bị chuyển xám, R = G = B
        """
        # Set xác suất p=1.0 để BẮT BUỘC tất cả ảnh đều thành ảnh xám để test
        config = {'transforms': {'random_grayscale': 1.0}}
        dataset = OwnOriginalDataset(config=config, is_train=True)
        
        if len(dataset.samples) == 0:
            self.skipTest("Thư mục data/collect không có ảnh thật nào để test. Skip test này.")
            
        # Lấy 1 ảnh thật từ dataset
        img_tensor, label = dataset[0]
        
        # Kiểm tra shape phải là [3, H, W]
        self.assertEqual(img_tensor.shape[0], 3, "Ảnh không giữ được 3 channels!")
        
        # Vì ta ép p=1.0 (100% grayscale), nên kênh R, G, B phải giống hệt nhau
        channel_r = img_tensor[0]
        channel_g = img_tensor[1]
        channel_b = img_tensor[2]
        
        # Kiểm tra sự giống nhau tuyệt đối giữa các kênh
        # Chú ý: Do có bước Normalize ở cuối, R=G=B ở bản gốc vẫn sẽ thành R=G=B ở bản Normalize 
        # (với điều kiện Normalize mean và std giống nhau cho 3 kênh [0.5, 0.5, 0.5])
        self.assertTrue(torch.allclose(channel_r, channel_g, atol=1e-4), "Kênh R và G không giống nhau!")
        self.assertTrue(torch.allclose(channel_r, channel_b, atol=1e-4), "Kênh R và B không giống nhau!")
        print("\n[OK] Dữ liệu thật đã được chuyển xám thành công (3 channel R=G=B).")

if __name__ == '__main__':
    unittest.main()
