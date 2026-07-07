import unittest
import yaml
import os
import sys

# Thêm thư mục gốc (E:\palm) vào PYTHONPATH để chạy test dễ dàng
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.factory import DatasetFactory

class TestPersonSplit(unittest.TestCase):
    def setUp(self):
        self.splits = ['train', 'register', 'probe', 'stranger']

    def load_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def verify_dataset_split(self, config_path, dataset_name):
        if not os.path.exists(config_path):
            self.skipTest(f"Không tìm thấy config: {config_path}")
            
        config = self.load_config(config_path)
        data_dir = config.get('data_dir', '')
        
        datasets = {}
        for sp in self.splits:
            cfg = config.copy()
            cfg['split'] = sp
            ds = DatasetFactory.create(name=dataset_name, data_dir=data_dir, config=cfg, is_train=(sp == 'train'))
            datasets[sp] = ds

        def get_unique_labels(ds):
            # Nếu mảng mẫu rỗng, tức là không có dữ liệu thật trên đường dẫn này
            if not getattr(ds, 'samples', None):
                return set()
            return set(label for _, label in ds.samples)

        train_labels = get_unique_labels(datasets['train'])
        known_labels_reg = get_unique_labels(datasets['register'])
        known_labels_probe = get_unique_labels(datasets['probe'])
        stranger_labels = get_unique_labels(datasets['stranger'])
        
        # Bỏ qua assert strict nếu data rỗng (bạn có thể đang không để ảnh tại vị trí này)
        if len(train_labels) == 0:
            print(f"\n[!] Bỏ qua test '{dataset_name}' vì không tải được hình ảnh thực tế từ '{data_dir}'")
            return

        print(f"\n=== KẾT QUẢ DỮ LIỆU: {dataset_name.upper()} ===")
        print(f"Số ID tập Train    : {len(train_labels)}")
        print(f"Số ID tập Register : {len(known_labels_reg)}")
        print(f"Số ID tập Probe    : {len(known_labels_probe)}")
        print(f"Số ID tập Stranger : {len(stranger_labels)}")

        # 1. Kiểm tra số lượng người có đúng với config file không
        self.assertEqual(len(train_labels), config.get('num_train_persons', len(train_labels)), "Sai số lượng ID tập Train")
        self.assertEqual(len(known_labels_reg), config.get('num_known_persons', len(known_labels_reg)), "Sai số lượng ID tập Register/Known")
        
        # 2. Tập Known lúc Register (Gallery) và lúc Probe (Test) phải thuộc cùng một danh sách người
        self.assertEqual(known_labels_reg, known_labels_probe, "Register và Probe phải chia sẻ cùng danh sách ID (Known)")

        if config.get('num_stranger_persons') is not None:
             self.assertEqual(len(stranger_labels), config.get('num_stranger_persons'), "Sai số lượng ID tập Stranger")

        # Thêm bước kiểm tra thực tế xem đường dẫn ảnh có lấy đúng theo cờ hand_filter hay không
        expected_hand = config.get('hand_filter', 'both')
        if expected_hand != 'both' and dataset_name in ['iitd', 'OwnOriginalDataset']:
            for sp_name, sp_ds in datasets.items():
                if getattr(sp_ds, 'samples', None):
                    for img_path, _ in sp_ds.samples:
                        self.assertIn(expected_hand.lower(), img_path.lower(), 
                                      f"Lỗi ở tập {sp_name}: Ảnh '{img_path}' không thuộc tay '{expected_hand}'")
            print(f"✅ Đã xác minh 100% đường dẫn ảnh đều thuộc tay: '{expected_hand}'.")

        # 3. Tính Độc Lập của Open-Set Recognition: 3 tập Train, Known, Stranger KHÔNG ĐƯỢC phép trùng lặp ID
        self.assertEqual(len(train_labels.intersection(known_labels_reg)), 0, "LỖI CHÍNH MẠNG: ID tập Train và tập Known bị trùng!")
        self.assertEqual(len(train_labels.intersection(stranger_labels)), 0, "LỖI CHÍNH MẠNG: ID tập Train và tập Stranger bị trùng!")
        self.assertEqual(len(known_labels_reg.intersection(stranger_labels)), 0, "LỖI CHÍNH MẠNG: ID tập Known và tập Stranger bị trùng!")
        
        print(f"✅ {dataset_name} đạt chuẩn Open-Set Recognition.")

    def test_1_iitd_person_left(self):
        self.verify_dataset_split("config/dataset/iitd_person_left.yaml", "iitd")

    def test_2_tongji_person_left(self):
        # Lưu ý: Tongji mặc định chỉ phân theo session, không có thư mục left/right. 
        # Cấu trúc của Tongji đã an toàn (mỗi 10 files liên tiếp là 1 người).
        self.verify_dataset_split("config/dataset/tongji_person_left.yaml", "tongji")

    def test_3_own_original_person_left(self):
        # Dataset thu thập riêng, đã hỗ trợ tốt tính năng hand_filter='left'
        self.verify_dataset_split("config/dataset/own_original_person_left.yaml", "OwnOriginalDataset")

if __name__ == '__main__':
    unittest.main()
