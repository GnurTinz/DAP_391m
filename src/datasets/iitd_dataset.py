import os
from collections import defaultdict
from .base import BaseDataset
import random

class IITDDataset(BaseDataset):
    """
    IITD Palmprint V1 Dataset.
    Cấu trúc thư mục dự kiến:
    - data/IITD Palmprint V1/Segmented/Left
    - data/IITD Palmprint V1/Segmented/Right

    Quy luật: Tên file dạng <id>_<num>.bmp (vd: 001_01.bmp)
    ID của người là phần trước dấu gạch dưới `_`.

    split_mode hỗ trợ:
      - 'hand'   : Left = Train, Right = Val (closed-set)
      - 'ratio'  : Gộp Left+Right, chia tỷ lệ ảnh per-person (closed-set)
      - 'person' : Chia theo số người → hỗ trợ Open-Set Recognition
                   Cần thêm 'split' trong config: 'train' | 'register' | 'probe' | 'stranger'
    """

    def __init__(self, data_dir: str, config: dict, is_train: bool = True):
        self.split_mode = config.get('split_mode', 'hand')
        self.train_ratio = config.get('train_ratio', 0.8)
        self.seed = config.get('seed', 42)

        self.left_dir  = os.path.join(data_dir, 'Left')
        self.right_dir = os.path.join(data_dir, 'Right')

        # Dictionary map string ID (vd '001') -> int ID (vd 0)
        self.id_to_label = {}

        super().__init__(data_dir, config, is_train)

    def _parse_id(self, filename):
        """Tách tên file '001_01.bmp' -> label int (chỉ dùng cho mode hand/ratio)."""
        basename = os.path.splitext(filename)[0]
        person_id_str = basename.split('_')[0]

        if person_id_str not in self.id_to_label:
            self.id_to_label[person_id_str] = len(self.id_to_label)

        return self.id_to_label[person_id_str]

    def _get_files(self, folder):
        if not os.path.exists(folder):
            return []
        valid_exts = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
        return [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]

    def _load_data(self):
        left_files  = self._get_files(self.left_dir)
        right_files = self._get_files(self.right_dir)

        hand_filter = self.config.get('hand_filter', 'both')
        if hand_filter == 'left':
            right_files = []
        elif hand_filter == 'right':
            left_files = []

        if self.split_mode == 'hand':
            # Kịch bản 1: Left = train, Right = val
            target_dir = self.left_dir if self.is_train else self.right_dir
            files = left_files if self.is_train else right_files

            for f in files:
                label = self._parse_id(f)
                self.samples.append((os.path.join(target_dir, f), label))

        elif self.split_mode == 'ratio':
            # Kịch bản 2: Gộp chung Left và Right, chia theo tỷ lệ
            person_dict = defaultdict(list)

            for f in left_files:
                label = self._parse_id(f)
                person_dict[label].append(os.path.join(self.left_dir, f))

            for f in right_files:
                label = self._parse_id(f)
                person_dict[label].append(os.path.join(self.right_dir, f))

            rng = random.Random(self.seed)
            for label, paths in person_dict.items():
                rng.shuffle(paths)
                split_idx = int(len(paths) * self.train_ratio)

                if self.is_train:
                    selected_paths = paths[:split_idx]
                else:
                    selected_paths = paths[split_idx:]

                for p in selected_paths:
                    self.samples.append((p, label))

        elif self.split_mode == 'person':
            # Kịch bản 3: Open-Set Recognition — chia theo số người
            self._load_person_split(left_files, right_files)

        else:
            raise ValueError(f"Không hỗ trợ split_mode: {self.split_mode} cho IITDDataset")

        current_split = self.config.get('split', 'train' if self.is_train else 'val')
        print(f"Loaded IITDDataset ({self.split_mode}) "
              f"split='{current_split}': {len(self.samples)} samples, "
              f"{len(set(lbl for _, lbl in self.samples))} identities")

    def _load_person_split(self, left_files, right_files):
        """
        Person-based split cho đánh giá Open-Set Recognition chuẩn.

        Config keys:
          num_train_persons   (int)   : X người dùng để train
          num_known_persons   (int)   : Y người known (register + probe)
          num_stranger_persons (int|None): Z người lạ; None → toàn bộ còn lại
          register_ratio      (float) : tỷ lệ ảnh của Y người dùng làm gallery
          split               (str)   : 'train' | 'register' | 'probe' | 'stranger'
        """
        num_train    = self.config.get('num_train_persons', 160)
        num_known    = self.config.get('num_known_persons', 40)
        num_stranger = self.config.get('num_stranger_persons', None)
        reg_ratio    = self.config.get('register_ratio', 0.5)
        cur_split    = self.config.get('split', 'train' if self.is_train else 'val')

        # ── Xây dựng person_dict: pid_str -> [paths] ──────────────────────
        person_dict = defaultdict(list)
        for f in left_files:
            pid = os.path.splitext(f)[0].split('_')[0]
            person_dict[pid].append(os.path.join(self.left_dir, f))
        for f in right_files:
            pid = os.path.splitext(f)[0].split('_')[0]
            person_dict[pid].append(os.path.join(self.right_dir, f))

        # ── Shuffle danh sách người theo seed ─────────────────────────────
        all_persons = sorted(person_dict.keys())
        rng = random.Random(self.seed)
        rng.shuffle(all_persons)

        total = len(all_persons)
        if num_stranger is None:
            num_stranger = total - num_train - num_known

        used = num_train + num_known + num_stranger
        assert used <= total, (
            f"X({num_train}) + Y({num_known}) + Z({num_stranger}) = {used} "
            f"vượt quá tổng số người ({total}) trong dataset."
        )

        train_persons    = all_persons[:num_train]
        known_persons    = all_persons[num_train : num_train + num_known]
        stranger_persons = all_persons[num_train + num_known : num_train + num_known + num_stranger]

        print(f"  [Person Split] Total={total} | "
              f"Train={len(train_persons)} | "
              f"Known={len(known_persons)} | "
              f"Stranger={len(stranger_persons)} | "
              f"register_ratio={reg_ratio}")

        # ── Nạp mẫu theo cur_split ────────────────────────────────────────
        pid_to_label = {}
        for pid in sorted(train_persons) + sorted(known_persons) + sorted(stranger_persons):
            if pid not in pid_to_label:
                pid_to_label[pid] = len(pid_to_label)

        if cur_split == 'train':
            # X người train: toàn bộ ảnh
            for pid in sorted(train_persons):
                new_label = pid_to_label[pid]
                for path in sorted(person_dict[pid]):
                    self.samples.append((path, new_label))

        elif cur_split == 'register':
            # Y người known: phần đầu register_ratio làm gallery
            for pid in sorted(known_persons):
                new_label = pid_to_label[pid]
                paths = sorted(person_dict[pid])
                split_idx = max(1, int(len(paths) * reg_ratio))
                for path in paths[:split_idx]:
                    self.samples.append((path, new_label))

        elif cur_split == 'probe':
            # Y người known: phần sau (1 - register_ratio) làm probe
            for pid in sorted(known_persons):
                new_label = pid_to_label[pid]
                paths = sorted(person_dict[pid])
                split_idx = max(1, int(len(paths) * reg_ratio))
                for path in paths[split_idx:]:
                    self.samples.append((path, new_label))

        elif cur_split == 'stranger':
            # Z người stranger: toàn bộ làm impostor/probe
            for pid in sorted(stranger_persons):
                new_label = pid_to_label[pid]
                for path in sorted(person_dict[pid]):
                    self.samples.append((path, new_label))

        elif cur_split == 'val':
            # Gom chung register, probe, stranger để visualization
            for pid in sorted(known_persons) + sorted(stranger_persons):
                new_label = pid_to_label[pid]
                for path in sorted(person_dict[pid]):
                    self.samples.append((path, new_label))

        else:
            raise ValueError(
                f"Unknown split='{cur_split}' cho person mode. "
                f"Hợp lệ: 'train', 'register', 'probe', 'stranger', 'val'."
            )

    def __len__(self):
        return len(self.samples) if len(self.samples) > 0 else 100  # Mock length for dry run

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            import torch
            return torch.zeros(3, self.image_size[0], self.image_size[1]), 0

        img_path, label = self.samples[idx]
        image = self._load_image(img_path)
        return image, label

    def get_labels(self):
        if not self.samples:
            return [0] * 100
        return [label for _, label in self.samples]
