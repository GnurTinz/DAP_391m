import os
import glob
from collections import defaultdict
from .base import BaseDataset
import random

class TongjiDataset(BaseDataset):
    """
    Tongji Dataset.
    Cấu trúc thư mục:
    - data/Tongji/session1
    - data/Tongji/session2

    Quy luật: Sắp xếp theo tên file alphabet, cứ 10 file liên tiếp thuộc về 1 người.
    Mỗi session có 6000 ảnh => 600 người.

    split_mode hỗ trợ:
      - 'session' : session1 = Train, session2 = Val (closed-set)
      - 'mixed'   : Gộp 2 session, chia tỷ lệ ảnh per-person (closed-set)
      - 'person'  : Chia theo số người → hỗ trợ Open-Set Recognition
                    Cần thêm 'split' trong config: 'train' | 'register' | 'probe' | 'stranger'
    """

    def __init__(self, data_dir: str, config: dict, is_train: bool = True):
        self.split_mode = config.get('split_mode', 'session')
        self.train_ratio = config.get('train_ratio', 0.8)
        self.seed = config.get('seed', 42)

        self.session1_dir = os.path.join(data_dir, 'session1')
        self.session2_dir = os.path.join(data_dir, 'session2')

        super().__init__(data_dir, config, is_train)

    def _get_session_files(self, session_dir):
        if not os.path.exists(session_dir):
            return []
        # Lọc các file ảnh và sắp xếp alphabet để đảm bảo thứ tự
        valid_exts = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
        files = [f for f in os.listdir(session_dir) if f.lower().endswith(valid_exts)]
        files.sort()
        return files

    def _load_data(self):
        s1_files = self._get_session_files(self.session1_dir)
        s2_files = self._get_session_files(self.session2_dir)

        if self.split_mode == 'session':
            # Kịch bản 1: session1 = train, session2 = val
            if self.is_train:
                for i, f in enumerate(s1_files):
                    label = i // 10
                    self.samples.append((os.path.join(self.session1_dir, f), label))
            else:
                for i, f in enumerate(s2_files):
                    label = i // 10
                    self.samples.append((os.path.join(self.session2_dir, f), label))

        elif self.split_mode == 'mixed':
            # Kịch bản 2: Trộn session1 và session2 lại, chia tỷ lệ
            person_dict = defaultdict(list)

            for i, f in enumerate(s1_files):
                label = i // 10
                person_dict[label].append(os.path.join(self.session1_dir, f))

            for i, f in enumerate(s2_files):
                label = i // 10
                person_dict[label].append(os.path.join(self.session2_dir, f))

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
            self._load_person_split(s1_files, s2_files)

        else:
            raise ValueError(f"Không hỗ trợ split_mode: {self.split_mode} cho TongjiDataset")

        current_split = self.config.get('split', 'train' if self.is_train else 'val')
        print(f"Loaded TongjiDataset ({self.split_mode}) "
              f"split='{current_split}': {len(self.samples)} samples, "
              f"{len(set(lbl for _, lbl in self.samples))} identities")

    def _load_person_split(self, s1_files, s2_files):
        """
        Person-based split cho đánh giá Open-Set Recognition chuẩn.

        Config keys:
          num_train_persons    (int)       : X người dùng để train
          num_known_persons    (int)       : Y người known (register + probe)
          num_stranger_persons (int|None)  : Z người lạ; None → toàn bộ còn lại
          register_ratio       (float)     : tỷ lệ ảnh của Y người dùng làm gallery
          split                (str)       : 'train' | 'register' | 'probe' | 'stranger'
        """
        num_train    = self.config.get('num_train_persons', 400)
        num_known    = self.config.get('num_known_persons', 100)
        num_stranger = self.config.get('num_stranger_persons', None)
        reg_ratio    = self.config.get('register_ratio', 0.5)
        cur_split    = self.config.get('split', 'train' if self.is_train else 'val')

        # ── Xây dựng person_dict: orig_label_int -> [paths] ───────────────
        # Tongji: cứ 10 file liên tiếp (sorted) = 1 người
        person_dict = defaultdict(list)
        for i, f in enumerate(s1_files):
            label = i // 10
            person_dict[label].append(os.path.join(self.session1_dir, f))
        for i, f in enumerate(s2_files):
            label = i // 10
            person_dict[label].append(os.path.join(self.session2_dir, f))

        # ── Shuffle danh sách người theo seed ─────────────────────────────
        all_persons = sorted(person_dict.keys())   # [0, 1, ..., 599]
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

        # ── Global Label Mapping ──────────────────────────────────────────
        pid_to_label = {}
        for pid in sorted(train_persons) + sorted(known_persons) + sorted(stranger_persons):
            if pid not in pid_to_label:
                pid_to_label[pid] = len(pid_to_label)

        # ── Nạp mẫu theo cur_split ────────────────────────────────────────
        if cur_split == 'train':
            for orig_label in sorted(train_persons):
                new_label = pid_to_label[orig_label]
                for path in person_dict[orig_label]:
                    self.samples.append((path, new_label))

        elif cur_split == 'register':
            for orig_label in sorted(known_persons):
                new_label = pid_to_label[orig_label]
                paths = person_dict[orig_label]
                split_idx = max(1, int(len(paths) * reg_ratio))
                for path in paths[:split_idx]:
                    self.samples.append((path, new_label))

        elif cur_split == 'probe':
            for orig_label in sorted(known_persons):
                new_label = pid_to_label[orig_label]
                paths = person_dict[orig_label]
                split_idx = max(1, int(len(paths) * reg_ratio))
                for path in paths[split_idx:]:
                    self.samples.append((path, new_label))

        elif cur_split == 'stranger':
            for orig_label in sorted(stranger_persons):
                new_label = pid_to_label[orig_label]
                for path in person_dict[orig_label]:
                    self.samples.append((path, new_label))

        elif cur_split == 'val':
            for orig_label in sorted(known_persons) + sorted(stranger_persons):
                new_label = pid_to_label[orig_label]
                for path in person_dict[orig_label]:
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
