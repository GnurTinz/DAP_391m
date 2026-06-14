"""
train_model_comparison.py
=========================
Script so sánh hiệu năng các kiến trúc mô hình (MobileNetV2, ResNet50, EfficientNet-B0)
huấn luyện bằng Triplet Loss trên tập dữ liệu palmprint.

Cách chạy:
    python train_model_comparison.py

Yêu cầu:
    - Đã có dataloader (triplet_loader) từ notebook gốc, HOẶC
    - Chạy độc lập bằng cách cung cấp DATASET_PATH bên dưới.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models as models

# ============================================================
# CẤU HÌNH — chỉnh sửa các tham số tại đây
# ============================================================
DATASET_PATH = "./my_palmprint_dataset/total"   # Đường dẫn tập dữ liệu
EMBEDDING_DIM = 128                              # Số chiều vector embedding
EPOCHS        = 10                               # Số epoch huấn luyện
BATCH_SIZE    = 32                               # Batch size
LEARNING_RATE = 1e-4                             # Learning rate (Adam)
MARGIN        = 0.5                              # Biên độ Triplet Loss
MODELS_TO_TEST = ["mobilenet_v2", "resnet50", "efficientnet_b0"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị: {device}")


# ============================================================
# 1. DATALOADER (tự khởi tạo nếu chạy độc lập)
# ============================================================
def build_dataloader(dataset_path, batch_size=32):
    """Khởi tạo PalmprintTripletDataset và DataLoader."""
    import os
    import random
    import platform
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    from functools import lru_cache

    MAX_CACHE_SIZE = 1000

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def _cached_load(path):
        return Image.open(path).convert("RGB")

    class PalmprintTripletDataset(Dataset):
        def __init__(self, base_path, transform=None):
            self.transform = transform
            folder_names = [
                f for f in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, f))
            ]
            self.all_images_by_class = {}
            self.all_image_paths = []
            self.image_to_class = []

            for idx, folder in enumerate(folder_names):
                folder_path = os.path.join(base_path, folder)
                self.all_images_by_class[idx] = []
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                            img_path = os.path.join(root, file)
                            self.all_images_by_class[idx].append(img_path)
                            self.all_image_paths.append(img_path)
                            self.image_to_class.append(idx)

            self.valid_classes = [
                c for c, imgs in self.all_images_by_class.items() if len(imgs) >= 2
            ]
            print(f"Dataset: {len(self.all_image_paths)} ảnh | {len(self.valid_classes)} lớp hợp lệ")

        def __len__(self):
            return len(self.all_image_paths)

        def __getitem__(self, idx):
            anchor_path  = self.all_image_paths[idx]
            anchor_class = self.image_to_class[idx]

            if anchor_class not in self.valid_classes:
                anchor_class = random.choice(self.valid_classes)
                anchor_path  = random.choice(self.all_images_by_class[anchor_class])

            available     = [p for p in self.all_images_by_class[anchor_class] if p != anchor_path]
            positive_path = random.choice(available if available else self.all_images_by_class[anchor_class])
            negative_class = random.choice([c for c in self.valid_classes if c != anchor_class])
            negative_path  = random.choice(self.all_images_by_class[negative_class])

            def load(p):
                img = _cached_load(p)
                return img.copy()

            imgs = [load(anchor_path), load(positive_path), load(negative_path)]
            if self.transform:
                imgs = [self.transform(i) for i in imgs]
            return tuple(imgs)

    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = PalmprintTripletDataset(dataset_path, transform=img_transforms)
    num_workers = 0 if platform.system() == "Windows" else 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader


# ============================================================
# 2. KHỞI TẠO MÔ HÌNH
# ============================================================
def get_metric_model(model_name: str, embedding_dim: int = 128) -> nn.Module:
    """Thay tầng phân loại cuối bằng Embedding Head (embedding_dim chiều)."""
    if model_name == "mobilenet_v2":
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, embedding_dim))

    elif model_name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, embedding_dim)

    elif model_name == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, embedding_dim))

    else:
        raise ValueError(f"Kiến trúc '{model_name}' chưa được hỗ trợ.")

    return base


# ============================================================
# 3. HUẤN LUYỆN VÀ ĐÁNH GIÁ MỘT MÔ HÌNH
# ============================================================
def train_and_evaluate(model_name: str, triplet_loader, epochs: int = 10):
    print(f"\n{'='*50}")
    print(f" BẮT ĐẦU HUẤN LUYỆN: {model_name.upper()}")
    print(f"{'='*50}")

    model     = get_metric_model(model_name, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

    history = {"loss": [], "gap": []}
    t0 = time.time()

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for anchor, positive, negative in triplet_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            loss = criterion(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(triplet_loader)

        # --- Tính Separation Gap ---
        model.eval()
        intra_sims, inter_sims = [], []
        with torch.no_grad():
            for anchor, positive, negative in triplet_loader:
                emb_a = nn.functional.normalize(model(anchor.to(device)),   p=2, dim=1)
                emb_p = nn.functional.normalize(model(positive.to(device)), p=2, dim=1)
                emb_n = nn.functional.normalize(model(negative.to(device)), p=2, dim=1)
                intra_sims.extend((emb_a * emb_p).sum(dim=1).cpu().numpy())
                inter_sims.extend((emb_a * emb_n).sum(dim=1).cpu().numpy())
                if len(intra_sims) >= 100:
                    break

        gap = float(np.mean(intra_sims) - np.mean(inter_sims))
        history["loss"].append(avg_loss)
        history["gap"].append(gap)
        print(f"  Epoch [{epoch+1:>2}/{epochs}]  Loss: {avg_loss:.4f}  Gap: {gap:.4f}")

    total_time = time.time() - t0
    num_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Xong {model_name}! Thời gian: {total_time:.1f}s | Params: {num_params/1e6:.2f}M")

    # Lưu trọng số mô hình
    save_path = f"{model_name}_triplet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Đã lưu model → {save_path}")

    return history, total_time, num_params, gap


# ============================================================
# 4. VẼ BIỂU ĐỒ VÀ IN BẢNG KẾT QUẢ
# ============================================================
def plot_and_summarize(all_histories, results_summary, epochs):
    colors = {"mobilenet_v2": "#1f77b4", "resnet50": "#ff7f0e", "efficientnet_b0": "#2ca02c"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = range(1, epochs + 1)

    for name, hist in all_histories.items():
        ax1.plot(x, hist["loss"], marker="o", color=colors.get(name), label=name.upper())
        ax2.plot(x, hist["gap"],  marker="s", linestyle="--", color=colors.get(name), label=name.upper())

    ax1.set_title("Biến thiên Loss theo Epoch", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Triplet Loss"); ax1.legend()

    ax2.set_title("Separation Gap theo Epoch", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Gap (Intra − Inter)"); ax2.legend()

    plt.tight_layout()
    plt.savefig("model_comparison_curves.png", dpi=300)
    print("\nĐã lưu biểu đồ → model_comparison_curves.png")
    plt.show()

    df = pd.DataFrame(results_summary)
    print("\n" + "="*55)
    print(" BẢNG SO SÁNH HIỆU NĂNG MÔ HÌNH")
    print("="*55)
    print(df.to_string(index=False))
    df.to_csv("model_comparison_results.csv", index=False)
    print("\nĐã lưu bảng → model_comparison_results.csv")


# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    # Xây dựng DataLoader (bỏ comment dòng dưới nếu chạy độc lập)
    triplet_loader = build_dataloader(DATASET_PATH, batch_size=BATCH_SIZE)

    # Nếu bạn muốn dùng dataloader đã có từ notebook, thay bằng:
    # from your_notebook_setup import dataloader as triplet_loader

    all_histories   = {}
    results_summary = []

    for model_name in MODELS_TO_TEST:
        history, duration, params, final_gap = train_and_evaluate(
            model_name, triplet_loader, epochs=EPOCHS
        )
        all_histories[model_name] = history
        results_summary.append({
            "Model Architecture":   model_name.upper(),
            "Parameters (M)":       round(params / 1e6, 2),
            "Train Time (seconds)": round(duration, 1),
            "Final Separation Gap": round(final_gap, 4),
        })

    plot_and_summarize(all_histories, results_summary, EPOCHS)
