import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.models.palm_model import ProbabilisticPalmModel
from src.models.unet_model import UNetPalmModel
from src.datasets import DatasetFactory
from src.engine.represent import optimize_representation
from src.metrics.evaluator import LatentEvaluator
from src.models.verifier import TestTimeVerifier

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    checkpoint_path = config.get('checkpoint', '')
    rep_cfg = config.get('represent', {})
    samples = rep_cfg.get('num_samples', 512)
    max_steps = rep_cfg.get('max_steps', 200)
    lr = rep_cfg.get('lr', 0.01)
    bce_threshold = rep_cfg.get('bce_threshold', 0.05)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    # 1. Tải checkpoint nếu có
    checkpoint_data = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location=device)

    # 2. Khởi tạo mô hình dựa vào config (ưu tiên config lưu trong checkpoint)
    model_config = config.get('model', {})
    if checkpoint_data and 'hyper_parameters' in checkpoint_data and 'model' in checkpoint_data['hyper_parameters']:
        model_config = checkpoint_data['hyper_parameters']['model']
        
    model_type = model_config.get('type', 'default')
    
    if 'decoder' not in model_config:
        model_config['decoder'] = {}
    if 'image_size' not in model_config['decoder']:
        model_config['decoder']['image_size'] = config.get('dataset', {}).get('image_size', [128, 128])
    
    if model_type == 'unet':
        model = UNetPalmModel(model_config).to(device)
    else:
        model = ProbabilisticPalmModel(model_config).to(device)
    
    # 3. Nạp trọng số từ checkpoint
    if checkpoint_data:
        state_dict = checkpoint_data.get('model_state_dict', checkpoint_data.get('state_dict', checkpoint_data))
        clean_state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
    else:
        print("Cảnh báo: Không có checkpoint hợp lệ.")

    # Load file Gallery
    if checkpoint_path:
        version_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    else:
        version_dir = "logs/unversioned_results"
        
    gallery_path = os.path.join(version_dir, 'gallery.pt')
    if not os.path.exists(gallery_path):
        print(f"Lỗi: Không tìm thấy file {gallery_path}! Hãy chạy build_gallery.py trước.")
        sys.exit(1)
        
    print(f"Đang tải Gallery r và Verifier từ {gallery_path}...")
    gallery_data = torch.load(gallery_path, map_location=device)
    gallery_r = gallery_data['gallery_r']
    verifier_state_dict = gallery_data['verifier_state_dict']
    
    # Khởi tạo lại Verifier
    latent_dim = next(iter(gallery_r.values())).size(1)
    shared_verifier = TestTimeVerifier(latent_dim, hidden_dim=128).to(device)
    if verifier_state_dict is not None:
        shared_verifier.load_state_dict(verifier_state_dict)
    shared_verifier.eval()

    # Khởi tạo Dataset
    dataset_config = config.get('dataset', {})
    data_dir = dataset_config.get('data_dir', 'data/collect')
    dataset_name = dataset_config.get('name', 'OwnOriginalDataset')
    
    print(f"\nĐang tải dữ liệu từ {dataset_name} ({data_dir})...")
    dataset = DatasetFactory.create(dataset_name, data_dir=data_dir, config=dataset_config, is_train=False)
    
    val_person_indices = {}
    if hasattr(dataset, 'samples') and len(dataset.samples) > 0:
        person_indices = {}
        for i, item in enumerate(dataset.samples):
            label = item[1]
            if label not in person_indices:
                person_indices[label] = []
            person_indices[label].append(i)
            
        for label, indices in person_indices.items():
            split_idx = max(1, len(indices) // 2)
            val_person_indices[label] = indices[split_idx:]
    else:
        val_person_indices = {0: [1]}
        
    print(f"Tổng số Person trên tập Val: {len([v for v in val_person_indices.values() if len(v)>0])}")

    # ==========================================
    # Trích xuất r trên tập Val (Probe)
    # ==========================================
    print("\n" + "="*50)
    print(" BẮT ĐẦU TRÍCH XUẤT PROBE TỪ TẬP VAL ")
    print("="*50)
    
    probe_r = {}
    all_pos_scores = []
    all_neg_scores = []
    
    # Để code gọn hơn, ta có thể ghi đè verbose=False trong optimize_r_from_latent.
    # Nhưng do đã dùng tqdm, ta sẽ không in dòng thông báo liên tục nữa.
    pbar = tqdm(val_person_indices.items(), desc="Xử lý Person")
    for person_id, indices in pbar:
        if len(indices) == 0:
            continue
            
        idx_to_use = indices[:2]
        if dataset is not None and hasattr(dataset, 'samples'):
            images = [dataset[idx][0] for idx in idx_to_use]
            batch_images = torch.stack(images).to(device)
        else:
            img_size = dataset_config.get('image_size', [128, 128])
            batch_images = torch.randn(len(idx_to_use), 3, img_size[0], img_size[1]).to(device)
            
        optimized_r, _, z_pos, z_neg = optimize_representation(
            model=model,
            image=batch_images,
            config=config,
            device=device,
            verifier=shared_verifier,
            freeze_net=True,
            num_samples=samples,
            max_steps=max_steps,
            lr=lr,
            bce_threshold=bce_threshold,
            pbar=pbar
        )
        
        probe_r[person_id] = optimized_r.cpu().detach()
        
        # Đánh giá EER
        with torch.no_grad():
            r_pos = optimized_r.expand(z_pos.size(0), -1).to(device)
            r_neg = optimized_r.expand(z_neg.size(0), -1).to(device)
            
            pos_scores = torch.sigmoid(shared_verifier(z_pos, r_pos)).cpu().numpy().flatten()
            neg_scores = torch.sigmoid(shared_verifier(z_neg, r_neg)).cpu().numpy().flatten()
            
            all_pos_scores.extend(pos_scores)
            all_neg_scores.extend(neg_scores)

    # ==========================================
    # Đánh giá Khớp r (Matching)
    # ==========================================
    print("\n" + "="*50)
    print(" KẾT QUẢ SO SÁNH VAL VÀ TRAIN (MATCHING) ")
    print("="*50)
    
    correct_matches = 0
    total_probes = len(probe_r)
    
    for probe_label, p_r in probe_r.items():
        best_match = None
        best_dist = float('inf')
        
        for gal_label, g_r in gallery_r.items():
            dist = torch.norm(p_r - g_r, p=2).item()
            if dist < best_dist:
                best_dist = dist
                best_match = gal_label
                
        is_correct = best_match == probe_label
        if is_correct:
            correct_matches += 1
        
    accuracy = (correct_matches / total_probes) * 100 if total_probes > 0 else 0
    print(f"=> Matching Accuracy (Rank-1): {accuracy:.2f}% ({correct_matches}/{total_probes})")
    
    # Tính EER
    latent_eval = LatentEvaluator()
    eer, best_thresh = latent_eval.compute_eer_from_scores(np.array(all_pos_scores), np.array(all_neg_scores))
    if eer is not None:
        print(f"=> Global Equal Error Rate (EER): {eer:.4f} (tại ngưỡng {best_thresh:.4f})")
    else:
        print("=> EER: Lỗi tính toán (thiếu mẫu).")

    print("\nHoàn tất toàn bộ chu trình đánh giá!")

if __name__ == '__main__':
    main()
