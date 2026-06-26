import torch
import torch.nn.functional as F
import numpy as np

# Thử import các metric từ torchmetrics, nếu chưa có thì báo lỗi (nhưng user đã báo là đã cài)
try:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    print("Cảnh báo: Chưa cài đặt torchmetrics. Hãy chạy: pip install torchmetrics[image]")

class ImageEvaluator:
    def __init__(self, device='cpu'):
        self.device = device
        if HAS_TORCHMETRICS:
            self.psnr = PeakSignalNoiseRatio().to(device)
            # data_range=1.0 vì ta sẽ đưa ảnh về [0, 1] trước khi đo
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            try:
                # normalize=True giúp LPIPS nhận đầu vào [-1, 1] chuẩn xác
                self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
            except Exception as e:
                print(f"Cảnh báo: Lỗi khi load LPIPS (có thể thiếu model weight): {e}")
                self.lpips = None
        else:
            self.psnr = None
            self.ssim = None
            self.lpips = None

    def evaluate(self, real_image, generated_image):
        """
        Đánh giá các metric tái tạo hình ảnh.
        real_image, generated_image: tensors có shape (B, C, H, W), giá trị trong khoảng [-1, 1]
        """
        # Đưa ảnh về chuẩn [0, 1] cho tính toán PSNR/SSIM/MSE
        real_norm = (real_image + 1) / 2.0
        gen_norm = (generated_image + 1) / 2.0
        
        # Đảm bảo kẹp trong khoảng [0, 1] để tránh nhiễu do sai số float
        real_norm = torch.clamp(real_norm, 0.0, 1.0)
        gen_norm = torch.clamp(gen_norm, 0.0, 1.0)

        # Tính L1 (MAE) và L2 (MSE)
        mse = F.mse_loss(gen_norm, real_norm).item()
        mae = F.l1_loss(gen_norm, real_norm).item()
        
        psnr_val = None
        ssim_val = None
        lpips_val = None
        
        if HAS_TORCHMETRICS:
            psnr_val = self.psnr(gen_norm, real_norm).item()
            ssim_val = self.ssim(gen_norm, real_norm).item()
            if self.lpips is not None:
                # LPIPS ăn đầu vào dạng [-1, 1]
                lpips_val = self.lpips(generated_image, real_image).item()

        return {
            'MSE': mse,
            'MAE': mae,
            'PSNR': psnr_val,
            'SSIM': ssim_val,
            'LPIPS': lpips_val
        }

class LatentEvaluator:
    def __init__(self):
        pass

    def compute_latent_shift(self, mu_original, r_optimized):
        """
        Đo khoảng cách L2 (Euclidean) giữa vector r sau tối ưu và mu ban đầu (từ Encoder).
        Đại diện cho mức độ "dịch chuyển" trong không gian tiềm ẩn.
        """
        shift = torch.norm(r_optimized - mu_original, p=2).item()
        return shift

    def compute_eer(self, verifier, r, z_pos, z_neg):
        """
        Tính Equal Error Rate (EER) - Điểm cân bằng sai số.
        - r: Vector đại diện
        - z_pos: Mẫu Positive (đúng người)
        - z_neg: Mẫu Negative (sai người)
        """
        verifier.eval()
        with torch.no_grad():
            r_pos = r.expand(z_pos.size(0), -1)
            r_neg = r.expand(z_neg.size(0), -1)
            
            pos_scores = torch.sigmoid(verifier(z_pos, r_pos)).cpu().numpy().flatten()
            neg_scores = torch.sigmoid(verifier(z_neg, r_neg)).cpu().numpy().flatten()

        return self.compute_eer_from_scores(pos_scores, neg_scores)

    def compute_eer_from_scores(self, pos_scores, neg_scores):
        """
        Tính EER từ mảng điểm số cho trước. Hữu ích khi gộp điểm của nhiều Person.
        """
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return None, None

        thresholds = np.linspace(0.0, 1.0, 1000)
        eer_diffs = []
        
        for t in thresholds:
            far = np.sum(neg_scores >= t) / len(neg_scores)
            frr = np.sum(pos_scores < t) / len(pos_scores)
            eer_diffs.append(abs(far - frr))
            
        best_idx = np.argmin(eer_diffs)
        best_t = thresholds[best_idx]
        
        far_best = np.sum(neg_scores >= best_t) / len(neg_scores)
        frr_best = np.sum(pos_scores < best_t) / len(pos_scores)
        
        eer = (far_best + frr_best) / 2.0
        
        return eer, best_t
