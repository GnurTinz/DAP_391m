import torch
import torchvision.utils as vutils
import os
import math

class ImageGenerator:
    """
    Lớp xử lý việc sinh (sample) và tái tạo (reconstruct) hình ảnh chung cho mọi mô hình.
    """
    def __init__(self, model, dataloader=None, device='cpu'):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.model.eval()

    def generate_unconditional(self, num_images=16, latent_dim=128, output_path='logs/generated_samples.png'):
        """
        Sinh ảnh vô điều kiện từ vector nhiễu ngẫu nhiên.
        Cảnh báo: Chỉ dùng cho các mạng sinh thuần túy không dựa vào skip-connection.
        """
        if hasattr(self.model, 'use_decoder') and not self.model.use_decoder:
            raise ValueError("Mô hình hiện tại không có decoder để sinh ảnh.")
            
        with torch.no_grad():
            z = torch.randn(num_images, latent_dim).to(self.device)
            # Đối với VAE cơ bản, gọi decoder trực tiếp
            if hasattr(self.model, 'decoder'):
                generated_images = self.model.decoder(z)
            else:
                raise ValueError("Mô hình không hỗ trợ sinh vô điều kiện (Ví dụ: U-Net cần ảnh đầu vào).")
            
            # Đưa pixel từ [-1, 1] về [0, 1]
            generated_images = (generated_images + 1) / 2.0
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            vutils.save_image(generated_images, output_path, nrow=int(num_images**0.5), padding=2, normalize=False)
            print(f"Đã lưu {num_images} ảnh sinh ngẫu nhiên vô điều kiện tại: {output_path}")

    def generate_reconstruction(self, num_images=8, output_path='logs/reconstructed.png'):
        """
        Sinh ảnh dựa trên việc nén và giải nén (Reconstruction) ảnh thật.
        Phù hợp cho cả U-Net và VAE tiêu chuẩn.
        """
        if self.dataloader is None:
            raise ValueError("Cần truyền dataloader để thực hiện Reconstruction.")
            
        # Lấy 1 batch
        batch_x, _ = next(iter(self.dataloader))
        
        # Giới hạn số lượng ảnh
        x = batch_x[:num_images].to(self.device)
        
        with torch.no_grad():
            out = self.model(x, decode=True)
            if 'x_hat' not in out:
                raise ValueError("Mô hình không trả về ảnh tái tạo (x_hat).")
                
            x_hat = out['x_hat']
            
            # Đưa giá trị pixel từ [-1, 1] về [0, 1]
            x_display = (x + 1) / 2.0
            x_hat_display = (x_hat + 1) / 2.0
            
            # Nối ảnh gốc (hàng trên) và ảnh tái tạo (hàng dưới)
            comparison = torch.cat([x_display, x_hat_display], dim=0)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            vutils.save_image(comparison, output_path, nrow=num_images, padding=2, normalize=False)
            print(f"Đã lưu ảnh so sánh Gốc - Tái tạo tại: {output_path}")

    def generate_variations(self, num_variations=8, temperature=1.0, output_path='logs/variations.png'):
        """
        Sinh các biến thể của MỘT ảnh duy nhất bằng cách khuếch đại nhiễu không gian ẩn (z).
        Đặc biệt hiệu quả với U-Net khi kết hợp với tham số temperature.
        """
        if self.dataloader is None:
            raise ValueError("Cần truyền dataloader để lấy ảnh gốc.")
            
        # Lấy 1 ảnh duy nhất từ batch đầu tiên
        batch_x, _ = next(iter(self.dataloader))
        x_single = batch_x[0:1].to(self.device)  # shape (1, C, H, W)
        
        # Nhân bản ảnh này lên `num_variations` lần
        # Việc này khiến Forward Pass sẽ sinh ra nhiều eps ngẫu nhiên khác nhau cho cùng 1 mu, logvar
        x_expanded = x_single.expand(num_variations, -1, -1, -1).clone()
        
        with torch.no_grad():
            # Truyền temperature và sample_mode
            if 'sample_mode' in self.model.forward.__code__.co_varnames:
                out = self.model(x_expanded, decode=True, temperature=temperature, sample_mode='stochastic')
            else:
                out = self.model(x_expanded, decode=True)
                
            if 'x_hat' not in out:
                raise ValueError("Mô hình không trả về ảnh tái tạo (x_hat).")
                
            x_hat = out['x_hat']
            
            # Đưa giá trị pixel từ [-1, 1] về [0, 1]
            x_single_display = (x_single + 1) / 2.0
            x_hat_display = (x_hat + 1) / 2.0
            
            # Chèn ảnh gốc vào đầu tiên để dễ so sánh
            comparison = torch.cat([x_single_display, x_hat_display], dim=0)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Lưu lại thành 1 hàng ngang: [Ảnh Gốc] | [Biến thể 1] | [Biến thể 2] | ...
            vutils.save_image(comparison, output_path, nrow=num_variations + 1, padding=2, normalize=False)
            print(f"Đã lưu {num_variations} biến thể của cùng một ảnh (temperature={temperature}) tại: {output_path}")

    def generate_contrastive(self, output_path='logs/contrastive.png'):
        """
        Lấy ra các biểu diễn giống (Positive) và khác (Negative) so với đầu vào (Anchor).
        Mục đích: Hiển thị trực quan việc mô hình tái tạo Anchor, Positive, và Negative.
        """
        if self.dataloader is None:
            raise ValueError("Cần truyền dataloader để lấy ảnh gốc.")
            
        anchor, positive, negative = None, None, None
        
        # Quét qua dataloader để tìm 1 bộ (Anchor, Positive, Negative)
        for batch_x, batch_y in self.dataloader:
            unique_labels = torch.unique(batch_y)
            if len(unique_labels) >= 2:
                # Lấy 1 class làm Anchor
                anchor_label = unique_labels[0].item()
                negative_label = unique_labels[1].item()
                
                anchor_indices = (batch_y == anchor_label).nonzero(as_tuple=True)[0]
                negative_indices = (batch_y == negative_label).nonzero(as_tuple=True)[0]
                
                if len(anchor_indices) >= 2 and len(negative_indices) >= 1:
                    anchor = batch_x[anchor_indices[0]].unsqueeze(0)
                    positive = batch_x[anchor_indices[1]].unsqueeze(0)
                    negative = batch_x[negative_indices[0]].unsqueeze(0)
                    break
                    
        if anchor is None:
            raise ValueError("Không tìm đủ số lượng ảnh cùng class và khác class trong 1 batch (hãy tăng batch_size lên).")
            
        # Gom lại thành 1 batch 3 ảnh: [Anchor, Positive, Negative]
        x = torch.cat([anchor, positive, negative], dim=0).to(self.device)
        
        with torch.no_grad():
            out = self.model(x, decode=True)
            x_hat = out['x_hat']
            
            x_display = (x + 1) / 2.0
            x_hat_display = (x_hat + 1) / 2.0
            
            # Xếp thành 2 hàng: Hàng 1 (Ảnh gốc), Hàng 2 (Ảnh tái tạo)
            comparison = torch.cat([x_display, x_hat_display], dim=0)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # nrow=3 tương ứng với [Anchor, Positive, Negative]
            vutils.save_image(comparison, output_path, nrow=3, padding=2, normalize=False)
            print(f"Đã lưu ảnh so sánh Anchor (Trái) - Positive (Giữa) - Negative (Phải) tại: {output_path}")

    def generate_from_latent(self, num_images=8, output_path='logs/latent_sampling.png'):
        """
        Trích xuất mu và sigma từ một ảnh đầu vào duy nhất.
        Sau đó tạo ra không gian mẫu bao gồm các biểu diễn GIỐNG (nhiễu nhỏ quanh mu) 
        và KHÁC (nhiễu lớn, chệch khỏi mu) để xem sức mạnh sinh từ Latent Space.
        """
        if self.dataloader is None:
            raise ValueError("Cần truyền dataloader để lấy ảnh gốc.")
            
        # Lấy 1 ảnh
        batch_x, _ = next(iter(self.dataloader))
        x_single = batch_x[0:1].to(self.device)
        
        with torch.no_grad():
            # 1. Forward 1 lần để lấy mu, logvar và skip-connections (nếu có)
            out_single = self.model(x_single, decode=True)
            mu = out_single['mu']
            logvar = out_single['logvar']
            
            # 2. Chuẩn bị các biến thể của z
            # - z0: Giống hệt (temperature = 0)
            # - z1 -> z_mid: Giống (temperature = 1.0)
            # - z_mid -> z_end: Khác (temperature = 3.0 -> 5.0) hoặc random hoàn toàn
            
            z_list = []
            std = torch.exp(0.5 * logvar)
            
            # Mẫu 1: Tái tạo chuẩn xác (z = mu)
            z_list.append(mu)
            
            # Các mẫu tiếp theo
            for i in range(1, num_images):
                if i < num_images // 2:
                    # 1. MẪU GIỐNG (POSITIVE / IN-DISTRIBUTION)
                    # Contrastive Loss ép mu thành cụm định danh. 
                    # Để lấy mẫu giống, ta chỉ lấy mẫu nội bộ trong bán kính sigma (độ bất định).
                    # Dùng temperature (tau) <= 1.0 để không văng khỏi cụm.
                    tau = 0.5 + (i * 0.2) # Tăng nhẹ độ đa dạng nhưng vẫn giữ danh tính
                    eps = torch.randn_like(std)
                    z_pos = mu + tau * eps * std
                    z_list.append(z_pos)
                else:
                    # Cách C: Xoay vector tiềm ẩn z (Latent Rotation / Spherical Interpolation)
                    # Góc xoay theta trải từ 45 độ (pi/4) đến 180 độ (pi) tuỳ theo i
                    idx = i - (num_images // 2)
                    max_idx = num_images - (num_images // 2)
                    theta = (math.pi / 4) + (math.pi * 3 / 4) * (idx / max_idx)
                    
                    # Tìm hướng ngẫu nhiên trực giao với mu để tạo mặt phẳng xoay
                    v = torch.randn_like(mu)
                    mu_norm = torch.norm(mu, p=2, dim=1, keepdim=True) + 1e-8
                    mu_unit = mu / mu_norm
                    
                    v_proj = torch.sum(v * mu_unit, dim=1, keepdim=True) * mu_unit
                    v_ortho = v - v_proj
                    v_ortho_unit = v_ortho / (torch.norm(v_ortho, p=2, dim=1, keepdim=True) + 1e-8)
                    
                    # Xoay mu trên mặt phẳng 2D (mu_unit, v_ortho_unit)
                    z_rot = mu_norm * (mu_unit * math.cos(theta) + v_ortho_unit * math.sin(theta))
                    
                    # Thêm chút nhiễu bất định để ảnh sinh ra trông thực tế
                    eps = torch.randn_like(std)
                    z_neg = z_rot + eps * std
                        
                    z_list.append(z_neg)
                    
            z_batch = torch.cat(z_list, dim=0)
            
            # 3. Giải mã từ z_batch
            if hasattr(self.model, 'use_decoder') and not self.model.use_decoder:
                raise ValueError("Mô hình không có decoder.")
                
            if hasattr(self.model, 'decode_from_z_and_x'): # Nếu là U-Net
                # Nhân bản skip connections
                x_expanded = x_single.expand(num_images, -1, -1, -1).clone()
                x_hat = self.model.decode_from_z_and_x(z_batch, x_expanded)
            else: # Nếu là VAE thường
                x_hat = self.model.decoder(z_batch)
                
            # Trực quan hóa
            x_single_display = (x_single + 1) / 2.0
            x_hat_display = (x_hat + 1) / 2.0
            
            # Pad thêm ảnh trắng để cân bằng hàng gốc (có 1 ảnh) với hàng sinh ra (có num_images ảnh)
            pad = torch.ones(num_images - 1, *x_single_display.shape[1:]).to(self.device)
            row1 = torch.cat([x_single_display, pad], dim=0)
            comparison = torch.cat([row1, x_hat_display], dim=0)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            vutils.save_image(comparison, output_path, nrow=num_images, padding=2, normalize=False)
            print(f"Đã lưu kết quả sinh từ Latent (Mu & Sigma) tại: {output_path}")

    def generate_average(self, num_images=8, temperature=1.0, output_path='logs/average.png'):
        """
        Lấy 2 ảnh, tính trung bình trong không gian Latent (mu_avg = (mu1 + mu2) / 2).
        Sau đó sinh ra ảnh chính xác từ mu_avg, và các biến thể xung quanh mu_avg.
        """
        if self.dataloader is None:
            raise ValueError("Cần truyền dataloader để lấy ảnh gốc.")
            
        # Lấy batch đầu tiên
        batch_x, _ = next(iter(self.dataloader))
        if batch_x.size(0) < 2:
            raise ValueError("Batch size phải >= 2 để lấy 2 ảnh.")
            
        x1 = batch_x[0:1].to(self.device)
        x2 = batch_x[1:2].to(self.device)
        
        with torch.no_grad():
            # Lấy latent representation của 2 ảnh
            out1 = self.model(x1, decode=True)
            out2 = self.model(x2, decode=True)
            
            mu1 = out1['mu']
            mu2 = out2['mu']
            logvar1 = out1['logvar']
            logvar2 = out2['logvar']
            
            # Tính trung bình không gian latent
            mu_avg = (mu1 + mu2) / 2.0
            logvar_avg = (logvar1 + logvar2) / 2.0
            std_avg = torch.exp(0.5 * logvar_avg)
            
            # Tạo danh sách các z
            z_list = [mu_avg] # Mẫu đầu tiên là trung bình chính xác (z = mu_avg)
            for _ in range(1, num_images):
                eps = torch.randn_like(std_avg)
                z_list.append(mu_avg + temperature * eps * std_avg)
                
            z_batch = torch.cat(z_list, dim=0)
            
            # Giải mã từ z_batch
            if hasattr(self.model, 'use_decoder') and not self.model.use_decoder:
                raise ValueError("Mô hình không có decoder.")
                
            if hasattr(self.model, 'decode_from_z_and_x'): # Nếu là U-Net
                x_avg = (x1 + x2) / 2.0
                x_expanded = x_avg.expand(num_images, -1, -1, -1).clone()
                x_hat = self.model.decode_from_z_and_x(z_batch, x_expanded)
            else: # Nếu là VAE thường
                x_hat = self.model.decoder(z_batch)
                
            # Trực quan hóa
            x1_display = (x1 + 1) / 2.0
            x2_display = (x2 + 1) / 2.0
            x_hat_display = (x_hat + 1) / 2.0
            
            # Tạo hàng 1 chứa [Ảnh 1] [Ảnh 2] và các ô trống (trắng) cho đủ num_images cột
            pad = torch.ones(num_images - 2, *x1_display.shape[1:]).to(self.device)
            if num_images > 2:
                row1 = torch.cat([x1_display, x2_display, pad], dim=0)
            else:
                row1 = torch.cat([x1_display, x2_display], dim=0)
                
            # Tạo hàng 2 chứa x_hat_display (Mẫu 1: Trung bình chính xác, Mẫu 2..n: Biến thể)
            comparison = torch.cat([row1, x_hat_display], dim=0)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            vutils.save_image(comparison, output_path, nrow=num_images, padding=2, normalize=False)
            print(f"Đã lưu kết quả sinh {num_images} ảnh trung bình & biến thể tại: {output_path}")

    def generate_interpolate(self, num_images=8, output_path='logs/interpolate.png'):
        """
        Nội suy (Interpolation) tuyến tính giữa 2 ảnh trong không gian Latent.
        Sinh ra một chuỗi ảnh chuyển đổi mượt mà từ Ảnh 1 sang Ảnh 2.
        """
        if self.dataloader is None:
            raise ValueError("Cần truyền dataloader để lấy ảnh gốc.")
            
        # Lấy batch đầu tiên
        batch_x, _ = next(iter(self.dataloader))
        if batch_x.size(0) < 2:
            raise ValueError("Batch size phải >= 2 để lấy 2 ảnh.")
            
        x1 = batch_x[0:1].to(self.device)
        x2 = batch_x[1:2].to(self.device)
        
        with torch.no_grad():
            # Lấy latent representation của 2 ảnh
            out1 = self.model(x1, decode=True)
            out2 = self.model(x2, decode=True)
            
            mu1 = out1['mu']
            mu2 = out2['mu']
            
            # Nội suy tuyến tính (Linear Interpolation) từ 0 đến 1
            z_list = []
            x_skip_list = [] # Dành cho U-Net skip connections
            alphas = torch.linspace(0, 1, steps=num_images).to(self.device)
            
            for alpha in alphas:
                # Nội suy mu
                z_t = (1 - alpha) * mu1 + alpha * mu2
                z_list.append(z_t)
                
                # Nội suy skip-connections (pixel-level)
                x_t = (1 - alpha) * x1 + alpha * x2
                x_skip_list.append(x_t)
                
            z_batch = torch.cat(z_list, dim=0)
            x_skip_batch = torch.cat(x_skip_list, dim=0)
            
            # Giải mã từ z_batch
            if hasattr(self.model, 'use_decoder') and not self.model.use_decoder:
                raise ValueError("Mô hình không có decoder.")
                
            if hasattr(self.model, 'decode_from_z_and_x'): # Nếu là U-Net
                x_hat = self.model.decode_from_z_and_x(z_batch, x_skip_batch)
            else: # Nếu là VAE thường
                x_hat = self.model.decoder(z_batch)
                
            # Trực quan hóa
            x_hat_display = (x_hat + 1) / 2.0
            
            # Lưu thành 1 hàng duy nhất thể hiện quá trình chuyển đổi
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            vutils.save_image(x_hat_display, output_path, nrow=num_images, padding=2, normalize=False)
            print(f"Đã lưu kết quả nội suy (Interpolation) giữa 2 ảnh tại: {output_path}")
