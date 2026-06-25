import torch
import torchvision.utils as vutils
import os

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
            # Truyền temperature để tăng cường sự khác biệt giữa các biến thể (nếu model hỗ trợ)
            if 'temperature' in self.model.forward.__code__.co_varnames:
                out = self.model(x_expanded, decode=True, temperature=temperature)
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
                    # Mẫu giống (Nhiễu chuẩn N(0,1) * std)
                    eps = torch.randn_like(std)
                    z_list.append(mu + eps * std)
                else:
                    # Mẫu khác (Nhiễu khuếch đại cực đại, ép mô hình lệch khỏi cụm identity)
                    eps = torch.randn_like(std)
                    temperature = 4.0 + i  # Tăng dần độ nhiễu
                    z_list.append(mu + eps * std * temperature)
                    
            z_batch = torch.cat(z_list, dim=0)
            
            # 3. Giải mã từ z_batch
            if hasattr(self.model, 'use_decoder') and not self.model.use_decoder:
                raise ValueError("Mô hình không có decoder.")
                
            if hasattr(self.model, 'inc'): # Nếu là U-Net, phải truyền lại skip-connections
                # Nhân bản skip connections
                x_expanded = x_single.expand(num_images, -1, -1, -1).clone()
                x1 = self.model.inc(x_expanded)
                x2 = self.model.down1(x1)
                x3 = self.model.down2(x2)
                
                z_dec = self.model.fc_dec(z_batch)
                z_dec = z_dec.view(-1, 512, self.model.bottleneck_size, self.model.bottleneck_size)
                
                u1 = self.model.up1(z_dec, x3)
                u2 = self.model.up2(u1, x2)
                u3 = self.model.up3(u2, x1)
                x_hat = self.model.outc(u3)
                
                import torch.nn.functional as F
                if list(x_hat.shape[-2:]) != list(self.model.image_size):
                    x_hat = F.interpolate(x_hat, size=tuple(self.model.image_size), mode='bilinear', align_corners=False)
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
