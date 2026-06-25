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
