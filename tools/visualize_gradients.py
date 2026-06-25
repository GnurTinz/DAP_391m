import os
import sys
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Ensure the root project directory is in the sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.palm_model import ProbabilisticPalmModel
from src.losses.custom import KLDivLoss, ReconstructionLoss, SupConLoss

def plot_grad_flow(named_parameters, save_path="gradient_flow.png"):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    '''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is not None:
                # Thu gọn tên layer cho dễ nhìn trên biểu đồ
                short_name = n.replace('encoder.', 'enc.') \
                              .replace('decoder.', 'dec.') \
                              .replace('verifier.', 'ver.')
                layers.append(short_name)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
            else:
                print(f"Warning: No gradient for {n}")
                
    plt.figure(figsize=(16, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="red")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.8, lw=1, color="blue")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="black" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=8)
    plt.xlim(left=0, right=len(ave_grads))
    
    # Scale Y axis flexibly
    max_val = max(max_grads) if max_grads else 0.02
    plt.ylim(bottom = -0.001, top=max_val * 1.1) 
    
    plt.xlabel("Layers", fontsize=12, fontweight='bold')
    plt.ylabel("Gradient Magnitude", fontsize=12, fontweight='bold')
    plt.title("Gradient Flow Across Network Architecture", fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend([plt.Line2D([0], [0], color="red", lw=4),
                plt.Line2D([0], [0], color="blue", lw=4),
                plt.Line2D([0], [0], color="black", lw=4)], 
               ['Max-gradient', 'Mean-gradient', 'Zero-gradient'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Gradient flow plot saved successfully at: {os.path.abspath(save_path)}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Gradient Flow of the Model")
    parser.add_argument('--config', type=str, default='config/mnist.yaml')
    parser.add_argument('--output', type=str, default='implement-idea/gradient_flow.png')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("1. Initializing model...")
    model_config = config.get('model', {})
    if 'decoder' not in model_config:
        model_config['decoder'] = {}
    model_config['decoder']['image_size'] = config.get('dataset', {}).get('image_size', [32, 32])
    
    model = ProbabilisticPalmModel(model_config)
    model.use_decoder = True
    model.train()

    print("2. Creating Dummy Data...")
    img_size = model_config['decoder']['image_size']
    dummy_x = torch.randn(4, 3, img_size[0], img_size[1]) # batch_size=4
    dummy_y = torch.tensor([0, 0, 1, 1])
    
    print("3. Forward pass...")
    recon_loss_fn = ReconstructionLoss({})
    kl_loss_fn = KLDivLoss({})
    supcon_loss_fn = SupConLoss({'temperature': 0.1})

    out = model(dummy_x, decode=True)
    
    L_rec = recon_loss_fn(dummy_x, out['x_hat'])
    L_kl = kl_loss_fn(out['mu'], out['logvar'])
    L_con = supcon_loss_fn(out['proj'], dummy_y)
    
    total_loss = L_rec + 0.01 * L_kl + 0.5 * L_con
    
    print("4. Backward pass (Calculating Gradients)...")
    model.zero_grad()
    total_loss.backward()

    print("5. Generating Graph...")
    plot_grad_flow(model.named_parameters(), save_path=args.output)

if __name__ == '__main__':
    main()
