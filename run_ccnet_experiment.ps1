# Chạy huấn luyện mạng U-Net với Backbone CCNet
# Cấu hình cụ thể:
# - model: unet_ccnet
# - losses: arcface
# - loss_schedules: contrastive_first
# - dataset: own_original

python tools/train_lightning.py model=unet_ccnet losses=arcface loss_schedules=contrastive_first dataset=own_original
