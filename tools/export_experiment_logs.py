import os
import re
import yaml
import glob
from collections import defaultdict

def extract_info_from_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    model_name = "UnknownModel"
    config_path = "UnknownConfig"
    
    filename = os.path.basename(log_path)
    run_info = filename.replace('.log', '')

    # 2. Tìm model name và đường dẫn config trong nội dung log
    for line in lines:
        if "Reading model config from" in line:
            parts = line.split("Reading model config from ")
            if len(parts) > 1:
                config_path = parts[1].strip()
                path_parts = config_path.replace('\\', '/').split('/')
                for part in path_parts:
                    if part.startswith('UnetPalmModel') or part.startswith('baseline') or part.startswith('own_'):
                        model_name = part
                        break
            break

    # 3. Lấy seed (random_state) từ file config
    random_state = "Unknown"
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f_cfg:
                config_data = yaml.safe_load(f_cfg)
                if 'dataset' in config_data and 'seed' in config_data['dataset']:
                    random_state = config_data['dataset']['seed']
        except Exception as e:
            pass

    return config_path, run_info, random_state, lines

def main():
    tasks_dir = os.path.join(os.path.dirname(__file__), "..", "tasks")
    experiments_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
    
    os.makedirs(experiments_dir, exist_ok=True)
    
    log_files = glob.glob(os.path.join(tasks_dir, "*.log"))
    
    if not log_files:
        print(f"Không tìm thấy file log nào trong thư mục {tasks_dir}")
        return
        
    print(f"Tìm thấy {len(log_files)} file logs. Bắt đầu gom nhóm...")
    
    # Gom nhóm theo config_path (Cùng kịch bản model config)
    grouped_logs = defaultdict(list)
    for log_path in log_files:
        config_path, run_info, random_state, lines = extract_info_from_log(log_path)
        if config_path != "UnknownConfig":
            grouped_logs[config_path].append((run_info, random_state, lines))

    for config_path, runs in grouped_logs.items():
        # Tạo tên file hợp lệ từ đường dẫn config
        sanitized_name = re.sub(r'[\\/:*?"<>|]+', '_', config_path)
        # Lấy phần đuôi có ý nghĩa (ví dụ: UnetPalmModel_Resnet18_version_32_Original_KL1_version_32)
        parts = sanitized_name.split('_logs_')
        if len(parts) > 1:
            sanitized_name = parts[1].replace('_config_backup_yaml', '')
        
        output_filename = f"{sanitized_name}.log"
        output_path = os.path.join(experiments_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"============================================================\n")
            f_out.write(f"TỔNG HỢP KẾT QUẢ CHO CONFIG: {config_path}\n")
            # Lấy random state chung (hoặc lấy từ run đầu tiên)
            general_seed = runs[0][1] if runs else "Unknown"
            f_out.write(f"Random State (Seed) được chọn: {general_seed}\n")
            f_out.write(f"Số lượng file log được gộp: {len(runs)}\n")
            f_out.write(f"============================================================\n\n")
            
            for run_info, seed, lines in runs:
                f_out.write(f"\n{'='*80}\n")
                f_out.write(f"--- RUN INFO: {run_info} ---\n")
                f_out.write(f"{'='*80}\n")
                for line in lines:
                    f_out.write(line)
                    
        print(f"[SUCCESS] Đã gộp {len(runs)} logs vào file: {output_path}")

if __name__ == "__main__":
    main()
