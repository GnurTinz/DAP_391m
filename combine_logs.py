import os
import re
import numpy as np

def parse_log_file(filepath):
    metrics = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define regex patterns to extract values
    patterns = {
        'Closed_Rank1': r'Rank-1 Accuracy\s*:\s*([\d\.]+)%',
        'Closed_EER': r'EER\s*:\s*([\d\.]+)%\s*\(thresh',
        'Closed_Mean_Genuine': r'\[Closed-Set.*?Mean Genuine\s*:\s*([\-\d\.]+)',
        'Closed_Mean_Impostor': r'Mean Impostor\s*:\s*([\-\d\.]+)',
        'Open_Rank1': r'Open-Set Rank-1\s*:\s*([\d\.]+)%',
        'KL_Gate_EER': r'KL-Gate EER\s*:\s*([\d\.]+)%',
        'Uncertainty_EER': r'Uncertainty EER\s*:\s*([\d\.]+)%',
        'Stage1_FRR': r'Stage-1 FRR\s*:\s*([\d\.]+)%',
        'FAR_Stranger': r'FAR \(stranger\)\s*:\s*([\d\.]+)%',
        'FRR_Known': r'FRR \(known\)\s*:\s*([\d\.]+)%',
        'Open_EER': r'EER \(open-set\)\s*:\s*([\d\.]+)%',
        'Open_Mean_Genuine': r'\[Open-Set.*?Mean Genuine\s*:\s*([\-\d\.]+)',
        'Open_Mean_Stranger': r'Mean Stranger\s*:\s*([\-\d\.]+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.DOTALL)
        if match:
            metrics[key] = float(match.group(1))
            
    # As some names might clash, let's use a simpler line-by-line parsing
    metrics = {}
    lines = content.split('\n')
    current_section = None
    for line in lines:
        if '[Closed-Set' in line:
            current_section = 'Closed'
        elif '[Open-Set' in line:
            current_section = 'Open'
            
        if 'Rank-1 Accuracy :' in line and current_section == 'Closed':
            val = re.search(r'([\d\.]+)%', line).group(1)
            metrics['Closed_Rank1'] = float(val)
        elif 'EER             :' in line and current_section == 'Closed':
            val = re.search(r'([\d\.]+)%', line).group(1)
            metrics['Closed_EER'] = float(val)
        elif 'Mean Genuine    :' in line and current_section == 'Closed':
            val = re.search(r':\s*([\-\d\.]+)', line).group(1)
            metrics['Closed_Mean_Genuine'] = float(val)
        elif 'Mean Impostor   :' in line and current_section == 'Closed':
            val = re.search(r':\s*([\-\d\.]+)', line).group(1)
            metrics['Closed_Mean_Impostor'] = float(val)
            
        elif 'Open-Set Rank-1 :' in line and current_section == 'Open':
            val = re.search(r'([\d\.]+)%', line).group(1)
            metrics['Open_Rank1'] = float(val)
        elif 'KL-Gate EER     :' in line and current_section == 'Open':
            val = re.search(r'([\d\.]+)%', line).group(1)
            metrics['Open_KL_Gate_EER'] = float(val)
        elif 'Uncertainty EER :' in line and current_section == 'Open':
            val = re.search(r'([\d\.]+)%', line).group(1)
            metrics['Open_Uncertainty_EER'] = float(val)
        elif 'Stage-1 FRR     :' in line and current_section == 'Open':
            val = re.search(r'([\d\.]+)%', line).group(1)
            metrics['Open_Stage1_FRR'] = float(val)
        elif 'FAR (stranger)  :' in line and current_section == 'Open':
            val = re.search(r'([\d\.]+)%', line).group(1)
            metrics['Open_FAR_Stranger'] = float(val)
        elif 'FRR (known)     :' in line and current_section == 'Open':
            val = re.search(r'([\d\.]+)%', line).group(1)
            metrics['Open_FRR_Known'] = float(val)
        elif 'EER (open-set)  :' in line and current_section == 'Open':
            val = re.search(r'([\d\.]+)%', line).group(1)
            metrics['Open_EER'] = float(val)
        elif 'Mean Genuine    :' in line and current_section == 'Open':
            val = re.search(r':\s*([\-\d\.]+)', line).group(1)
            metrics['Open_Mean_Genuine'] = float(val)
        elif 'Mean Stranger   :' in line and current_section == 'Open':
            val = re.search(r':\s*([\-\d\.]+)', line).group(1)
            metrics['Open_Mean_Stranger'] = float(val)
            
    return metrics

def process_directory(base_dir):
    tasks_dir = os.path.join(base_dir, 'tasks')
    if not os.path.exists(tasks_dir):
        print(f"Directory {tasks_dir} does not exist.")
        return

    subdirs = [d for d in os.listdir(tasks_dir) if os.path.isdir(os.path.join(tasks_dir, d))]
    
    overall_results = {}
    
    for subdir in subdirs:
        subdir_path = os.path.join(tasks_dir, subdir)
        log_files = [f for f in os.listdir(subdir_path) if f.endswith('.log')]
        
        if not log_files:
            continue
            
        print(f"Processing {subdir} ({len(log_files)} logs)...")
        
        all_metrics = {}
        for log_file in log_files:
            metrics = parse_log_file(os.path.join(subdir_path, log_file))
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
                
        # Calculate mean and std
        summary = {}
        for k, v_list in all_metrics.items():
            summary[k] = {
                'mean': np.mean(v_list),
                'std': np.std(v_list)
            }
            
        overall_results[subdir] = summary
        
        # Write combine.txt in the subdir
        combine_path = os.path.join(subdir_path, 'combine.txt')
        with open(combine_path, 'w', encoding='utf-8') as f:
            f.write(f"Combined Results for {subdir} (from {len(log_files)} logs)\n")
            f.write("="*50 + "\n\n")
            for k, stats in summary.items():
                f.write(f"{k:25s}: Mean = {stats['mean']:.4f}, Std = {stats['std']:.4f}\n")
                
    # Also write a combined report in tasks/
    report_path = os.path.join(tasks_dir, 'combined_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Overall Combined Report for All Tasks\n")
        f.write("="*50 + "\n\n")
        
        for subdir, summary in overall_results.items():
            f.write(f"Task: {subdir}\n")
            f.write("-" * 30 + "\n")
            for k, stats in summary.items():
                f.write(f"{k:25s}: Mean = {stats['mean']:.4f}, Std = {stats['std']:.4f}\n")
            f.write("\n")
            
    print(f"Done! Combined results written to individual folders and {report_path}")

if __name__ == '__main__':
    process_directory('e:\\palm')
