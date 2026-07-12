import os
import glob
import re

experiments_dir = "e:/palm/experiments"
log_files = glob.glob(os.path.join(experiments_dir, "*.log"))

output_lines = []
output_lines.append("| Model | Dataset | Mode & Strategy | Rank-1 (Closed) | EER (Closed) | Rank-1 (Open) | EER (Open) | FAR (Stranger) | FRR (Known) | Stage-1 FRR |")
output_lines.append("|---|---|---|---|---|---|---|---|---|---|")

for log_path in log_files:
    filename = os.path.basename(log_path).replace(".log", "")
    
    # Extract model name
    model = "Unknown"
    if "Resnet18" in filename or "resnet" in filename.lower(): model = "ResNet18"
    elif "ccnet" in filename.lower(): model = "CCNet"
    elif "Palmnet" in filename or "palmnet" in filename.lower(): model = "PalmNet"
    
    # Simplify dataset name
    if "Original" in filename: dataset = "Original"
    elif "Tongji" in filename and "own" not in filename: dataset = "Tongji"
    elif "IITD" in filename and "own" not in filename: dataset = "IITD"
    elif "own_iitd_tongji" in filename: dataset = "Combined (Own+IITD+Tongji)"
    else: dataset = filename[:20] + "..."
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    runs = content.split("--- RUN INFO:")
    
    for run in runs[1:]:
        lines = run.split('\n')
        run_info = lines[0].strip().replace("---", "").strip()
        
        # Regex to simplify run_info like eval_mode1_opt_proj_real_20260711_210543
        mode_match = re.search(r'mode(\d+)_(.*?)_20', run_info)
        mode_str = run_info
        if mode_match:
            mode_str = f"Mode {mode_match.group(1)} ({mode_match.group(2)})"
            
        rank1_closed = "-"
        eer_closed = "-"
        rank1_open = "-"
        eer_open = "-"
        far = "-"
        frr = "-"
        stage1_frr = "-"
        
        for line in lines:
            if "Rank-1 Accuracy :" in line:
                rank1_closed = line.split(":")[-1].strip()
            elif "EER" in line and "thresh=" in line and "KL-Gate" not in line and "Uncertainty" not in line and "open-set" not in line:
                eer_closed = line.split(":")[-1].strip().split()[0]
            elif "Open-Set Rank-1 :" in line:
                rank1_open = line.split(":")[-1].strip()
            elif "EER (open-set)" in line:
                eer_open = line.split(":")[-1].strip().split()[0]
            elif "FAR (stranger)" in line:
                far = line.split(":")[-1].strip().split()[0]
            elif "FRR (known)" in line:
                frr = line.split(":")[-1].strip().split()[0]
            elif "Stage-1 FRR" in line:
                stage1_frr = line.split(":")[-1].strip().split()[0]
                
        # Only add to table if there are actual results
        if rank1_closed != "-" or rank1_open != "-":
            output_lines.append(f"| {model} | {dataset} | {mode_str} | {rank1_closed} | {eer_closed} | {rank1_open} | {eer_open} | {far} | {frr} | {stage1_frr} |")

with open(os.path.join(experiments_dir, "summary_table.md"), 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))
    
print("Summary table generated at e:/palm/experiments/summary_table.md")
