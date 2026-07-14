import os
import csv
import glob
import re

def parse_report_file(filepath):
    # Returns a dict mapping task_name -> dict of metrics -> "mean ± std"
    results = {}
    current_task = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("Task: "):
            current_task = line.replace("Task: ", "").strip()
            results[current_task] = {}
        elif current_task and ":" in line and "Mean =" in line and "Std =" in line:
            # E.g.: Closed_Rank1             : Mean = 8.5867, Std = 2.3953
            parts = line.split(":")
            metric_name = parts[0].strip()
            val_part = parts[1]
            
            mean_match = re.search(r'Mean\s*=\s*([\-\d\.]+)', val_part)
            std_match = re.search(r'Std\s*=\s*([\-\d\.]+)', val_part)
            
            if mean_match and std_match:
                mean_val = mean_match.group(1)
                std_val = std_match.group(1)
                results[current_task][metric_name] = f"{mean_val} ± {std_val}"
                
    return results

def main():
    base_dir = r"e:\palm\tasks"
    report_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip 'tmp' folder if any
        if 'tmp' in dirs:
            dirs.remove('tmp')
            
        if 'combined_report.txt' in files:
            report_files.append(os.path.join(root, 'combined_report.txt'))
    
    if not report_files:
        print("No combined_report.txt found!")
        return

    all_data = []
    all_metrics_keys = set()
    
    for filepath in report_files:
        # Determine category from the path (the folder directly under e:\palm\tasks)
        # filepath: e:\palm\tasks\tasks_unet_ccnet_checkpoints_baseline\tasks\combined_report.txt
        rel_path = os.path.relpath(filepath, base_dir)
        parts = rel_path.split(os.sep)
        category = parts[0] if len(parts) > 0 else "Unknown"
        
        # Extract model from category (e.g., tasks_unet_ccnet_checkpoints_baseline -> unet_ccnet)
        model = category
        m = re.search(r'tasks_(.*?)_checkpoint', category)
        if m:
            model = m.group(1)
            
        task_results = parse_report_file(filepath)
        
        for task_name, metrics in task_results.items():
            if 'tmp' in task_name.lower():
                continue
                
            # Parse task_name, e.g., "mode0_iitd_iitd_newmetric"
            task_parts = task_name.split('_')
            mode = task_parts[0] if len(task_parts) > 0 else ""
            
            if mode == 'tmp':
                continue
                
            train_db = task_parts[1] if len(task_parts) > 1 else ""
            test_db = task_parts[2] if len(task_parts) > 2 else ""
            
            row = {
                "Category": category,
                "Model": model,
                "Mode": mode,
                "Train": train_db,
                "Test": test_db,
                "Task": task_name
            }
            row.update(metrics)
            all_metrics_keys.update(metrics.keys())
            all_data.append(row)
            
    # Define a preferred order for standard metrics if available
    preferred_order = [
        "Closed_Rank1", "Closed_EER", "Closed_Mean_Genuine", "Closed_Mean_Impostor",
        "Open_Rank1", "Open_KL_Gate_EER", "Open_Uncertainty_EER", 
        "Open_Stage1_FRR", "Open_FAR_Stranger", "Open_FRR_Known", "Open_EER",
        "Open_Mean_Genuine", "Open_Mean_Stranger", "Open_AUROC", "Open_DIR_1",
        "Open_DIR_01", "Open_OSCR"
    ]
    
    sorted_metrics = []
    for p in preferred_order:
        if p in all_metrics_keys:
            sorted_metrics.append(p)
            
    # Add any remaining metrics that were not in preferred_order
    for k in sorted(list(all_metrics_keys)):
        if k not in sorted_metrics:
            sorted_metrics.append(k)
    
    fieldnames = ["Category", "Model", "Mode", "Train", "Test", "Task"] + sorted_metrics
    
    output_csv = os.path.join(base_dir, "summary_table.csv")
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_data:
            writer.writerow(row)
            
    print(f"Generated CSV report with {len(all_data)} rows at {output_csv}")

if __name__ == "__main__":
    main()
