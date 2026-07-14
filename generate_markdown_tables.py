import pandas as pd
import argparse

def df_to_markdown(df):
    fmt = ['---' for _ in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    return df_formatted.to_csv(sep='|', index=False)

def generate_markdown_tables(csv_path):
    df = pd.read_csv(csv_path)
    
    # Filter columns to show
    metrics_to_show = ['Closed_Rank1', 'Closed_EER', 'Open_EER', 'Open_AUROC', 'Open_OSCR']
    
    print("### 1. Intra-dataset Performance (Huấn luyện và kiểm thử trên cùng tập dữ liệu)\n")
    intra_df = df[df['Train'] == df['Test']].copy()
    intra_df['Dataset'] = intra_df['Train']
    intra_table = intra_df[['Dataset', 'Model', 'Mode', 'Method'] + metrics_to_show].sort_values(by=['Dataset', 'Model', 'Mode', 'Method'])
    print(df_to_markdown(intra_table))
    print("\n\n")
    
    print("### 2. Cross-dataset Performance (Hiệu suất chéo miền)\n")
    cross_df = df[df['Train'] != df['Test']].copy()
    cross_df['Setting'] = cross_df['Train'] + ' -> ' + cross_df['Test']
    cross_table = cross_df[['Setting', 'Model', 'Mode', 'Method'] + metrics_to_show].sort_values(by=['Setting', 'Model', 'Mode', 'Method'])
    print(df_to_markdown(cross_table))
    print("\n\n")

    print("### 3. Ablation Study: Tác động của Mode (Mode0 vs Mode3)\n")
    # For ablation, we can pivot the table to put mode0 and mode3 side-by-side
    ablation_cols = ['Train', 'Test', 'Model', 'Method', 'Mode', 'Open_OSCR', 'Closed_Rank1']
    ablation_df = df[ablation_cols].copy()
    ablation_df['Setting'] = ablation_df['Train'] + ' -> ' + ablation_df['Test']
    
    pivot_df = ablation_df.pivot_table(
        index=['Setting', 'Model', 'Method'], 
        columns=['Mode'], 
        values=['Closed_Rank1', 'Open_OSCR'], 
        aggfunc='first'
    ).reset_index()
    
    # Flatten multi-level columns
    pivot_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in pivot_df.columns.values]
    pivot_df = pivot_df.sort_values(by=['Setting', 'Model', 'Method'])
    print(df_to_markdown(pivot_df))
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate markdown tables from summary CSV')
    parser.add_argument('--csv', type=str, default='tasks/summary_table.csv', help='Path to summary_table.csv')
    args = parser.parse_args()
    
    generate_markdown_tables(args.csv)
