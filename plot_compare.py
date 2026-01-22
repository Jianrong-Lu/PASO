import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker

def setup_plot_style():
    """
    Configure the plot's font and color style.
    """
    try:
        font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        for font_path in font_paths:
            if 'Arial' in font_path:
                fm.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = 'Arial'
                break
    except Exception:
        plt.rcParams['font.family'] = 'sans-serif'

    # Color Palette Definitions
    color_map = {
        'sgd': '#1F77B4', 'adam': '#1F77B4', 'adamw': '#1F77B4', 
        'serial': '#1F77B4', 'baseline': '#1F77B4',
        'parasgd': '#FF7F0E', 'paraadam': '#FF7F0E', 
        'paraadamw': '#FF7F0E', 'paso': '#FF7F0E',
        'naivedp': '#2CA02C', 'ddp': '#9467BD', 'deepspeed': '#D62728',
    }
    return color_map

def get_legend_settings(file_path):
    path_lower = file_path.lower()
    if 'parallel' in path_lower: return 'PASO', 'paso'
    if 'naive_dp' in path_lower: return 'Naive DP', 'naivedp'
    if 'ddp' in path_lower: return 'DDP', 'ddp'
    if 'deepspeed' in path_lower: return 'DeepSpeed', 'deepspeed'
    
    if 'serial' in path_lower:
        if 'adamw' in path_lower: return 'AdamW', 'adamw'
        elif 'sgd' in path_lower: return 'SGD', 'sgd'
        elif 'adam' in path_lower: return 'Adam', 'adam'
        else: return 'Serial', 'serial'
    return 'Method', 'baseline'

def generate_smart_filename(file_path, column_name, cur_type):
    abs_path = os.path.abspath(file_path)
    root, _ = os.path.splitext(abs_path)
    parts = root.split(os.sep)
    selected_parts = parts[-3:] if len(parts) >= 3 else parts
    prefix = "_".join(selected_parts)
    clean_col = column_name.replace(' ', '_')
    return f"{prefix}_{clean_col}_{cur_type}.png"

def plot_columns_comparison(args):
    csv_path1, csv_path2, column_name, x_interval, y_interval = args.csv_path1, args.csv_path2, args.column_name, args.x_interval, args.y_interval
    color_map = setup_plot_style()
    
    output_dir = os.path.dirname(os.path.abspath(csv_path2))
    output_filename = generate_smart_filename(csv_path2, column_name, args.type)
    output_path = os.path.join(output_dir, output_filename)
    
    label1, color_key1 = get_legend_settings(csv_path1)
    label2, color_key2 = get_legend_settings(csv_path2)
    
    files_to_plot = [
        {'path': csv_path2, 'label': label2, 'color_key': color_key2},
        {'path': csv_path1, 'label': label1, 'color_key': color_key1}
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DCDCDC')
    ax.spines['bottom'].set_color('#DCDCDC')
    ax.yaxis.grid(True, color='#EEEEEE', linestyle='-', linewidth=1)
    ax.tick_params(axis='both', colors='#333333', labelsize=34)
    
    is_fid = column_name.lower() == 'fid'

    if is_fid:
        target_x_col = 'milestone'
        xlabel_text = 'Iteration ($\\times 10^4$)'
    elif args.type in ["ppl_tokens", "loss_tokens"]:
        target_x_col = 'num_tokens'
        xlabel_text = 'Cumulative Tokens (M)'
    else:
        target_x_col = 'Iter'
        xlabel_text = 'Iteration ($\\times 10^4$)'

    if target_x_col == 'num_tokens':
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1_000_000:.1f}'))
    else:
        divisor = x_interval if x_interval else 10000
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/divisor:.0f}'))

    if y_interval:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_interval))
    if x_interval:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))

    ax.set_title(column_name.replace('_', ' ').title(), fontsize=34, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel_text, loc='right', fontsize=34)

    legend_handles = []
    plot_has_data = False
    std_milestones = [10000, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000]

    for info in files_to_plot:
        try:
            df = pd.read_csv(info['path'], on_bad_lines='skip')
            
            if target_x_col not in df.columns:
                df[target_x_col] = range(1, len(df) + 1)

            # Modified FID logic: Only attempt mapping if real_iters is provided
            if is_fid and 'parallel' in info['path'].lower() and args.real_iters is not None:
                if len(args.real_iters) == len(std_milestones):
                    mapping = dict(zip(std_milestones, args.real_iters))
                    df['milestone'] = df['milestone'].map(mapping)
                else:
                    print(f"⚠️ Warning: real_iters length does not match std_milestones for {info['path']}")

            if column_name in df.columns:
                if df[column_name].dtype == object:
                    df[column_name] = df[column_name].astype(str).str.replace(r'tensor\(', '', regex=True).str.replace(r'\)', '', regex=True)
                df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
                df[target_x_col] = pd.to_numeric(df[target_x_col], errors='coerce')

            df = df.dropna(subset=[target_x_col, column_name])
            if df.empty: continue

            plot_has_data = True
            line, = ax.plot(df[target_x_col], df[column_name], 
                            color=color_map.get(info['color_key'].lower(), '#000000'), 
                            linewidth=4.0, label=info['label'])
            ax.plot(df[target_x_col].iloc[-1], df[column_name].iloc[-1], 'o', markersize=5.0, color=line.get_color())
            legend_handles.append(line)

        except Exception as e:
            print(f"❌ Error processing '{info['path']}': {e}")

    if not plot_has_data:
        plt.close(fig)
        return

    legend = ax.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=34)
    for text, handle in zip(legend.get_texts(), legend.legend_handles):
        text.set_color(handle.get_color())
    
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Experimental Training Curve Comparison Tool")
    parser.add_argument('csv_path1', type=str, help="Path to baseline CSV")
    parser.add_argument('csv_path2', type=str, help="Path to experiment CSV")
    parser.add_argument('--column_name', type=str, required=True, help="Header of the CSV column")
    parser.add_argument('--type', type=str, default='loss_iter', help="Metric type")
    parser.add_argument('--x_interval', type=float, help="X-axis tick interval")
    parser.add_argument('--y_interval', type=float, help="Y-axis tick interval")
    # real_iters is now truly optional with a default of None
    parser.add_argument('--real_iters', type=int, default=None, nargs='+', help="Real iteration mapping (FID only)")
    
    args = parser.parse_args()
    plot_columns_comparison(args)

if __name__ == "__main__":
    main()