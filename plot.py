import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_plot_style():
    """Configure the plot's font and color style."""
    try:
        font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        for font_path in font_paths:
            if 'Arial' in font_path:
                fm.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = 'Arial'
                break
    except Exception:
        # Fallback to default sans-serif if font search fails
        plt.rcParams['font.family'] = 'sans-serif'

    # Define color mapping
    color_map = {
        'adamw': '#2B95C3', 
        'paraadamw': '#6E6E6E',
        'sgd': '#009E73',
        'adam': '#0072B2'
    }
    return color_map

def plot_columns_comparison(csv_path1, csv_path2, column_name):
    """
    Reads specified column data from two CSV files, plots them with their original styles,
    and saves the chart in their directory.

    Args:
        csv_path1 (str): Path to the first CSV file (styled as AdamW).
        csv_path2 (str): Path to the second CSV file (styled as ParaAdamW).
        column_name (str): Name of the column to plot.
    """
    color_map = setup_plot_style()
    
    # Determine output path automatically based on the first CSV's location
    output_dir = os.path.dirname(os.path.abspath(csv_path1))
    output_filename = f"{column_name.lower()}_comparison.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Prepare metadata for the files to be plotted (reverted to original style)
    files_to_plot = [
        {'path': csv_path2, 'label': 'ParaAdamW', 'color_key': 'paraadamw'},
        {'path': csv_path1, 'label': 'AdamW', 'color_key': 'adamw'}
    ]

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Configure axes and grid lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DCDCDC')
    ax.spines['bottom'].set_color('#DCDCDC')
    ax.yaxis.grid(True, color='#EEEEEE', linestyle='-', linewidth=1)
    ax.xaxis.grid(False)

    # Configure tick and title font sizes
    ax.tick_params(axis='y', colors='#777777', length=0, labelsize=30)
    ax.tick_params(axis='x', colors='#777777', length=4, labelsize=30)
    
    # Set specific titles for loss and accuracy curves
    if column_name.lower() == 'train_loss':
        title = 'Training Loss'
    elif column_name.lower() in ['train_acc', 'final_test_accuracy']:
        title = 'Training Accuracy'
    else:
        # Format column name for display
        title = column_name.replace('_', ' ').title()
    
    ax.set_title(
        title, fontsize=32, fontweight='bold',
        color='#333333', pad=15
    )
    ax.set_xlabel('Iteration', loc='right', color='#777777', fontsize=31)

    legend_handles = []
    plot_has_data = False

    # Loop through, read, and plot data from each file
    for info in files_to_plot:
        try:
            df = pd.read_csv(info['path'])
            if 'Iter' not in df.columns or column_name not in df.columns:
                print(f"⚠️ Warning: File '{info['path']}' is missing 'Iter' or '{column_name}' column. Skipping.")
                continue
            
            df = df.dropna(subset=['Iter', column_name])
            if df.empty:
                print(f"⚠️ Warning: No valid data found in '{info['path']}'. Skipping.")
                continue

            plot_has_data = True
            x_data, y_data = df['Iter'], df[column_name]
            label = info['label']
            color = color_map.get(info['color_key'])
            
            line, = ax.plot(x_data, y_data, color=color, linewidth=2.5, label=label)
            ax.plot(x_data.iloc[-1], y_data.iloc[-1], 'o', markersize=3.5, color=color)
            legend_handles.append(line)

        except FileNotFoundError:
            print(f"❌ Error: File not found at '{info['path']}'.")
            return
        except Exception as e:
            print(f"❌ Error reading or plotting file '{info['path']}': {e}")
            continue

    if not plot_has_data:
        print("❌ Error: No valid data to plot from either file.")
        plt.close(fig)
        return

    # --- Legend Configuration (Reverted to original style) ---
    legend = ax.legend(
        handles=legend_handles, loc='upper right',
        bbox_to_anchor=(0.98, 0.98), frameon=False,
        ncol=1, fontsize=30,
        labelspacing=0.8
    )
    # Set legend text color to match its corresponding line color
    for text, handle in zip(legend.get_texts(), legend.legend_handles):
        text.set_color(handle.get_color())
    
    # Adjust Y-axis limits based on data
    y_min, y_max = ax.get_ylim()
    if y_min >= 0:  # For metrics like accuracy that should be between 0-1
        ax.set_ylim(bottom=0)
        if y_max <= 1:
            ax.set_ylim(top=1)
    
    fig.tight_layout()
    fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"✅ Plot created successfully! Image saved to: {output_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a comparison plot for a specified column from two CSV files. "
                    "The plot is saved automatically in the same directory as the input files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'csv_path1',
        type=str,
        help="Path to the first CSV file (will be labeled 'AdamW')."
    )
    parser.add_argument(
        'csv_path2',
        type=str,
        help="Path to the second CSV file (will be labeled 'ParaAdamW')."
    )
    parser.add_argument(
        'column_name',
        type=str,
        help="Name of the column to plot (e.g., Train_Loss, Train_Acc, final_test_accuracy, etc.)."
    )
    args = parser.parse_args()
    
    plot_columns_comparison(args.csv_path1, args.csv_path2, args.column_name)

if __name__ == "__main__":
    main()