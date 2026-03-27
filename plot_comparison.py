import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path
import pandas as pd
import numpy as np
import os

def plot_model_comparison(
        csv_path_1: Path,
        csv_path_2: Path,
        label1: str = "Model A",
        label2: str = "Model B",
        target_list: list = None,  # List of dataset names to skip
        output_filename: str = "iou_comparison_plot.png"
):
    """
    Reads two dataset summary CSVs, filters out ignored datasets, and plots a comparison.
    """
    if target_list is None:
        target_list = []

    # 1. Load the data
    if not os.path.exists(csv_path_1) or not os.path.exists(csv_path_2):
        print(f"Error: Files not found.")
        return

    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    # FILTER: Remove datasets not in the target_list
    # ~df.isin() means "NOT in the list" (Use it for Monocolors, otherwise remove '~' to just consider Dualcolors)
    df1 = df1[~df1['dataset_split'].isin(target_list)]
    df2 = df2[~df2['dataset_split'].isin(target_list)]

    # 3. Merge data on dataset_split
    df = pd.merge(df1, df2, on='dataset_split', suffixes=(f'_{label1}', f'_{label2}'))

    if df.empty:
        logger.warning("No datasets left to plot after filtering.")
        return

    df = df.sort_values('dataset_split')

    # 4. Prepare Plot Data
    labels = df['dataset_split'].tolist()
    iou_1 = df[f'mean_IoU_{label1}'].tolist()
    iou_2 = df[f'mean_IoU_{label2}'].tolist()

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(20, 15))

    rects1 = ax.bar(x - width / 2, iou_1, width, label=label1, color='#3498db', edgecolor='white')
    rects2 = ax.bar(x + width / 2, iou_2, width, label=label2, color='#e74c3c', edgecolor='white')

    # Styling
    ax.set_ylabel('Mean $IoU$ Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Attach labels above bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    logger.info(f"Comparison plot saved at {output_filename}.")


if __name__ == "__main__":
    csv_path_1 = Path("/home/mehran/Git-Thesis/SparseInst/tools/output/Dual_NO_LORA_frozenBackbone_20260319_125500/inference_20260327_111720/Detectron2_dataset_summary.csv")
    csv_path_2 = Path("/home/mehran/Git-Thesis/SparseInst/tools/output/Dual_LORA_frozenBackbone_20260326_092110/inference_20260327_110252/Detectron2_dataset_summary.csv")
    target_datasets = [
        'blackpurple', 'blueblue', 'bluewhite', 'green_grey', 'grey_orange', 'pink_black', 'pinkblue',
        'red_blue', 'red_white_granules', 'yellow_grey',
    ]

    plot_model_comparison(
            csv_path_1=csv_path_1,
            csv_path_2=csv_path_2,
            label1="No_LoRA",
            label2="LoRA",
            target_list = target_datasets,
            output_filename="Monocolor_segmentation_comparison.png"
        )