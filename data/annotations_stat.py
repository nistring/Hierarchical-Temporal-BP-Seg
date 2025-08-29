import pandas as pd
from pathlib import Path

def compute_stats(csv_path: str, output_path: str | None = None):
    df = pd.read_csv(csv_path)
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include="number")

    means = numeric_df.mean()
    stds = numeric_df.std(ddof=1)  # sample std

    stats_df = pd.DataFrame({
        "mean": means,
        "std": stds
    }).reset_index().rename(columns={"index": "metric"})

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(output_path, index=False)

    return stats_df

if __name__ == "__main__":
    csv_file = "/home/nistring/object-detection-project/data/SUIT/demo/visualized_annotations_edited/annotation_statistics.csv"
    out_file = "/home/nistring/object-detection-project/data/SUIT/demo/visualized_annotations_edited/annotation_statistics_summary.csv"
    stats = compute_stats(csv_file, out_file)