import pandas as pd
from pathlib import Path
import re

def _extract_video_num(file_name: str) -> int | None:
    """
    Extract the trailing 4-digit video number from file_name like CNUH_DC04_BPB1_0052 -> 52.
    """
    if not isinstance(file_name, str):
        return None
    m = re.search(r'_(\d{4})$', file_name.strip())
    return int(m.group(1)) if m else None

def compute_stats(csv_path: str, data_list_path: str, output_path: str | None = None):
    # Read annotations
    df = pd.read_csv(csv_path)

    # Determine train/test file sets from annotations before merging
    video_nums_tmp = df["file_name"].apply(_extract_video_num)
    test_mask_tmp = (video_nums_tmp >= 51) & (video_nums_tmp <= 75)
    test_mask_tmp = test_mask_tmp.fillna(False)
    train_files = set(df.loc[~test_mask_tmp, "file_name"].unique())
    test_files = set(df.loc[test_mask_tmp, "file_name"].unique())

    # Read patient info (handle UTF-8 with BOM safely) and keep needed columns
    patients = pd.read_csv(data_list_path, encoding="cp949", dtype={"file_name": str})
    keep_cols = ["file_name", "patient_age", "patient_sex", "patient_ht", "patient_wt"]
    patients = patients[[c for c in keep_cols if c in patients.columns]].copy()

    # Coerce numeric patient fields
    for c in ("patient_age", "patient_ht", "patient_wt"):
        if c in patients.columns:
            patients[c] = pd.to_numeric(patients[c], errors="coerce")

    # Drop NA patients and compute dropped counts by split
    na_cols = [c for c in ("patient_age", "patient_sex", "patient_ht", "patient_wt") if c in patients.columns]
    patients_clean = patients.dropna(subset=na_cols) if na_cols else patients.copy()
    dropped_set = set(patients["file_name"]) - set(patients_clean["file_name"])
    dropped_patients_train = len(train_files & dropped_set)
    dropped_patients_test = len(test_files & dropped_set)
    dropped_patients_overall = dropped_patients_train + dropped_patients_test
    patients = patients_clean  # use cleaned patients for merge

    # Merge patient info
    df = df.merge(patients, on="file_name", how="left")

    # Identify test videos: numbered 51–75 inclusive (e.g., 0051–0075)
    df["video_num"] = df["file_name"].apply(_extract_video_num)
    test_mask = (df["video_num"] >= 51) & (df["video_num"] <= 75)
    test_mask = test_mask.fillna(False)

    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    # Helper to compute mean/std for numeric columns (exclude helper column)
    def numeric_summary(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        numeric_cols = d.select_dtypes(include="number").columns.tolist()
        if "video_num" in numeric_cols:
            numeric_cols.remove("video_num")
        if not numeric_cols:
            # No numeric columns; return empty series
            return pd.Series(dtype=float), pd.Series(dtype=float)
        means = d[numeric_cols].mean()
        stds = d[numeric_cols].std(ddof=1)
        return means, stds

    train_means, train_stds = numeric_summary(train_df)
    test_means, test_stds = numeric_summary(test_df)
    overall_means, overall_stds = numeric_summary(df)

    # Combine to a single summary DataFrame
    summary = pd.DataFrame({
        "train_mean": train_means,
        "train_std": train_stds,
        "test_mean": test_means,
        "test_std": test_stds,
        "overall_mean": overall_means,
        "overall_std": overall_stds,
    })
    summary["metric"] = summary.index
    summary = summary.reset_index(drop=True)
    summary = summary[["metric", "train_mean", "train_std", "test_mean", "test_std", "overall_mean", "overall_std"]]

    # Add patient_sex ratios per split (M/F) and dropped patient counts
    def sex_ratio(d: pd.DataFrame):
        if "patient_sex" not in d.columns:
            return {"M": 0.0, "F": 0.0}
        s = d["patient_sex"].astype(str).str.strip().str.upper()
        vc = s[s.isin(["M", "F"])].value_counts(normalize=True)
        return {"M": float(vc.get("M", 0.0)), "F": float(vc.get("F", 0.0))}

    ratios_train = sex_ratio(train_df)
    ratios_test = sex_ratio(test_df)
    ratios_overall = sex_ratio(df)

    sex_rows = pd.DataFrame([
        {
            "metric": "patient_sex_M_ratio",
            "train_mean": ratios_train["M"],
            "train_std": pd.NA,
            "test_mean": ratios_test["M"],
            "test_std": pd.NA,
            "overall_mean": ratios_overall["M"],
            "overall_std": pd.NA,
        },
        {
            "metric": "patient_sex_F_ratio",
            "train_mean": ratios_train["F"],
            "train_std": pd.NA,
            "test_mean": ratios_test["F"],
            "test_std": pd.NA,
            "overall_mean": ratios_overall["F"],
            "overall_std": pd.NA,
        },
        {
            "metric": "num_videos",
            "train_mean": len(train_df),
            "train_std": pd.NA,
            "test_mean": len(test_df),
            "test_std": pd.NA,
            "overall_mean": len(df),
            "overall_std": pd.NA,
        },
        {
            "metric": "num_patients_dropped",
            "train_mean": dropped_patients_train,
            "train_std": pd.NA,
            "test_mean": dropped_patients_test,
            "test_std": pd.NA,
            "overall_mean": dropped_patients_overall,
            "overall_std": pd.NA,
        },
    ])

    summary = pd.concat([summary, sex_rows], ignore_index=True)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)

    return summary

if __name__ == "__main__":
    csv_file = "/home/nistring/object-detection-project/data/SUIT/demo/visualized_annotations_edited/annotation_statistics.csv"
    data_list_file = "/home/nistring/object-detection-project/data/SUIT/demo/visualized_annotations_edited/data_list.csv"
    out_file = "/home/nistring/object-detection-project/data/SUIT/demo/visualized_annotations_edited/annotation_statistics_summary.csv"
    stats = compute_stats(csv_file, data_list_file, out_file)