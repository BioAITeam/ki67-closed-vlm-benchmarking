import csv
import sys
import math
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def calculate_metrics(csv_path: str) -> None:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

    pred_col_candidates = ["predict", "predicted", "pred"]
    pred_col = next((c for c in pred_col_candidates if c in fieldnames), None)
    if pred_col is None or "true" not in fieldnames:
        print("Column names not found.\n"
              f"    Found columns: {fieldnames}\n"
               "    Need one of {predict, predicted, pred} and 'true'.")
        sys.exit(1)

    y_true, y_pred = [], []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                y_true.append(float(row["true"]))
                y_pred.append(float(row[pred_col]))
            except (ValueError, KeyError):
                continue

    if not y_true:
        print(" No valid rows found - check your CSV content.")
        sys.exit(1)

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print("Metrics")
    print(f"R²   : {r2:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python 4.utils/calculate_metrics.py <results.csv>\n"

            "Example BCData:\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/4.5/BCData/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/4.1-mini-2025-04-14/BCData/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/4.1-2025-04-14/BCData/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/4o/BCData/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/gemini1.5pro/BCData/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/gemini1.5flash/BCData/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/grok2vision/BCData/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/claude-3-5-sonnet/BCData/ki67_results.csv\n"

            "Example SHIDC-B-Ki-67:\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/4.5/SHIDC-B-Ki-67/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/4.1-mini-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/4.1-2025-04-14/SHIDC-B-Ki-67/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/4o/SHIDC-B-Ki-67/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/gemini1.5pro/SHIDC-B-Ki-67/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/gemini1.5flash/SHIDC-B-Ki-67/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/grok2vision/SHIDC-B-Ki-67/ki67_results.csv\n"
            "  python 4.utils/calculate_metrics.py "
            "5.results/claude-3-5-sonnet/SHIDC-B-Ki-67/ki67_results.csv"
        )
        sys.exit(1)

    calculate_metrics(sys.argv[1])