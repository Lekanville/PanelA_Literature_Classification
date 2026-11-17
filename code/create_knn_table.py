from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter

from create_cv_table import RUNS


WEIGHTING_LABELS = {
    "uniform": "Uniform",
    "inverse_distance": "Inverse distance",
    "inverse_distance_squared": "Inverse distance squared",
    "exponential_similarity": "Exponential similarity",
    "rank": "Rank-based decay",
}
METRIC_COLUMNS = (
    ("multiclass_auc_roc", "Multi-class AUC-ROC"),
    ("test_accuracy", "Overall Test Accuracy"),
    ("macro_recall", "Macro Recall"),
    ("macro_precision", "Macro Precision"),
    ("macro_f1", "Macro F1-Score"),
)
LOG_METRICS = {
    "macro_precision": r"IBA Overall Macro Precision: ([\d.]+)",
    "macro_recall": r"IBA Overall Macro Recall: ([\d.]+)",
    "macro_f1": r"IBA Overall Macro F1-Score: ([\d.]+)",
    "test_accuracy": r"IBA Overall Test Accuracy: ([\d.]+)",
    "multiclass_auc_roc": r"IBA Multi-class Macro ROC-AUC: ([\d.]+)",
}


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")

    # Logs can hold several runs; the last match is the most recent.
    config = re.findall(r"Champion Config Found: k=(\d+), weighting=(\w+)", text)
    if not config:
        raise ValueError(f"No k-NN champion configuration found in {path}")
    k, weighting = config[-1]
    result = {"k": int(k), "weighting": weighting}

    for name, pattern in LOG_METRICS.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise ValueError(f"Metric {name!r} not found in {path}")
        result[name] = float(matches[-1])
    return result


def load_knn_results(results_root: Path, log_dir: Path) -> list[dict]:
    rows = []
    for run_directory, plm_name in RUNS:
        saved_path = results_root / run_directory / "knn_test_results.csv"
        log_path = log_dir / f"{run_directory}.log"

        if saved_path.is_file():
            result = pd.read_csv(saved_path).iloc[0].to_dict()
        elif log_path.is_file():
            result = parse_log(log_path)
        else:
            raise FileNotFoundError(f"Neither {saved_path} nor {log_path} exists")

        weighting = str(result["weighting"])
        rows.append(
            {
                "Model Identifier": plm_name,
                "k": int(result["k"]),
                "weighting": WEIGHTING_LABELS.get(weighting, weighting),
                **{label: float(result[key]) for key, label in METRIC_COLUMNS},
            }
        )
    return rows


def write_table(rows: list[dict], output_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "kNN Results"

    headers = list(rows[0])
    border = Border(*(Side(style="thin", color="000000"),) * 4)
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=column, value=header)
        cell.font = Font(name="Arial", size=10, bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border

    for row_number, row in enumerate(rows, start=2):
        for column, header in enumerate(headers, start=1):
            value = row[header]
            if isinstance(value, float):
                value = f"{value:.2f}"
            cell = sheet.cell(row=row_number, column=column, value=value)
            cell.font = Font(name="Arial", size=10, italic=header == "Model Identifier")
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border

    sheet.column_dimensions["A"].width = 25
    sheet.column_dimensions["C"].width = 28
    for column in (2, *range(4, len(headers) + 1)):
        sheet.column_dimensions[get_column_letter(column)].width = 16
    sheet.sheet_view.showGridLines = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the k-NN held-out test table from saved results or run logs."
    )
    parser.add_argument("--results-root", type=Path, default=Path("results/all_embeddings_modelling"))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("."),
        help="Directory holding <run>.log files, used when knn_test_results.csv is absent.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("results/all_embeddings_modelling/knn_results_table.xlsx"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_knn_results(args.results_root, args.log_dir)
    write_table(rows, args.output_file)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved the k-NN table to {args.output_file}")


if __name__ == "__main__":
    main()
