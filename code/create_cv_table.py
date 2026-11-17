from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter


RUNS = (
    ("all-mpnet_all_embeddings", "all-mpnet-base-v2"),
    ("Bio_ClinicalBERT_all_embeddings", "Bio_ClinicalBERT"),
    ("BioMedBert_all_embeddings", "BioMedBERT"),
    ("pubmedbert-base-embeddings_all_embeddings", "PubMedBERT-embed"),
)
ALGORITHM_LABELS = {
    "LinearSVC": "LinearSVC",
    "LightGBM": "LightGBM",
    "KerasMLP": "MLP",
}
UOAS = tuple(range(1, 7))
RESULT_COLUMNS = {"uoa", "algorithm", "cv_mean_auc", "cv_std_auc"}


def load_cv_results(results_root: Path) -> dict[tuple[int, str, str], tuple[float, float]]:
    scores: dict[tuple[int, str, str], tuple[float, float]] = {}

    for run_directory, plm_name in RUNS:
        path = results_root / run_directory / "modelling_results.xlsx"
        if not path.is_file():
            raise FileNotFoundError(f"Model results workbook not found: {path}")

        results = pd.read_excel(path)
        missing_columns = RESULT_COLUMNS.difference(results.columns)
        if missing_columns:
            raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

        results["uoa"] = pd.to_numeric(results["uoa"], errors="coerce")
        results["cv_mean_auc"] = pd.to_numeric(results["cv_mean_auc"], errors="coerce")
        results["cv_std_auc"] = pd.to_numeric(results["cv_std_auc"], errors="coerce")
        if results[["uoa", "cv_mean_auc", "cv_std_auc"]].isna().any().any():
            raise ValueError(f"{path} contains missing or non-numeric CV results")

        for row in results.itertuples(index=False):
            uoa = int(row.uoa)
            algorithm = ALGORITHM_LABELS.get(row.algorithm)
            if uoa not in UOAS:
                raise ValueError(f"Unexpected UoA {uoa} in {path}")
            if algorithm is None:
                raise ValueError(f"Unexpected algorithm {row.algorithm!r} in {path}")

            key = (uoa, plm_name, algorithm)
            if key in scores:
                raise ValueError(f"Duplicate CV result for UoA {uoa}, {plm_name}, {algorithm}")
            scores[key] = (float(row.cv_mean_auc), float(row.cv_std_auc))

    expected = {
        (uoa, plm_name, algorithm)
        for uoa in UOAS
        for _, plm_name in RUNS
        for algorithm in ALGORITHM_LABELS.values()
    }
    missing_results = expected.difference(scores)
    if missing_results:
        raise ValueError(f"Missing CV results, for example: {sorted(missing_results)[:5]}")

    return scores


def write_table(scores: dict[tuple[int, str, str], tuple[float, float]], output_path: Path) -> None:
    plm_names = [plm_name for _, plm_name in RUNS]
    algorithm_names = list(ALGORITHM_LABELS.values())
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "CV AUC Table"

    headers = ["Unit of Assessment", "PLM", *algorithm_names]
    thin_border = Border(
        left=Side(style="thin", color="000000"),
        right=Side(style="thin", color="000000"),
        top=Side(style="thin", color="000000"),
        bottom=Side(style="thin", color="000000"),
    )
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=column, value=header)
        cell.font = Font(name="Arial", size=10, bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    row_number = 2
    for uoa in UOAS:
        first_row = row_number
        best_scores = {
            algorithm: max(
                round(scores[(uoa, plm_name, algorithm)][0], 2)
                for plm_name in plm_names
            )
            for algorithm in algorithm_names
        }

        for plm_name in plm_names:
            plm_cell = sheet.cell(row=row_number, column=2, value=plm_name)
            plm_cell.font = Font(name="Arial", size=10, italic=True)
            plm_cell.alignment = Alignment(horizontal="center", vertical="center")
            plm_cell.border = thin_border

            for column, algorithm in enumerate(algorithm_names, start=3):
                mean_auc, std_auc = scores[(uoa, plm_name, algorithm)]
                cell = sheet.cell(
                    row=row_number,
                    column=column,
                    value=f"{mean_auc:.2f} ({std_auc:.2f})",
                )
                cell.font = Font(
                    name="Arial",
                    size=10,
                    bold=round(mean_auc, 2) == best_scores[algorithm],
                )
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = thin_border

            row_number += 1

        last_row = row_number - 1
        sheet.merge_cells(start_row=first_row, start_column=1, end_row=last_row, end_column=1)
        uoa_cell = sheet.cell(row=first_row, column=1, value=f"UoA {uoa}")
        uoa_cell.font = Font(name="Arial", size=10, bold=True)
        uoa_cell.alignment = Alignment(horizontal="right", vertical="center")
        uoa_cell.border = thin_border
        for merged_row in range(first_row, last_row + 1):
            sheet.cell(row=merged_row, column=1).border = thin_border

    sheet.column_dimensions["A"].width = 23
    sheet.column_dimensions["B"].width = 25
    for column in range(3, len(headers) + 1):
        sheet.column_dimensions[get_column_letter(column)].width = 18
    sheet.freeze_panes = "C2"
    sheet.sheet_view.showGridLines = False
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{sheet.max_row}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a publication-style table from saved nested-CV results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/all_embeddings_modelling"),
        help="Directory containing one results subdirectory per PLM.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("results/all_embeddings_modelling/cv_auc_table.xlsx"),
        help="Path for the combined Excel table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_cv_results(args.results_root)
    write_table(scores, args.output_file)

    check = load_workbook(args.output_file, read_only=True, data_only=True)
    print(f"Saved the CV table to {args.output_file}")
    print(f"Table rows (including header): {check.active.max_row}")


if __name__ == "__main__":
    main()