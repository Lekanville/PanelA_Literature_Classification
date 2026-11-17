from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", message="X does not have valid feature names")

UOAS = tuple(range(1, 7))
EMBEDDING_COLUMNS = [f"corpus_embeddings_{index}" for index in range(768)]


def normalize_doi(value: object) -> str:
    if pd.isna(value):
        return ""
    doi = str(value).strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return doi.strip().casefold()


def load_historical_uoas(path: Path) -> dict[str, set[int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Historical submissions workbook not found: {path}")

    master_df = pd.read_excel(path, skiprows=4)
    required_columns = {"DOI", "Unit of assessment number"}
    missing_columns = required_columns.difference(master_df.columns)
    if missing_columns:
        raise ValueError(f"Historical workbook is missing columns: {sorted(missing_columns)}")

    master_df["_doi_key"] = master_df["DOI"].map(normalize_doi)
    master_df["_uoa"] = pd.to_numeric(
        master_df["Unit of assessment number"], errors="coerce"
    )
    master_df = master_df[
        master_df["_doi_key"].ne("") & master_df["_uoa"].isin(UOAS)
    ]

    return {
        doi: set(group.astype(int))
        for doi, group in master_df.groupby("_doi_key")["_uoa"]
    }


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    model_output_dir = args.model_output_dir
    test_path = args.test_dataset or model_output_dir / "test_dataset.csv"
    output_path = args.output_file or model_output_dir / "multilabel_precision.csv"

    if not test_path.is_file():
        raise FileNotFoundError(f"Held-out test dataset not found: {test_path}")

    test_df = pd.read_csv(test_path)
    unnamed_columns = [column for column in test_df.columns if column.startswith("Unnamed:")]
    test_df.drop(columns=unnamed_columns, inplace=True, errors="ignore")

    required_columns = {"DOI", "Unit_of_assessment_number", *EMBEDDING_COLUMNS}
    missing_columns = required_columns.difference(test_df.columns)
    if missing_columns:
        raise ValueError(f"Test dataset is missing columns: {sorted(missing_columns)}")

    test_uoas = pd.to_numeric(test_df["Unit_of_assessment_number"], errors="coerce")
    if test_uoas.isna().any():
        raise ValueError("Test dataset contains missing or non-numeric UoA labels")
    test_uoas = test_uoas.astype(int)
    uoas = sorted(test_uoas.unique().tolist())
    unsupported_uoas = set(uoas).difference(UOAS)
    if unsupported_uoas:
        raise ValueError(f"Test dataset contains unsupported UoAs: {sorted(unsupported_uoas)}")

    test_doi_keys = test_df["DOI"].map(normalize_doi).tolist()
    historical_uoas = load_historical_uoas(args.historical_file)
    embeddings = test_df[EMBEDDING_COLUMNS]
    results = []

    for uoa in uoas:
        model_path = model_output_dir / "model" / f"champion_ovr_model_uoa_{uoa}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved UoA {uoa} model not found: {model_path}")

        model = joblib.load(model_path)
        if not hasattr(model, "predict_proba"):
            raise TypeError(f"Saved model does not support predict_proba: {model_path}")

        class_labels = list(getattr(model, "classes_", []))
        if 1 not in class_labels:
            raise ValueError(f"Saved model has no positive class label 1: {model_path}")
        positive_class_index = class_labels.index(1)

        probabilities = model.predict_proba(embeddings)[:, positive_class_index]
        predicted_positive = probabilities > args.threshold
        predicted_positive_count = int(predicted_positive.sum())
        strict_true = test_uoas.eq(uoa).to_numpy()
        adjusted_true = np.fromiter(
            (uoa in historical_uoas.get(doi, set()) for doi in test_doi_keys),
            dtype=bool,
            count=len(test_doi_keys),
        )

        strict_precision = (
            float(np.count_nonzero(predicted_positive & strict_true))
            / predicted_positive_count
            if predicted_positive_count
            else 0.0
        )
        adjusted_precision = (
            float(np.count_nonzero(predicted_positive & adjusted_true))
            / predicted_positive_count
            if predicted_positive_count
            else 0.0
        )
        strict_precision = round(strict_precision, 2)
        adjusted_precision = round(adjusted_precision, 2)

        results.append(
            {
                "UoA": uoa,
                "Original Strict Precision": strict_precision,
                "Multi-Label Adjusted Precision": adjusted_precision,
                "Precision Direct Increase": round(
                    adjusted_precision - strict_precision, 2
                ),
            }
        )

    summary = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved evaluation summary to {output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare strict and historical multi-label UoA precision."
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        required=True,
        help="Run directory containing test_dataset.csv and the model/ folder.",
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        help="Optional held-out dataset path; defaults to <model-output-dir>/test_dataset.csv.",
    )
    parser.add_argument(
        "--historical-file",
        type=Path,
        default=Path("data/REF_all_submissions.xlsx"),
        help="REF submissions workbook used to build historical UoA associations.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional summary CSV path; defaults to <model-output-dir>/multilabel_precision.csv.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    evaluate(args)


if __name__ == "__main__":
    main()