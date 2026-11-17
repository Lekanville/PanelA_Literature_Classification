from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


UOAS = tuple(range(1, 7))
REQUIRED_SOURCE_COLUMNS = {
    "Unit of assessment number",
    "Output type",
    "Title",
    "DOI",
}
RENAME_COLUMNS = {
    "Institution name": "Institution_name",
    "Unit of assessment number": "Unit_of_assessment_number",
    "Output type": "Output_type",
    "Number of additional authors": "Number_of_additional_authors",
    "Journal title": "Journal_title",
    "Altmetric score": "Altmetric_score",
}
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def normalize_doi(value: object) -> str:
    if pd.isna(value):
        return ""
    doi = str(value).strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return doi.strip().casefold()


def extract_abstract(payload: dict) -> str:
    response = payload.get("abstracts-retrieval-response", {})
    item = response.get("item", {}) if isinstance(response, dict) else {}
    bibrecord = item.get("bibrecord", {}) if isinstance(item, dict) else {}
    head = bibrecord.get("head", {}) if isinstance(bibrecord, dict) else {}
    abstract_data = head.get("abstracts") if isinstance(head, dict) else None
    if abstract_data is None and isinstance(response, dict):
        coredata = response.get("coredata", {})
        if isinstance(coredata, dict):
            abstract_data = coredata.get("dc:description")

    paragraphs: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                paragraphs.append(text)
        elif isinstance(value, list):
            for entry in value:
                collect(entry)
        elif isinstance(value, dict):
            for key, entry in value.items():
                if key in {"ce:para", "para", "paragraph", "#text", "dc:description"}:
                    collect(entry)
                elif key == "abstract":
                    collect(entry)
                elif isinstance(entry, (dict, list)):
                    collect(entry)

    collect(abstract_data)
    return re.sub(r"\s+", " ", " ".join(paragraphs)).strip()


def fetch_abstract(
    session: requests.Session,
    doi: str,
    api_key: str,
    timeout: float,
    max_attempts: int,
) -> dict:
    url = f"https://api.elsevier.com/content/abstract/doi/{quote(doi, safe='')}"

    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(
                url,
                params={"apiKey": api_key},
                headers={"accept": "application/json"},
                timeout=timeout,
            )
        except requests.RequestException as error:
            if attempt == max_attempts:
                return {
                    "doi": doi,
                    "abstract": "",
                    "status": "request_error",
                    "attempts": attempt,
                    "error": type(error).__name__,
                }
            time.sleep(min(60, 2 ** (attempt - 1)))
            continue

        status_code = response.status_code
        if status_code in {401, 403}:
            raise RuntimeError(
                f"Elsevier API authentication failed (HTTP {status_code}); "
                "check SCOPUS_API_KEY."
            )
        if status_code in RETRYABLE_STATUS_CODES:
            if attempt == max_attempts:
                return {
                    "doi": doi,
                    "abstract": "",
                    "status": f"http_{status_code}",
                    "attempts": attempt,
                    "error": f"HTTP {status_code}",
                }
            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = float(retry_after)
            except ValueError:
                delay = min(60, 2 ** (attempt - 1))
            time.sleep(max(0, min(60, delay)))
            continue
        if status_code >= 400:
            return {
                "doi": doi,
                "abstract": "",
                "status": f"http_{status_code}",
                "attempts": attempt,
                "error": f"HTTP {status_code}",
            }

        try:
            payload = response.json()
        except ValueError:
            return {
                "doi": doi,
                "abstract": "",
                "status": "invalid_json",
                "attempts": attempt,
                "error": "Response was not valid JSON",
            }

        abstract = extract_abstract(payload) if isinstance(payload, dict) else ""
        return {
            "doi": doi,
            "abstract": abstract,
            "status": "success" if abstract else "empty_abstract",
            "attempts": attempt,
            "error": "" if abstract else "No abstract in API response",
        }

    raise RuntimeError("Abstract retrieval ended unexpectedly")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_api_cache(path: Path) -> dict[str, dict]:
    cached: dict[str, dict] = {}
    if not path.exists():
        return cached

    with path.open(encoding="utf-8") as cache_file:
        for line in cache_file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = normalize_doi(record.get("doi", ""))
            if doi:
                cached[doi] = record
    return cached


def append_cache_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as cache_file:
        cache_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        cache_file.flush()


def prepare_sample(
    input_path: Path,
    output_dir: Path,
    sample_per_uoa: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    source = pd.read_excel(input_path, skiprows=4)
    missing_columns = REQUIRED_SOURCE_COLUMNS.difference(source.columns)
    if missing_columns:
        raise ValueError(f"Source workbook is missing columns: {sorted(missing_columns)}")

    if "Article_id" not in source.columns:
        source.insert(0, "Article_id", source.index)

    source["_uoa"] = pd.to_numeric(source["Unit of assessment number"], errors="coerce")
    panel_a = source[source["_uoa"].isin(UOAS)].copy()
    journal_articles = panel_a[
        panel_a["Output type"].astype("string").str.strip().str.upper().eq("D")
    ].copy()
    journal_articles["_doi_key"] = journal_articles["DOI"].map(normalize_doi)
    journal_articles["Unit of assessment number"] = journal_articles["_uoa"].astype(int)
    journal_articles.drop(columns=["_uoa"], inplace=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / "journal_articles.csv"
    journal_articles.drop(columns=["_doi_key"]).rename(columns=RENAME_COLUMNS).to_csv(
        journal_path, index=False
    )

    valid_doi = journal_articles[journal_articles["_doi_key"].ne("")].copy()
    doi_groups = valid_doi.groupby("_doi_key", sort=False)
    registry_records = []
    for doi_key, group in doi_groups:
        uoas = sorted(group["Unit of assessment number"].astype(int).unique().tolist())
        for uoa in uoas:
            registry_records.append(
                {
                    "DOI": doi_key,
                    "Unit_of_assessment_number": uoa,
                    "is_multi_uoa": len(uoas) > 1,
                    "submission_count_for_doi": len(group),
                }
            )
    registry = pd.DataFrame(
        registry_records,
        columns=[
            "DOI",
            "Unit_of_assessment_number",
            "is_multi_uoa",
            "submission_count_for_doi",
        ],
    )
    registry.to_csv(output_dir / "doi_uoa_registry.csv", index=False)

    primary = valid_doi.drop_duplicates("_doi_key", keep="first").copy()
    primary["DOI"] = primary["_doi_key"]
    primary["Unit of assessment number"] = primary[
        "Unit of assessment number"
    ].astype(int)
    primary.drop(columns=["_doi_key"], inplace=True)
    primary.rename(columns=RENAME_COLUMNS, inplace=True)
    primary.to_csv(output_dir / "unique_articles.csv", index=False)

    selected_groups = []
    for uoa in UOAS:
        group = primary[primary["Unit_of_assessment_number"].eq(uoa)].reset_index(drop=True)
        if len(group) < sample_per_uoa:
            raise ValueError(
                f"UoA {uoa} has only {len(group)} unique DOI records; "
                f"cannot sample {sample_per_uoa}."
            )
        rng = np.random.default_rng(seed)
        selected_positions = rng.choice(len(group), size=sample_per_uoa, replace=False)
        selected_groups.append(group.iloc[selected_positions])

    sample = pd.concat(selected_groups, ignore_index=True)
    sample.to_csv(output_dir / "sampled_articles.csv", index=False)

    doi_uoa_counts = (
        primary["Unit_of_assessment_number"].value_counts().sort_index().to_dict()
    )
    report = {
        "input_file": str(input_path),
        "panel_a_uoas": list(UOAS),
        "journal_article_rows": int(len(journal_articles)),
        "journal_article_rows_by_uoa": {
            str(key): int(value)
            for key, value in journal_articles[
                "Unit of assessment number"
            ].value_counts().sort_index().items()
        },
        "missing_or_blank_doi_rows": int(journal_articles["_doi_key"].eq("").sum()),
        "unique_dois": int(valid_doi["_doi_key"].nunique()),
        "duplicate_doi_rows_removed": int(len(valid_doi) - valid_doi["_doi_key"].nunique()),
        "dois_submitted_to_multiple_uoas": int(
            doi_groups["Unit of assessment number"].nunique().gt(1).sum()
        ),
        "unique_articles_by_primary_uoa": {
            str(key): int(value) for key, value in doi_uoa_counts.items()
        },
        "primary_uoa_selection": "first occurrence in source workbook order",
        "sample_per_uoa": sample_per_uoa,
        "random_seed": seed,
        "sampled_rows": int(len(sample)),
        "sampled_rows_by_uoa": {
            str(key): int(value)
            for key, value in sample["Unit_of_assessment_number"]
            .value_counts()
            .sort_index()
            .items()
        },
        "abstracts_retrieved": None,
        "abstract_retrieval_failures": None,
    }
    write_json(output_dir / "preprocessing_report.json", report)
    return sample, report


def retrieve_sample_abstracts(
    sample: pd.DataFrame,
    output_dir: Path,
    final_output: Path,
    api_key: str,
    request_delay: float,
    timeout: float,
    max_attempts: int,
    retry_failed: bool,
    report: dict,
) -> None:
    cache_path = output_dir / "abstract_api_cache.jsonl"
    cached = load_api_cache(cache_path)
    abstracts: list[str] = []
    result_records: list[dict] = []

    with requests.Session() as session:
        for index, row in enumerate(sample.itertuples(index=False), start=1):
            doi = normalize_doi(getattr(row, "DOI"))
            cached_record = cached.get(doi)
            if cached_record and (cached_record.get("abstract") or not retry_failed):
                result = cached_record
            else:
                if request_delay:
                    time.sleep(request_delay)
                result = fetch_abstract(
                    session=session,
                    doi=doi,
                    api_key=api_key,
                    timeout=timeout,
                    max_attempts=max_attempts,
                )
                result["retrieved_at"] = datetime.now(timezone.utc).isoformat()
                append_cache_record(cache_path, result)
                cached[doi] = result

            abstracts.append(str(result.get("abstract", "") or ""))
            result_records.append(result)
            if index % 100 == 0 or index == len(sample):
                retrieved = sum(bool(value.strip()) for value in abstracts)
                print(f"Processed {index}/{len(sample)} sampled DOIs; abstracts present: {retrieved}")

    completed = sample.copy()
    completed["Abstract"] = abstracts
    has_abstract = completed["Abstract"].astype("string").str.strip().ne("")
    final_data = completed[has_abstract].copy()

    failure_rows = [
        {
            "DOI": record.get("doi", ""),
            "status": record.get("status", "unknown"),
            "attempts": record.get("attempts", 0),
            "error": record.get("error", ""),
        }
        for record in result_records
        if not str(record.get("abstract", "") or "").strip()
    ]
    pd.DataFrame(
        failure_rows, columns=["DOI", "status", "attempts", "error"]
    ).to_csv(output_dir / "abstract_failures.csv", index=False)

    final_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = final_output.with_name(final_output.name + ".tmp")
    final_data.to_csv(temporary_output, index=False)
    os.replace(temporary_output, final_output)

    report["abstracts_retrieved"] = int(len(final_data))
    report["abstract_retrieval_failures"] = int(len(failure_rows))
    report["final_output"] = str(final_output)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "preprocessing_report.json", report)
    print(
        f"Saved {len(final_data)} rows with abstracts to {final_output}; "
        f"{len(failure_rows)} sampled abstracts were unavailable."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate REF Panel A journal articles and retrieve abstracts."
    )
    parser.add_argument("--input-file", type=Path, default=Path("data/REF_all_submissions.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ref_preprocessing"))
    parser.add_argument("--final-output", type=Path, default=Path("data/REF_Titles_and_Abstracts.csv"))
    parser.add_argument("--sample-per-uoa", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Create the filtered, deduplicated and sampled files without calling Elsevier.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.sample_per_uoa <= 0:
        raise ValueError("--sample-per-uoa must be positive")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.request_delay < 0:
        raise ValueError("--request-delay cannot be negative")

    sample, report = prepare_sample(
        input_path=args.input_file,
        output_dir=args.output_dir,
        sample_per_uoa=args.sample_per_uoa,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))

    if args.sample_only:
        print(f"Sample-only run complete. API input saved to {args.output_dir / 'sampled_articles.csv'}")
        return

    api_key = os.environ.get("SCOPUS_API_KEY")
    if not api_key:
        raise RuntimeError("SCOPUS_API_KEY is required for abstract retrieval.")

    retrieve_sample_abstracts(
        sample=sample,
        output_dir=args.output_dir,
        final_output=args.final_output,
        api_key=api_key,
        request_delay=args.request_delay,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        retry_failed=args.retry_failed,
        report=report,
    )


if __name__ == "__main__":
    main()