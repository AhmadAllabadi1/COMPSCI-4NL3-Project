#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SUBMISSION_REQUIRED_COLUMNS = {"id", "label"}
TRUTH_REQUIRED_COLUMNS = {"id", "label"}


def normalize_label(value: str) -> str:
    return str(value).strip().upper()


def resolve_submission_csv(res_dir: Path) -> Path:
    if not res_dir.exists():
        raise FileNotFoundError(f"Submission directory does not exist: {res_dir}")
    if not res_dir.is_dir():
        raise ValueError(f"Submission path is not a directory: {res_dir}")

    preferred = res_dir / "submission.csv"
    if preferred.exists():
        return preferred

    csv_files = sorted(res_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in submission directory: {res_dir}")
    raise ValueError(
        "Multiple CSV files found in submission directory; expected exactly one or 'submission.csv': "
        f"{[p.name for p in csv_files]}"
    )


def resolve_truth_csv(ref_dir: Path) -> Path:
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference directory does not exist: {ref_dir}")
    if not ref_dir.is_dir():
        raise ValueError(f"Reference path is not a directory: {ref_dir}")

    preferred = ref_dir / "test_clean_with_labels.csv"
    if preferred.exists():
        return preferred

    csv_files = sorted(ref_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in reference directory: {ref_dir}")
    raise ValueError(
        "Multiple CSV files found in reference directory; expected exactly one or 'test_clean_with_labels.csv': "
        f"{[p.name for p in csv_files]}"
    )


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        rows = list(reader)
        return rows, reader.fieldnames


def validate_required_columns(columns: Iterable[str], required: set[str], name: str) -> None:
    missing = required.difference(set(columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def rows_to_label_map(rows: List[Dict[str, str]], dataset_name: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for idx, row in enumerate(rows, start=2):
        sample_id = str(row["id"]).strip()
        label = normalize_label(row["label"])
        if sample_id == "":
            raise ValueError(f"{dataset_name} has empty id at CSV row {idx}")
        if label == "":
            raise ValueError(f"{dataset_name} has empty label at CSV row {idx}")
        if sample_id in result:
            raise ValueError(f"{dataset_name} has duplicate id '{sample_id}' (row {idx})")
        result[sample_id] = label
    return result


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        raise ValueError("No rows available to score.")
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def compute_macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        raise ValueError("No labels available to score.")

    f1_values: List[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_values.append(f1)

    return sum(f1_values) / len(f1_values)


def write_scores(output_dir: Path, accuracy: float, macro_f1: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "score": macro_f1,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }
    (output_dir / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"score: {macro_f1:.6f}",
        f"accuracy: {accuracy:.6f}",
        f"macro_f1: {macro_f1:.6f}",
    ]
    (output_dir / "scores.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_error_outputs(output_dir: Path, error_message: str, traceback_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "score": 0.0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "error": error_message,
    }
    (output_dir / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "error: scorer failed",
        error_message,
        "",
        "traceback:",
        traceback_text,
    ]
    (output_dir / "scores.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codabench scorer for CSV submissions.")
    parser.add_argument("input_dir", help="Codabench input directory (contains ref/ and res/).")
    parser.add_argument("output_dir", help="Codabench output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    try:
        ref_dir = input_dir / "ref"
        res_dir = input_dir / "res"

        submission_csv = resolve_submission_csv(res_dir)
        truth_csv = resolve_truth_csv(ref_dir)

        submission_rows, submission_columns = read_csv_rows(submission_csv)
        truth_rows, truth_columns = read_csv_rows(truth_csv)

        validate_required_columns(submission_columns, SUBMISSION_REQUIRED_COLUMNS, "Submission CSV")
        validate_required_columns(truth_columns, TRUTH_REQUIRED_COLUMNS, "Ground-truth CSV")

        pred_map = rows_to_label_map(submission_rows, "Submission CSV")
        truth_map = rows_to_label_map(truth_rows, "Ground-truth CSV")

        if len(pred_map) != len(truth_map):
            raise ValueError(
                f"Row count mismatch: submission has {len(pred_map)} rows, ground truth has {len(truth_map)} rows."
            )

        pred_ids = set(pred_map.keys())
        truth_ids = set(truth_map.keys())
        if pred_ids != truth_ids:
            missing_ids = sorted(truth_ids - pred_ids)
            extra_ids = sorted(pred_ids - truth_ids)
            raise ValueError(
                "Submission IDs do not match ground truth. "
                f"Missing IDs count: {len(missing_ids)}, extra IDs count: {len(extra_ids)}."
            )

        ordered_ids = list(truth_map.keys())
        y_true = [truth_map[sample_id] for sample_id in ordered_ids]
        y_pred = [pred_map[sample_id] for sample_id in ordered_ids]

        accuracy = compute_accuracy(y_true, y_pred)
        macro_f1 = compute_macro_f1(y_true, y_pred)
        write_scores(output_dir, accuracy, macro_f1)
    except Exception as exc:
        write_error_outputs(output_dir, f"{type(exc).__name__}: {exc}", traceback.format_exc().rstrip())


if __name__ == "__main__":
    main()
