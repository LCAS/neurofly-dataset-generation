"""Shared filesystem and CSV helpers."""

import csv
from pathlib import Path

import numpy as np


def default_output_path(input_path):
    path = Path(input_path)
    return path.with_name(f"{path.stem}_noisy{path.suffix}")


def read_csv(input_path):
    with Path(input_path).open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{input_path} is empty.")

    return rows[0], rows[1:]


def write_csv(output_path, header, rows):
    with Path(output_path).open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def rows_to_numeric_matrix(rows):
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.asarray([[float(value) for value in row] for row in rows], dtype=float)


def require_columns(header_index, columns):
    missing = [column for column in columns if column not in header_index]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def select_columns(matrix, header_index, columns):
    return np.column_stack([matrix[:, header_index[column]] for column in columns])
