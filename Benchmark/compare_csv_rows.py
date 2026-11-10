"""
Compares two CSV files by content: same column names (in any order) and the same rows (in any order).
Prints "same rows" when the content is equal; otherwise what differs.

Usage: python3 compare_csv_rows.py new.csv old.csv
"""
import sys

import pandas as pd


def main(new_path, old_path):
    a = pd.read_csv(new_path, dtype=str, keep_default_na=False, low_memory=False)
    b = pd.read_csv(old_path, dtype=str, keep_default_na=False, low_memory=False)
    ca, cb = set(a.columns), set(b.columns)
    if ca != cb:
        print(f"DIFFERENT columns: only in the first file {sorted(ca - cb)[:10]}, "
              f"only in the second file {sorted(cb - ca)[:10]}")
        return 1
    cols = sorted(ca)
    a = a[cols].sort_values(cols, kind='mergesort').reset_index(drop=True)
    b = b[cols].sort_values(cols, kind='mergesort').reset_index(drop=True)
    if len(a) != len(b):
        print(f"DIFFERENT number of rows: {len(a)} vs {len(b)}")
        return 1
    diff = (a != b).any(axis=1)
    if diff.any():
        bad = (a != b).any(axis=0)
        print(f"DIFFERENT: {int(diff.sum())} rows; columns with differences: {list(bad[bad].index)[:10]}")
        return 1
    order = "same column order" if list(pd.read_csv(new_path, nrows=0).columns) == \
        list(pd.read_csv(old_path, nrows=0).columns) else "column order differs"
    print(f"same rows ({len(a)} rows, {len(cols)} columns, {order})")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
