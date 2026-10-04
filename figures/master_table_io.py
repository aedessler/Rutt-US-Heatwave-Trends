"""What the plot-only figure scripts share: the master annual table and its command line.

Not a figure. build_master_table.py writes PAPER_FIGURES_FINAL/master_annual_table.csv (rows = years,
columns = <domain>_<quantity>__<dataset>); figure{1,2,4,5}_20cr_era20c.py read it through here and
take the columns of the datasets named on the command line (`--datasets`).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def default_csv():
    from common import FIGD
    return Path(FIGD) / "master_annual_table.csv"


def parse_args(description, all_datasets, default_datasets, default_out, extra=None):
    """The options every figure takes: --csv, --datasets, --out, --list (+ `extra(ap)` for its own)."""
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(default_csv()), help="master annual table (default: %(default)s)")
    ap.add_argument("--datasets", nargs="*", default=None, metavar="DS",
                    help=f"datasets to draw, from: {' '.join(all_datasets)}.  "
                         f"Default: {' '.join(default_datasets)}")
    ap.add_argument("--out", default=default_out, help="output PNG name inside PAPER_FIGURES_FINAL "
                    "(or a path); default %(default)s")
    ap.add_argument("--list", action="store_true", help="list this figure's columns in the table and exit")
    if extra:
        extra(ap)
    a = ap.parse_args()
    a.datasets = list(default_datasets if a.datasets is None else a.datasets)
    unknown = [d for d in a.datasets if d not in all_datasets]
    if unknown:
        ap.error(f"unknown dataset(s) {unknown}; choose from {all_datasets}")
    return a


def load_table(csv):
    csv = Path(csv)
    if not csv.exists():
        sys.exit(f"{csv} does not exist -- build it first:  python figures/build_master_table.py")
    df = pd.read_csv(csv, index_col=0)
    df.index = df.index.astype(int)
    return df


def column(df, name):
    """The column as a float Series, or None when the table lacks it."""
    return df[name].astype(float) if name in df.columns else None


def need(df, names, what):
    """Columns the figure draws that the table does not have: say how to get them and stop."""
    missing = [n for n in names if n not in df.columns]
    if missing:
        sys.exit(f"{what}: the table has no column(s) {missing}.\n"
                 f"  python figures/build_master_table.py --datasets <those datasets>")


def list_columns(df, prefixes):
    print("columns of the table this figure can use:")
    for c in df.columns:
        if any(c.startswith(p) for p in prefixes):
            v = df[c].dropna()
            print(f"  {c:<46} {int(v.index.min())}-{int(v.index.max())}" if len(v) else f"  {c:<46} (empty)")


def out_path(out, figd):
    p = Path(out)
    return p if p.is_absolute() or p.parent != Path(".") else Path(figd) / p
