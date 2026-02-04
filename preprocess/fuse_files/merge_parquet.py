#!/usr/bin/env python3
import argparse
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def merge_parquet(filelist_path: str, output_file: str) -> None:
    # Read file list
    with open(filelist_path, "r") as f:
        filelist = [line.strip() for line in f if line.strip()]

    if not filelist:
        raise ValueError("No input files provided.")

    # Merge using Arrow dataset
    dataset = ds.dataset(filelist, format="parquet")
    pq.write_table(dataset.to_table(), output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple Parquet files into a single Parquet file."
    )
    parser.add_argument(
        "filelist",
        type=str,
        help="Path to a text file containing a list of Parquet file paths (one per line).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output Parquet file path.",
    )

    args = parser.parse_args()

    merge_parquet(args.filelist, args.output)


if __name__ == "__main__":
    main()
