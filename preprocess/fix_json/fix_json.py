#!/usr/bin/env python3
"""
Repair fused JSONL streams where records may be glued together (e.g., `}{`)
without newlines, especially after concatenating .jsonl(.zstd/.gz) shards.

Strategy: parse the *entire decompressed stream* as a sequence of JSON values
(using incremental `raw_decode`) and re‑emit exactly one JSON object per line.
Works even if multiple objects are jammed on one physical line or split across
chunk boundaries.

Usage
-----
# Single file, write alongside with .fixed.jsonl.zstd
python3 fix_json.py /path/to/fused_000.jsonl.zstd

# Many files at once
python3 fix_json.py fused_*.jsonl.zstd

# Custom output directory and format
python3 fix_json.py fused_000.jsonl.zstd \
  --out-dir /tmp/fixed \
  --suffix .fixed \
  --out-format zstd --zstd-level 19 --zstd-threads 0

# Write plain .jsonl next to inputs
python3 fix_json.py fused_000.jsonl.zstd --out-format plain
"""

import argparse
import io
import json
import os
import sys
import gzip
from typing import Iterator, Tuple

try:
    import zstandard as zstd
except Exception:
    zstd = None  # Only required for .zstd IO

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


# ----------------------------- IO helpers -----------------------------
def _open_text_reader(path: str) -> io.TextIOBase:
    """Open text reader for .zstd/.gz/plain using streaming decompression.
    Returns an object with .read(size) -> str.
    """
    if path.endswith('.zstd'):
        if zstd is None:
            raise RuntimeError("zstandard module is required for .zstd files. pip install zstandard")
        fh = open(path, 'rb')
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(fh)
        # TextIOWrapper will own 'reader' and close it (and fh) when closed
        return io.TextIOWrapper(reader, encoding='utf-8', errors='replace', newline='')
    elif path.endswith('.gz'):
        # gzip supports text mode directly
        return gzip.open(path, 'rt', encoding='utf-8', errors='replace', newline='')
    else:
        return open(path, 'r', encoding='utf-8', errors='replace', newline='')


def _open_text_writer(path: str, out_format: str, zstd_level: int, zstd_threads: int) -> io.TextIOBase:
    """Open a text writer in the requested format: 'zstd', 'gz', or 'plain'."""
    if out_format == 'zstd':
        if zstd is None:
            raise RuntimeError("zstandard module is required for zstd output. pip install zstandard")
        fh = open(path, 'wb')
        cctx = zstd.ZstdCompressor(level=zstd_level, threads=zstd_threads)
        writer = cctx.stream_writer(fh)
        return io.TextIOWrapper(writer, encoding='utf-8')
    elif out_format == 'gz':
        return gzip.open(path, 'wt', encoding='utf-8')
    else:
        return open(path, 'w', encoding='utf-8', newline='')


# ----------------------------- Core logic -----------------------------
def _stream_json_values(text_reader: io.TextIOBase) -> Iterator[str]:
    """Incrementally parse a stream of UTF-8 text into consecutive JSON values.

    Yields each JSON *value* as a compact JSON string (one per emission).
    This does not assume any line boundaries in the input.
    """
    decoder = json.JSONDecoder()
    buf = ''
    i = 0  # current parse index into buf

    def _need_more() -> bool:
        nonlocal buf, i
        chunk = text_reader.read(CHUNK_SIZE)
        if not chunk:
            return False
        # Append and keep index where it is
        buf += chunk
        return True

    while True:
        # Ensure we have data to process
        if i >= len(buf):
            if not _need_more():
                # End of input: emit nothing further (ignore trailing whitespace)
                break

        # Skip any whitespace between JSON values
        n = len(buf)
        while i < n and buf[i].isspace():
            i += 1
        if i >= len(buf):
            # need more bytes to continue
            if not _need_more():
                break
            continue
        # Try to decode a JSON value starting at position i
        try:
            obj, j = decoder.raw_decode(buf, i)
        except json.JSONDecodeError:
            # Need more input to complete a JSON value; fetch more and retry
            if not _need_more():
                # We are at EOF and cannot decode a final value -> malformed tail
                snippet = buf[max(0, i-40):i+40]
                raise ValueError(f"Malformed JSON at end of stream near: {snippet!r}")
            continue
        # Successfully decoded a value; emit it
        yield json.dumps(obj, ensure_ascii=False)
        i = j
        # Periodically compact the buffer to avoid unbounded growth
        if i > 1_000_000:
            buf = buf[i:]
            i = 0


def fix_file(in_path: str, out_path: str, out_format: str, zstd_level: int, zstd_threads: int) -> Tuple[int, int]:
    """Fix one input file and write to out_path.

    Returns: (objects_written, bytes_written_out)
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    total_objs = 0
    bytes_out = 0

    with _open_text_reader(in_path) as reader, _open_text_writer(out_path, out_format, zstd_level, zstd_threads) as writer:
        for json_str in _stream_json_values(reader):
            writer.write(json_str)
            writer.write('\n')
            total_objs += 1
            bytes_out += len(json_str) + 1
        writer.flush()

    return total_objs, bytes_out


# ----------------------------- CLI -----------------------------
def guess_output_path(in_path: str, out_dir: str, suffix: str, out_format: str) -> str:
    base = os.path.basename(in_path)
    # Strip known input compressions for clarity in output basename
    if base.endswith('.jsonl.zstd'):
        stem = base[:-len('.jsonl.zstd')]
    elif base.endswith('.jsonl.gz'):
        stem = base[:-len('.jsonl.gz')]
    elif base.endswith('.jsonl'):
        stem = base[:-len('.jsonl')]
    else:
        # unknown suffix; keep whole name
        stem, _ = os.path.splitext(base)

    stem = f"{stem}{suffix}.jsonl"
    ext = { 'zstd': '.zstd', 'gz': '.gz', 'plain': '' }[out_format]
    out_base = stem + ext
    return os.path.join(out_dir or os.path.dirname(in_path) or '.', out_base)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Repair fused JSONL streams where newline boundaries may be missing. "
            "Parses a stream of consecutive JSON values and writes one object per line."
        )
    )
    p.add_argument('inputs', nargs='+', help='Input files (.jsonl, .jsonl.gz, .jsonl.zstd)')
    p.add_argument('--out-dir', default='', help='Directory for outputs (default: alongside inputs)')
    p.add_argument('--suffix', default='.fixed', help='Suffix to append before .jsonl (default: .fixed)')
    p.add_argument('--out-format', choices=['zstd', 'gz', 'plain'], default='zstd', help='Output format (default: zstd)')
    p.add_argument('--zstd-level', type=int, default=19, help='zstd compression level (default: 19)')
    p.add_argument('--zstd-threads', type=int, default=0, help='zstd worker threads (0 = auto)')
    p.add_argument('--stats', action='store_true', help='Print per-file statistics after writing')

    args = p.parse_args(argv)

    any_errors = False

    for in_path in args.inputs:
        try:
            out_path = guess_output_path(in_path, args.out_dir, args.suffix, args.out_format)
            objs, bytes_out = fix_file(in_path, out_path, args.out_format, args.zstd_level, args.zstd_threads)
            if args.stats:
                size_mb = bytes_out / (1024*1024)
                print(f"✔ {in_path} -> {out_path} | {objs} objects | ~{size_mb:.2f} MiB written")
            else:
                print(f"✔ Wrote: {out_path}")
        except Exception as e:
            any_errors = True
            print(f"✖ ERROR processing {in_path}: {e}", file=sys.stderr)

    sys.exit(1 if any_errors else 0)


if __name__ == '__main__':
    main()