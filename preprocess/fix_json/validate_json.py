#!/usr/bin/env python3
import sys, io, argparse, gzip
import json
import zstandard as zstd

def iter_lines(path):
    """Yield text lines from .jsonl / .jsonl.gz / .jsonl.zstd using streaming IO."""
    if path.endswith(".zstd"):
        with open(path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                for line in text:
                    yield line
    elif path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line
import tqdm
def validate_fused_jsonl(path, sample_bad=5):
    right = 0
    decoder = json.JSONDecoder()
    bad_lines = 0
    multi_object_lines = 0
    total_lines = 0
    examples = []

    for line in tqdm.tqdm(iter_lines(path)):
        total_lines += 1
        stripped = line.strip()
        if not stripped:
            continue
        try:
            _, end = decoder.raw_decode(stripped)
            if end != len(stripped):
                multi_object_lines += 1
                if len(examples) < sample_bad:
                    # show a small window around the split point
                    s = stripped[max(0, end-30):min(len(stripped), end+30)]
                    examples.append(("multi", total_lines, s))
        except json.JSONDecodeError as e:
            bad_lines += 1
            if len(examples) < sample_bad:
                start = max(0, e.pos - 30)
                endp  = min(len(stripped), e.pos + 30)
                examples.append(("invalid", total_lines, stripped[start:endp]))

    print(f"Checked: {path}")
    print(f"Total lines: {total_lines}")
    print(f"Lines with >1 JSON object: {multi_object_lines}")
    print(f"Invalid JSON lines: {bad_lines}")
    if multi_object_lines or bad_lines:
        print("❌ Problems detected — likely missing newline(s) between files or other corruption.")
        right = 0
    else:
        print("✅ No glued or invalid lines detected.")
        right = 1

    if examples:
        print("\nExamples (type, line_no, snippet around issue):")
        for kind, lineno, snippet in examples:
            print(f"  {kind:7s}  line {lineno}: {snippet}")
    return right

def main():
    ap = argparse.ArgumentParser(description="Validate fused JSONL(.gz/.zstd) files for glued records and malformed lines.")
    ap.add_argument("paths", nargs="+", help="File(s) to check (.jsonl, .jsonl.gz, .jsonl.zstd)")
    ap.add_argument("--sample-bad", type=int, default=5, help="Show up to N example bad lines (default: 5)")
    ap.add_argument("--output", type=str, default='fault.txt', help='filename that contains the list of the json files that is faulty')
    args = ap.parse_args()
    fout = open(args.output, 'w')

    for p in args.paths:
        right = validate_fused_jsonl(p, sample_bad=args.sample_bad)
        print("-" * 60)
        if right == 0:
            fout.write(p + "\n")
    fout.close()

if __name__ == "__main__":
    main()