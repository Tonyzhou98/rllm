import argparse
import json
from pathlib import Path

import pandas as pd


def load_messages(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    trajectories = data.get("trajectories", [])

    if not trajectories:
        return None
    last_step = trajectories[-1]['steps']

    if not last_step:
        return None
    last = last_step[-1]
    messages = last.get("chat_completions")
    if messages is None:
        return None
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data from custom_evaluate outputs.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder containing episode JSON files.")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {input_dir}")

    output_dir = Path("/fsx/zyhang/rllm/data/datasets/mle_bench_sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(p for p in input_dir.glob("*.json") if "episode" not in p.name)

    print(f"Found {len(json_files)} JSON files in {input_dir}")

    rows = []
    for path in json_files:
        row = load_messages(path)
        if row is not None:
            rows.append(row)

    test_size = min(8, len(rows))
    test_rows = rows[:test_size]
    train_rows = rows[test_size:]
    train_rows = train_rows * 8

    pd.DataFrame(train_rows).to_parquet(output_dir / "train.parquet", index=False)
    pd.DataFrame(test_rows).to_parquet(output_dir / "test.parquet", index=False)

    print(f"Loaded {len(rows)} episodes from {input_dir}")
    print(f"Saved {len(train_rows)} train rows and {len(test_rows)} test rows to {output_dir}")


if __name__ == "__main__":
    main()
