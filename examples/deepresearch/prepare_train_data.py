import argparse
from pathlib import Path

from rllm.data.dataset import DatasetRegistry

from custom_evaluate import task_specific_prompt, user_prompt_template, load_task_description


def build_rows(competition_ids: list[str], data_root: Path) -> list[dict]:
    """
    Build rows with question + placeholder ground_truth for the given competitions.
    """
    rows = []
    for comp_id in competition_ids:
        desc = load_task_description(comp_id, data_root)
        specific_prompt = (
            task_specific_prompt.replace("{id}", comp_id)
            .replace("{task_description}", desc)
            .replace("{data_root}", str(data_root))
        )
        question = user_prompt_template.replace("{specific_task_description}", specific_prompt)
        rows.append({"prompt": [{"role": "user", "content": question}], "extra_info": {"competition_id": comp_id, "question": question}})
    return rows


def prepare_mle_train_data(competition_ids: list[str], data_root: Path, dataset_name: str):
    """
    Prepare a small mle_bench dataset (train/val) from the DeepResearch prompts.
    Validation takes 10% of rows and is removed from training data.
    """
    rows = build_rows(competition_ids, data_root)

    if not rows:
        raise ValueError("No rows generated; please check competition ids and data root.")

    val_size = int(len(rows) * 0.1)
    if val_size == 0 and len(rows) > 1:
        val_size = 1

    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    train_dataset = DatasetRegistry.register_dataset(dataset_name, train_rows, "train")
    test_dataset = DatasetRegistry.register_dataset(dataset_name, val_rows, "test")

    print(f"Train dataset size: {len(train_dataset)} at {train_dataset.get_data_path()}")
    print(f"Validation dataset size: {len(test_dataset)} at {test_dataset.get_data_path()}")

    # Print a sample
    # print("\nSample train example:")
    # print(train_dataset[0])

    return train_dataset, test_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare mle_bench datasets for DeepResearch training.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset root at /fsx/zyhang/mle-bench-syn.",
    )
    parser.add_argument(
        "--competition-ids",
        default="spaceship-titanic,spooky-author-identification",
        help="Comma-separated competition ids.",
    )
    return parser.parse_args()

def _list_synthetic_competitions(data_root: Path) -> list[str]:
    competition_ids = []
    if not data_root.exists():
        return competition_ids
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        desc_path = entry / "prepared" / "public" / "description.md"
        if desc_path.exists():
            competition_ids.append(entry.name)
    return competition_ids


if __name__ == "__main__":
    args = parse_args()
    data_root = Path("/fsx/zyhang/mle-bench-syn" if args.synthetic else "/fsx/zyhang/mle-bench-data")
    dataset_name = "mle_bench_syn" if args.synthetic else "mle_bench"
    if args.synthetic:
        competition_ids = _list_synthetic_competitions(data_root)
    else:
        competition_ids = [cid.strip() for cid in args.competition_ids.split(",") if cid.strip()]
    prepare_mle_train_data(competition_ids, data_root, dataset_name)
