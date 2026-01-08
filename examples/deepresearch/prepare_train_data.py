import pandas as pd

from rllm.data.dataset import DatasetRegistry

from custom_evaluate import task_specific_prompt, user_prompt_template, load_task_description


def build_rows(competition_ids: list[str]) -> list[dict]:
    """
    Build rows with question + placeholder ground_truth for the given competitions.
    """
    rows = []
    for comp_id in competition_ids:
        desc = load_task_description(comp_id)
        specific_prompt = task_specific_prompt.replace("{id}", comp_id).replace("{task_description}", desc)
        question = user_prompt_template.replace("{specific_task_description}", specific_prompt)
        rows.append({"prompt": [{"role": "user", "content": question}], "extra_info": {"competition_id": comp_id, 'question': question}})
    return rows


def prepare_mle_train_data():
    """
    Prepare a small mle_bench dataset (train/test) from the DeepResearch prompts.
    Train and test contain the same rows; ground_truth is a placeholder.
    """
    competition_ids = ["spaceship-titanic", "spooky-author-identification"]
    rows = build_rows(competition_ids)
    repeated_rows = rows * 4  # expand to 8 examples

    train_dataset = DatasetRegistry.register_dataset("mle_bench", repeated_rows, "train")
    test_dataset = DatasetRegistry.register_dataset("mle_bench", rows, "test")

    print(f"Train dataset size: {len(train_dataset)} at {train_dataset.get_data_path()}")
    print(f"Test dataset size: {len(test_dataset)} at {test_dataset.get_data_path()}")

    # Print a sample
    # print("\nSample train example:")
    # print(train_dataset[0])

    return train_dataset, test_dataset


if __name__ == "__main__":
    prepare_mle_train_data()
