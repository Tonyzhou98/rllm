import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout import OpenAIEngine

from deepresearch_tools import PythonInterpreterTool, ScoreTool, SynScoreTool
from deepresearch_workflow import DeepResearchWorkflow

SYSTEM_PROMPT = """You are an expert Kaggle competitor. Produce one Python script that trains a model and writes `submission.csv` for the dataset in the user prompt.

Rules:
- Use only already-installed common libraries (no installs).
- Use the PythonInterpreter tool to iteratively write/run/update your script.
- After producing a submission, use the Score tool to grade it; if the score is unsatisfying, keep refining the code and re-grading until you are satisfied.
- Be concise and task-focused.

Loop:
1) You are a multi-turn generation agent: in each turn, propose/refine the script or reasoning, then wait for environment/tool feedback.
2) Execute via the tool until it runs cleanly and produces the file. STRICT: each response may contain exactly ONE <tool_call> block—do not emit multiple tool calls.
3) After generating the code, the Python environment will provide feedback. You must observe at least one tool feedback (execution result wrapped in <tool_response></tool_response> tags) before deciding to end. Only when feedback looks good do you reply with <answer>submission</answer>; otherwise continue iterating (do not output <answer> tags).
4) Use PythonInterpreter to run updated code; use Score tool to grade `submission.csv`. Repeat this refine-grade loop until the submission is acceptable, then end with <answer>submission</answer>.

Tool usage:
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
- Wrap executable code exactly like this:
<tool_call>
python
<code>
# Your Python code here
print("Hello World")
</code>
</tool_call>
Code inside those tags runs in Python; keep the tool name `python` and include <code></code>.

- To grade the submission.csv file, you need to use Score tool and output json object like this:
<tool_call>
{"name": "Score", "arguments": {"competition_id": "competition id here, such as aerial-cactus-identification, which can be found in the task description"}}
</tool_call>

Current date: """

task_specific_prompt = """## Description

## Competition ID: {id}

{task_description}

## Dataset Folder:
{data_root}/{id}/prepared/public/
"""

user_prompt_template = """
You are solving the task below. Follow the requirements precisely.

{specific_task_description}

Your code should adhere to the following requirements:
- Prefer and explicitly use GPU (CUDA) acceleration when available (one A100 GPU should be available): move models/tensors to GPU and handle CPU fallback if CUDA is not present.
- Each Python interpreter execution must finish within 5 mins (hard limit), so please do not use cross-validation or long training loops.
- Overall runtime limits: the agent may take up to 30 turns, and the total program time budget (total tool calling + token generation) is 24 hours.
- Load train/test data from the provided dataset folder (## Dataset Folder). Please first check the data files and their formats (file types, column names, row counts, etc.).
- Match the exact columns/headers in sample_submission.csv (## Dataset Folder) and write submission.csv to the **current directory**.
- Use only common preinstalled libraries (no installs).
- DO NOT display progress bars due to the context window limit. If you have to use function integrated with progress bars, disable progress bars or use the appropriate parameter to silence them.
- Please restrict the use of external libraries to the common libraries.
- The task is an out-of-date competition, so please ignore the timeline in the task description.
"""

# competition_id_list = [
#     "aerial-cactus-identification",
#     "aptos2019-blindness-detection",
#     "denoising-dirty-documents",
#     "detecting-insults-in-social-commentary",
#     "dog-breed-identification",
#     "dogs-vs-cats-redux-kernels-edition",
#     "histopathologic-cancer-detection",
#     "jigsaw-toxic-comment-classification-challenge",
#     "leaf-classification",
#     "mlsp-2013-birds",
#     "new-york-city-taxi-fare-prediction",
#     "nomad2018-predict-transparent-conductors",
#     "plant-pathology-2020-fgvc7",
#     "random-acts-of-pizza",
#     "ranzcr-clip-catheter-line-classification",
#     "siim-isic-melanoma-classification",
#     "spooky-author-identification",
#     "tabular-playground-series-dec-2021",
#     "tabular-playground-series-may-2022",
#     "text-normalization-challenge-english-language",
#     "text-normalization-challenge-russian-language",
#     "the-icml-2013-whale-challenge-right-whale-redux",
# ]  # Hardcoded competition ids to run


# competition_id_list = ["spaceship-titanic"]
# competition_id_list = ["spaceship-titanic", "spooky-author-identification"]

competition_id_list = ["multi-organ-thoracic-cancer-detection-from-low-dose-chest-ct-volumes"]


def load_task_description(competition_id: str, data_root: Path) -> str:
    """Read description.md for a competition and return its content."""
    competition_path = data_root / competition_id / "prepared" / "public"
    if not competition_path.exists():
        raise FileNotFoundError(f"Competition data path does not exist: {competition_path}")

    print(f"Using competition data path: {competition_path}")
    description_file = competition_path / "description.md"
    if not description_file.exists():
        raise FileNotFoundError(f"description.md not found for competition: {competition_id}")

    return description_file.read_text().strip()


def build_tasks(competition_ids: list[str], data_root: Path) -> list[dict]:
    """Build user prompts/tasks for the provided competition ids."""
    tasks = []
    for competition_id in competition_ids:
        specific_task_description = load_task_description(competition_id, data_root)
        specific_prompt = (
            task_specific_prompt.replace("{id}", competition_id)
            .replace("{task_description}", specific_task_description)
            .replace("{data_root}", str(data_root))
        )
        tasks.append(
            {"question": user_prompt_template.replace("{specific_task_description}", specific_prompt)}
        )
    return tasks


def _slugify(text: str) -> str:
    """Make a filesystem-friendly slug from model names like provider/model."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in text)


def setup_output_directory(model_name: str) -> Path:
    """
    Create output/<model>-<timestamp> under this file's directory and switch cwd there so
    all generated files (e.g., submission.csv) land inside the run folder.
    """
    base_dir = Path(__file__).resolve().parent
    output_root = base_dir / "output"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"{_slugify(model_name)}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    # Let other modules know where to write artifacts/logs
    os.environ["DEEPRESEARCH_OUTPUT_DIR"] = str(run_dir)
    print(f"Saving run artifacts to {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model selection and runtime configuration."""
    parser = argparse.ArgumentParser(description="Run DeepResearch custom evaluation over Kaggle tasks.")
    parser.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4.5",
        help="Base model identifier for OpenRouter (e.g., 'anthropic/claude-sonnet-4.5' or 'qwen3-8b' placeholder).",
    )
    parser.add_argument(
        "--parallel-tasks",
        type=int,
        default=5,
        help="Number of parallel tasks to run.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Use synthetic dataset from /fsx/zyhang/mle-bench-syn and SynScoreTool.",
    )
    return parser.parse_args()


def create_rollout_engine(model: str) -> OpenAIEngine:
    """Helper to create the rollout engine with the desired base model."""
    resolved_api_key = os.environ.get("OPENROUTER_API_KEY")

    if "claude" in model:
        return OpenAIEngine(model=model, api_key=resolved_api_key, base_url="https://openrouter.ai/api/v1")
    elif "qwen" in model:
        return OpenAIEngine(model="qwen3_8b_serve", api_key="None", base_url="http://h200-060-102:8001/v1")
    else:
        raise ValueError(f"Unsupported model specified: {model}")

async def main():
    args = parse_args()
    setup_output_directory(args.model)
    data_root = Path("/fsx/zyhang/mle-bench-syn" if args.synthetic else "/fsx/zyhang/mle-bench-data")
    print("Data root:", data_root)

    engine = create_rollout_engine(model=args.model)
    score_tool = SynScoreTool() if args.synthetic else ScoreTool()
    workflow_engine = AgentWorkflowEngine(
        workflow_cls=DeepResearchWorkflow,
        workflow_args={
            "tools": {
                "PythonInterpreter": PythonInterpreterTool(),
                "Score": score_tool,
            },
            "max_prompt_length": 4096,
            "max_response_length": 2048,
            "system_prompt": SYSTEM_PROMPT,
        },
        rollout_engine=engine,
        n_parallel_tasks=args.parallel_tasks,
    )

    tasks = build_tasks(competition_id_list, data_root)
    episodes = await workflow_engine.execute_tasks(tasks)

    # Episodes contain full trajectories for training
    for episode in episodes:
        # print(f"Task: {episode.task}")
        # print(f"Prediction: {episode.metrics.get('prediction')}")
        print(f"Is correct: {episode.is_correct}")
        print(f"Reward: {episode.metrics.get('reward')}")
        print(f"Reward details: {episode.metrics.get('reward_details')}")

if __name__ == "__main__":
    asyncio.run(main())
