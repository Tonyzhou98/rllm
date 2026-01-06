import os
import asyncio
from datetime import datetime
from pathlib import Path
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from deepresearch_workflow import DeepResearchWorkflow
from deepresearch_tools import PythonInterpreterTool, ScoreTool
from rllm.engine.rollout import OpenAIEngine

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
/fsx/zyhang/mle-bench-data/{id}/prepared/public/
"""

user_prompt_template = """
You are solving the task below. Follow the requirements precisely.

{specific_task_description}

Your code should adhere to the following requirements:
- Prefer and explicitly use GPU (CUDA) acceleration when available (one A100 GPU should be available): move models/tensors to GPU and handle CPU fallback if CUDA is not present.
- Each PythonInterpreter execution must finish within 1 hour (hard limit). 
- Overall runtime limits: the agent may take up to 100 turns, and the total program time budget (total tool calling + token generation) is 24 hours.
- Load train/test data from the provided dataset folder (## Dataset Folder).
- Match the exact columns/headers in sample_submission.csv (## Dataset Folder) and write submission.csv to the **current directory**.
- Use only common preinstalled libraries (no installs).
- Please restrict the use of external libraries to the common libraries.
- The task is an out-of-date competition, so please ignore the timeline in the task description.
"""


def load_task_description(competition_id: str) -> str:
    """Read description.md for a competition and return its content."""
    competition_path = Path(f"/fsx/zyhang/mle-bench-data/{competition_id}/prepared/public/")
    if not competition_path.exists():
        raise FileNotFoundError(f"Competition data path does not exist: {competition_path}")

    print(f"Using competition data path: {competition_path}")
    description_file = competition_path / "description.md"
    if not description_file.exists():
        raise FileNotFoundError(f"description.md not found for competition: {competition_id}")

    return description_file.read_text().strip()


# Setup rollout engine
engine = OpenAIEngine(
    model="anthropic/claude-sonnet-4.5",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Create workflow engine for parallel execution
workflow_engine = AgentWorkflowEngine(
    workflow_cls=DeepResearchWorkflow,
    workflow_args={
        "tools": {
            "PythonInterpreter": PythonInterpreterTool(),
            "Score": ScoreTool(),
        },
        "max_prompt_length": 4096,
        "max_response_length": 2048,
        "system_prompt": SYSTEM_PROMPT,
    },
    rollout_engine=engine,
    n_parallel_tasks=5  # Run 5 tasks in parallel
)

# Run evaluation on multiple tasks
competition_id_list = [
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
]  # Hardcoded competition ids to run

tasks = []
for competition_id in competition_id_list:
    specific_task_description = load_task_description(competition_id)
    specific_prompt = task_specific_prompt.replace("{id}", competition_id).replace("{task_description}", specific_task_description)
    tasks.append(
        {"question": user_prompt_template.replace("{specific_task_description}", specific_prompt), "answer": "submission"}
    )


def setup_output_directory() -> Path:
    """
    Create output/<timestamp> under this file's directory and switch cwd there so
    all generated files (e.g., submission.csv) land inside the run folder.
    """
    base_dir = Path(__file__).resolve().parent
    output_root = base_dir / "output"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    # Let other modules know where to write artifacts/logs
    os.environ["DEEPRESEARCH_OUTPUT_DIR"] = str(run_dir)
    print(f"Saving run artifacts to {run_dir}")
    return run_dir


async def main():
    setup_output_directory()
    episodes = await workflow_engine.execute_tasks(tasks)

    # Episodes contain full trajectories for training
    for episode in episodes:
        print(f"Task: {episode.task}")
        print(f"Prediction: {episode.metrics.get('prediction')}")
        print(f"Is correct: {episode.is_correct}")

if __name__ == "__main__":
    asyncio.run(main())
