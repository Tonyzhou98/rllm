import os
from pathlib import Path

import hydra
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.data.dataset import DatasetRegistry
from .deepresearch_tools import PythonInterpreterTool, ScoreTool, SynScoreTool
from .deepresearch_workflow import DeepResearchWorkflow

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

SYNTHETIC_DATA = True
MAX_LLM_CALLS = int(os.environ.get("DEEPRESEARCH_MAX_LLM_CALLS", "15"))
MAX_TIME_S = int(os.environ.get("DEEPRESEARCH_MAX_TIME_S", str(20 * 60)))
CALL_SERVER_TIMEOUT_S = int(os.environ.get("DEEPRESEARCH_CALL_SERVER_TIMEOUT_S", "1800"))
PYTHON_TIMEOUT_S = int(os.environ.get("DEEPRESEARCH_PYTHON_TIMEOUT_S", "90"))
API_JOB_NAME = os.environ.get("DEEPRESEARCH_API_JOB_NAME", "deepresearch_api_job")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    dataset_name = "mle_bench_syn" if SYNTHETIC_DATA else "mle_bench"
    score_tool = SynScoreTool() if SYNTHETIC_DATA else ScoreTool()
    # Create trainer with your workflow
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")
    trainer = AgentTrainer(
        workflow_class=DeepResearchWorkflow,
        workflow_args={
            "tools": {
                "PythonInterpreter": PythonInterpreterTool(timeout=PYTHON_TIMEOUT_S, job_name=API_JOB_NAME),
                "Score": score_tool,
            },
            "system_prompt": SYSTEM_PROMPT,
            "max_llm_calls": MAX_LLM_CALLS,
            "max_time_s": MAX_TIME_S,
            "call_server_timeout_s": CALL_SERVER_TIMEOUT_S,
            "python_timeout_s": PYTHON_TIMEOUT_S,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    
    # Train!
    trainer.train()

if __name__ == "__main__":
    main()
