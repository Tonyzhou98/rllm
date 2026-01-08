"""
DeepResearch Workflow for rLLM

This workflow integrates the DeepResearch agent with rLLM's AgentWorkflowEngine,
enabling parallel execution and trajectory tracking while maintaining DeepResearch's
core reasoning capabilities.
"""

import json

try:
    # When imported as a package (python -m examples.deepresearch.custom_train)
    from .deepresearch_agent import MultiTurnReactAgent
except ImportError:
    # Fallback for direct script execution
    from deepresearch_agent import MultiTurnReactAgent

from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine.rollout import RolloutEngine
from rllm.workflows.workflow import TerminationReason, Workflow


class DeepResearchWorkflow(Workflow):
    """
    Workflow that wraps the DeepResearch MultiTurnReactAgent for use with AgentWorkflowEngine.

    This workflow:
    1. Creates a DeepResearch agent instance
    2. Executes the research task using the agent's ReAct loop
    3. Converts the results to rLLM Episode format for trajectory tracking
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor,
        tools: dict = None,
        system_prompt: str = None,
        **kwargs,
    ):
        """
        Initialize the DeepResearch workflow.

        Args:
            rollout_engine: rLLM rollout engine for model inference
            executor: Thread pool executor for async operations
            tools: Dictionary of available tools for research tasks
            system_prompt: Custom system prompt (optional, uses default if None)
            **kwargs: Additional arguments passed to parent Workflow
        """
        super().__init__(rollout_engine, executor, **kwargs)

        self.tools = tools or {}
        self.system_prompt = system_prompt

        # Auto-detect if we should use native function calling
        # O3 models require native function calling, other models use XML format
        # model_name = rollout_engine.model.lower()
        # use_native_fc = "o3" in model_name or "o1" in model_name

        # Create the DeepResearch agent
        self.agent = MultiTurnReactAgent(
            rollout_engine=rollout_engine,
            tools=self.tools,
            system_prompt=self.system_prompt,
            use_native_function_calling=False,
        )

        # Note: We don't register the agent since DeepResearch handles its own trajectory

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Execute the DeepResearch workflow on a single task.

        Args:
            task: Task dictionary containing:
                - question: The research question to answer
                - answer: Ground truth answer (optional, for evaluation)
                - Any other task metadata
            uid: Unique identifier for this episode

        Returns:
            Episode object with trajectory and results
        """
        # Reset workflow state for this task
        self.reset(task=task, uid=uid)

        # Extract question and answer from task
        question = task.get("question", task.get("query", "No question provided"))
        answer = task.get("answer", "")

        print(f"🚀 Starting DeepResearch workflow for task {uid}")
        print(f"   Question: {question}")

        try:
            # Run the DeepResearch agent
            result = await self.agent.run(question=question, answer=answer, **kwargs)

            # Convert the result to rLLM Episode format
            episode = self._convert_to_episode(result, task, uid)

            print(f"✅ DeepResearch workflow completed for task {uid}")
            print(f"   Prediction: {result.get('prediction', 'No prediction')}")

            return episode

        except Exception as e:
            print(f"❌ DeepResearch workflow failed for task {uid}: {e}")

            # Create a failed episode
            episode = Episode()
            episode.id = uid
            episode.task = task
            episode.termination_reason = TerminationReason.UNKNOWN
            episode.is_correct = False
            episode.trajectories = []
            episode.metrics = {"error": str(e)}
            return episode

    def _extract_score_metrics(self, messages: list[dict]) -> dict:
        """
        Find the latest ScoreTool output (handles both ReAct <tool_response> and native tool roles).
        """
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")

            # Native tool_call path: role == "tool" with raw JSON string
            if msg.get("role") == "tool" and isinstance(content, str):
                try:
                    content = content.split("Details: ", 1)[1]
                    parsed = json.loads(content.strip())
                    if isinstance(parsed, dict) and "score_primary (main competition metric for current code)" in parsed:
                        return parsed
                except Exception:
                    pass

            # ReAct path: user message with <tool_response> wrapper
            if isinstance(content, str) and "<tool_response>" in content:
                try:
                    payload = content.split("<tool_response>", 1)[1].split("</tool_response>", 1)[0]
                    payload = payload.split("Details: ", 1)[1]
                    parsed = json.loads(payload.strip())
                    if isinstance(parsed, dict) and "score_primary (main competition metric for current code)" in parsed:
                        return parsed
                except Exception:
                    continue

        return {}

    def _count_errors(self, trajectory: Trajectory) -> tuple[int, float]:
        """Count observations that look like errors/timeouts and return (count, rate)."""
        err_markers = ("[error", "error:", "timeout", "[timeout", "failed", "exception")
        count = 0
        for step in trajectory.steps:
            obs = step.observation or ""
            if not isinstance(obs, str):
                continue
            low = obs.lower()
            if any(marker in low for marker in err_markers):
                count += 1
        total_steps = max(1, len(trajectory.steps))
        return count, count / total_steps

    def _score_to_reward(self, metrics: dict, error_count: int, error_rate: float, message_count: int) -> tuple[float, dict]:
        """
        Convert ScoreTool metrics into a scalar reward.
        - Uses score_primary and metric_lower_is_better.
        - Applies a small penalty per observed error and conversation length to encourage clean, short runs.
        Returns (reward, extra_metrics).
        """
        reward = 0.0
        score = metrics.get("score_primary (main competition metric for current code)")
        lower_is_better = metrics.get("metric_lower_is_better") or metrics.get("metric_lower_is_better (true means lower score is better)")

        if score is not None:
            try:
                score_val = float(score)
                reward = -score_val if lower_is_better else score_val
            except Exception:
                reward = 0.0

        # Light penalties for noisy/long runs
        reward -= 0.02 * error_rate  # scaled by rate
        reward -= 0.005 * message_count  # discourage excessive turns

        # Clamp to keep extremes bounded
        reward = max(min(reward, 1e3), -1e3)

        extra = {
            "reward_raw": reward,
            "score_primary": score,
            "metric_lower_is_better": lower_is_better,
            "metric_lower_is_better (true means lower score is better)": lower_is_better,
            "error_count": error_count,
            "error_rate": error_rate,
            "message_count": message_count,
        }
        return reward, extra

    def _convert_to_episode(self, result: dict, task: dict, uid: str) -> Episode:
        """
        Convert DeepResearch result to rLLM Episode format.

        Args:
            result: Result dictionary from DeepResearch agent
            task: Original task dictionary
            uid: Episode unique identifier

        Returns:
            Episode object with trajectory
        """
        # Create trajectory from the conversation messages
        trajectory = Trajectory(task=task.get("question", ""))

        # Convert conversation to steps
        messages = result.get("messages", [])

        i = 0
        while i < len(messages):
            # Look for assistant messages (model responses)
            if messages[i]["role"] == "assistant":
                # Build chat completion context up to this point
                current_context = messages[: i + 1]

                # Create step
                step = Step(
                    chat_completions=current_context.copy(),
                    model_response=messages[i]["content"],
                    action=self._extract_action_from_response(messages[i]["content"]),
                    observation=self._get_next_observation(messages, i),
                    reward=0.0,  # Will be computed later if needed
                )

                trajectory.steps.append(step)

            i += 1

        # Determine if the answer is correct (if ground truth available)
        prediction = result.get("prediction", "")
        score_metrics = self._extract_score_metrics(messages)

        error_count, error_rate = self._count_errors(trajectory)
        message_count = len(messages)
        reward_value, reward_details = self._score_to_reward(score_metrics, error_count, error_rate, message_count)

        # For our Kaggle-style tasks, correctness is tied to reward from ScoreTool
        is_correct = reward_value > 0

        # Map termination reason
        termination_reason = self._map_termination_reason(result.get("termination", "unknown"))

        # Create episode
        episode = Episode()
        episode.id = uid
        episode.task = task
        episode.termination_reason = termination_reason
        episode.is_correct = is_correct
        trajectory.name = "deepresearch_agent"
        trajectory.reward = reward_value
        episode.trajectories = [trajectory]
        episode.metrics = {
            "rounds": result.get("rounds", 0),
            "time_taken": result.get("time_taken", 0),
            # "prediction": prediction,
            # "score_metrics": score_metrics,
            "reward": reward_value,
            "error_count": error_count,
            # "reward_details": reward_details,
        }

        return episode

    def _extract_action_from_response(self, response: str) -> Action:
        """
        Extract action information from model response.

        Args:
            response: Model response text

        Returns:
            Action object
        """
        # Check for tool calls
        if "<tool_call>" in response and "</tool_call>" in response:
            tool_call_text = response.split("<tool_call>")[1].split("</tool_call>")[0]
            return Action(action={"type": "tool_call", "tool_call": tool_call_text.strip()})
        # Check for final answer
        elif "<answer>" in response and "</answer>" in response:
            answer = response.split("<answer>")[1].split("</answer>")[0].strip()
            return Action(action={"type": "final_answer", "answer": answer})
        else:
            # Just thinking/reasoning
            return Action(action={"type": "reasoning", "content": response})

    def _get_next_observation(self, messages: list, current_index: int) -> str:
        """
        Get the observation that follows the current assistant message.

        Args:
            messages: List of all messages
            current_index: Index of current assistant message

        Returns:
            Next observation string (tool response or empty)
        """
        if current_index + 1 < len(messages):
            next_msg = messages[current_index + 1]
            if next_msg["role"] == "user" and "<tool_response>" in next_msg["content"]:
                return next_msg["content"]

        return ""

    def _evaluate_answer(self, prediction: str, ground_truth: str) -> bool:
        """
        Simple answer evaluation (can be enhanced with specific metrics).

        Args:
            prediction: Model's predicted answer
            ground_truth: Correct answer

        Returns:
            True if correct, False otherwise
        """
        if not prediction or not ground_truth:
            return False

        # Simple string matching (can be enhanced with fuzzy matching, etc.)
        return prediction.strip().lower() == ground_truth.strip().lower()

    def _map_termination_reason(self, termination: str) -> TerminationReason:
        """
        Map DeepResearch termination reasons to rLLM TerminationReason enum.

        Args:
            termination: DeepResearch termination string

        Returns:
            Mapped TerminationReason
        """
        mapping = {
            "answer": TerminationReason.ENV_DONE,
            "timeout": TerminationReason.TIMEOUT,
            "max_rounds_reached": TerminationReason.MAX_TURNS_EXCEEDED,
            "token_limit_no_answer": TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED,
            "answer_token_limit": TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED,
        }

        return mapping.get(termination, TerminationReason.UNKNOWN)

    def reset(self, task: dict = None, uid: str = None):
        """
        Reset the workflow for a new task.

        Args:
            task: New task dictionary
            uid: New unique identifier
        """
        # Keep base class bookkeeping so uid/task exist for error handling
        try:
            super().reset(task=task, uid=uid)
        except Exception:
            # Fallback assignment if parent reset changes in future
            self.task = task
            self.uid = uid

    def is_multithread_safe(self) -> bool:
        """
        Indicate whether this workflow is safe for multithreaded execution.

        Returns:
            True, as each workflow instance manages its own state
        """
        return True
