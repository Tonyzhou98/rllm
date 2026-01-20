"""
DeepResearch Agent - Adapted from Tongyi DeepResearch for rLLM

This is the core ReAct agent that implements DeepResearch's reasoning and tool-calling logic,
adapted to work with rLLM's OpenAI engine instead of the original server-based approach.

Original: https://github.com/Alibaba-NLP/DeepResearch/blob/main/inference/react_agent.py
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
import os
import sys

# rLLM imports
from rllm.engine.rollout import RolloutEngine

# Constants from original DeepResearch
OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"
MAX_LLM_CALL_PER_RUN = 15

# System prompt adapted from DeepResearch
# DEEPRESEARCH_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You MUST use the provided tools to research and verify information before answering. Do NOT answer directly from memory - always use tools to gather current, accurate information.

# IMPORTANT: You are REQUIRED to use at least one tool before providing any answer. Even if you think you know the answer, you must verify it using the appropriate tools. Direct answers without tool use are not acceptable.

# When you have gathered sufficient information through tool use and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# # Tools

# You MUST use one or more of the following tools to research the query:

# You are provided with the following tools:
# - Search: for web searches to find current information
# - Scholar: for academic research and paper searches
# - Visit: for visiting and analyzing web pages
# - PythonInterpreter: for running Python code and calculations
# - FileParser: for reading and analyzing files

# For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
# <tool_call>
# {"name": <function-name>, "arguments": <args-json-object>}
# </tool_call>

# For Python code execution, use:
# <tool_call>
# python
# <code>
# # Your Python code here
# print("Hello World")
# </code>
# </tool_call>

# Current date: """


def today_date():
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().date().strftime("%Y-%m-%d")


def extract_competition_id_from_prompt(prompt: str) -> str:
    """
    Extract competition id from prompt text.
    Expected pattern: '## Competition ID: <id>'.
    """
    comp_id = "unknown_competition"
    for line in prompt.splitlines():
        if "## Competition ID:" in line:
            comp_id = line.split("## Competition ID:", 1)[1].strip() or comp_id
            break

    comp_id = comp_id.replace(" ", "_")
    comp_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in comp_id)
    return comp_id or "unknown_competition"


class _Tee:
    """Simple tee to duplicate stdout/stderr to a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def build_text_completion_prompt(messages: list[dict], allow_special: bool = True) -> str:
    """
    Build text completion prompt from messages list.
    Adapted from qwen_agent.utils.utils.build_text_completion_prompt

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        allow_special: Whether to allow special tokens (for compatibility)

    Returns:
        Formatted prompt string
    """
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    prompt_parts = []

    # Handle system message
    if messages and messages[0]["role"] == "system":
        sys_content = messages[0]["content"]
        prompt_parts.append(f"{im_start}system\n{sys_content}{im_end}")
        messages = messages[1:]

    # Ensure chat completes with assistant
    if messages and messages[-1]["role"] != "assistant":
        messages = messages + [{"role": "assistant", "content": ""}]

    # Format each message
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt_parts.append(f"{im_start}{role}\n{content}{im_end}")

    return "\n".join(prompt_parts)


class MultiTurnReactAgent:
    """
    Multi-turn ReAct Agent adapted from Tongyi DeepResearch.

    This agent implements the core reasoning loop with tool calling capabilities,
    using rLLM's OpenAI engine for model inference.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        tools: dict = None,
        system_prompt: str | None = None,
        use_native_function_calling: bool = False,
        **kwargs,
    ):
        """
        Initialize the ReAct agent.

        Args:
            rollout_engine: rLLM OpenAI engine for model inference
            tools: Dictionary of available tools {tool_name: tool_instance}
            system_prompt: Optional custom system prompt
            use_native_function_calling: Whether to use OpenAI native function calling (supports o3)
        """
        self.rollout_engine = rollout_engine
        self.tools = tools or {}
        self.system_prompt = system_prompt
        self.use_native_function_calling = use_native_function_calling
        # Remember the original base output directory once to avoid nesting across tasks
        env_output_dir = os.environ.get("DEEPRESEARCH_OUTPUT_DIR")
        if env_output_dir:
            self.base_output_dir = Path(env_output_dir).resolve()
        else:
            output_root = Path("/fsx/zyhang/rllm/examples/deepresearch/output")
            candidates = sorted(output_root.glob("train-*"), key=lambda p: p.name, reverse=True)
            self.base_output_dir = (candidates[0] if candidates else Path.cwd()).resolve()

        # Convert tools to OpenAI format if using native function calling
        if use_native_function_calling and self.tools:
            self.openai_tools = [tool.json for tool in self.tools.values()]
        else:
            self.openai_tools = None

        # Configuration from original DeepResearch
        self.max_llm_calls = MAX_LLM_CALL_PER_RUN
        self.max_time = 15 * 60  # 15 minutes timeout

        # Smart context management using actual API consumption
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Auto-detect context limit based on model capabilities
        # This ensures we don't hit limits too early for capable models
        self.max_context_tokens = self._get_model_context_limit(rollout_engine)
        # print(f"   🎯 Using max context length: {self.max_context_tokens:,} tokens")

    def _setup_logging(self, output_dir: Path):
        """
        Tee stdout/stderr to the given output directory.
        Supports reconfiguration per prompt by swapping the log file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "trajectory.log"

        # Preserve original streams once
        if not hasattr(self, "_orig_stdout"):
            self._orig_stdout = sys.stdout
            self._orig_stderr = sys.stderr

        # If already logging to this path and file is open, keep it to avoid races in parallel runs
        if getattr(self, "_current_log_path", None) == log_path and hasattr(self, "_log_file") and not self._log_file.closed:
            return

        # Open (do not close previous to avoid breaking other tasks still using the same tee)
        self._log_file = open(log_path, "a", encoding="utf-8")

        # Redirect to tee so prints are saved and still visible
        sys.stdout = _Tee(self._orig_stdout, self._log_file)
        sys.stderr = _Tee(self._orig_stderr, self._log_file)

        # print(f"[DeepResearch] Logging stdout/stderr to {log_path}")
        self._current_log_path = log_path

    def _prepare_run_directory(self, question: str) -> Path:
        """
        Create/ensure a competition-specific subfolder under the base output directory.
        To avoid collisions across parallel trajectories for the same competition,
        create a unique run subfolder per call.
        """
        base_dir = getattr(self, "base_output_dir", Path(os.environ.get("DEEPRESEARCH_OUTPUT_DIR", Path.cwd())))
        comp_id = extract_competition_id_from_prompt(question)
        # Use timestamped subdir to isolate multiple trajectories for the same competition
        run_dir = base_dir / comp_id / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _get_model_context_limit(self, rollout_engine) -> int:
        """
        Auto-detect context limit based on model capabilities.
        Uses LiteLLM's model info when available, falls back to conservative estimates.
        Returns 90% of max to leave safety headroom.
        """
        # model_name = rollout_engine.model
        model_name = "qwen3"

        # Method 1: Try LiteLLM's get_model_info (most accurate)
        try:
            import litellm

            model_info = litellm.get_model_info(model_name)
            if model_info and "max_input_tokens" in model_info:
                max_tokens = model_info["max_input_tokens"]
                conservative_limit = int(max_tokens * 0.90)  # Use 90% for safety
                if not hasattr(MultiTurnReactAgent, "_context_limit_reported"):
                    # print(f"   📏 Detected context window: {max_tokens:,} tokens (using 90% = {conservative_limit:,})")
                    MultiTurnReactAgent._context_limit_reported = True
                return conservative_limit
        except Exception:
            # LiteLLM might not have info for all models, that's ok
            pass

        # Method 2: Try tiktoken to get model family info
        try:
            import tiktoken

            # tiktoken.encoding_for_model will throw if model unknown
            encoding = tiktoken.encoding_for_model(model_name)
            # Map known encodings to context limits
            encoding_limits = {
                "cl100k_base": 128 * 1024,  # GPT-4, GPT-3.5-turbo-16k
                "p50k_base": 4 * 1024,  # text-davinci-002/003
                "r50k_base": 4 * 1024,  # GPT-3 base models
            }
            if encoding.name in encoding_limits:
                max_tokens = encoding_limits[encoding.name]
                conservative_limit = int(max_tokens * 0.90)
                if not hasattr(MultiTurnReactAgent, "_context_limit_reported"):
                    # print(f"   📏 Inferred context from encoding '{encoding.name}': {conservative_limit:,} tokens")
                    MultiTurnReactAgent._context_limit_reported = True
                return conservative_limit
        except Exception:
            pass

        # Method 3: Pattern matching fallback (least accurate but works)
        model_lower = model_name.lower()
        fallback_limits = {
            # OpenAI reasoning models
            ("o3", "o1"): 128 * 1024,
            # GPT-4 family
            ("gpt-4o", "gpt-4-turbo"): 128 * 1024,
            ("gpt-4-32k",): 32 * 1024,
            ("gpt-4",): 8 * 1024,
            # Claude family
            ("claude-3-5", "claude-3.5"): 200 * 1024,
            ("claude-3",): 200 * 1024,
            ("claude-2",): 100 * 1024,
            # Gemini family
            ("gemini-1.5", "gemini-2"): 1000 * 1024,
            ("gemini",): 32 * 1024,
            # Qwen
            ("qwen2", "qwen-2"): 128 * 1024,
            ("qwen",): 32 * 1024,
            ("qwen3"): 40960,
        }

        for patterns, max_tokens in fallback_limits.items():
            if any(pattern in model_lower for pattern in patterns):
                conservative_limit = int(max_tokens * 0.90)
                if not hasattr(MultiTurnReactAgent, "_context_limit_reported"):
                    # print(f"   📏 Pattern-matched context limit: {conservative_limit:,} tokens (90% of {max_tokens:,})")
                    MultiTurnReactAgent._context_limit_reported = True
                return conservative_limit

        # Method 4: Ultimate fallback
        default_limit = 100 * 1024
        if not hasattr(MultiTurnReactAgent, "_context_limit_reported"):
            # print(f"   ⚠️  Unknown model '{model_name}', using conservative default: {default_limit:,} tokens")
            MultiTurnReactAgent._context_limit_reported = True
        return default_limit

    def sanity_check_output(self, content: str) -> bool:
        """Check if the model output contains the expected thinking structure."""
        return "<think>" in content and "</think>" in content

    async def call_server(self, messages: list[dict], max_tries: int = 2, timeout_s: int = 120):
        """
        Call rLLM OpenAI engine with hybrid mode support.

        Supports both:
        - Native function calling (for o3, gpt-4-turbo)
        - ReAct text format (for gpt-4o, Claude)

        Args:
            messages: List of chat completion messages
            max_tries: Maximum number of retry attempts

        Returns:
            ModelOutput with text and tool_calls
        """
        try:
            api_params = {"messages": messages}
            task = asyncio.create_task(self.rollout_engine.get_model_response(**api_params))
            try:
                response = await asyncio.wait_for(task, timeout=timeout_s)
            except asyncio.TimeoutError:
                print(f"{self.competition_id}: [DeepResearch] call_server timed out after {timeout_s} seconds.")
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                return None
            if hasattr(response, "prompt_length") and hasattr(response, "completion_length"):
                self.total_prompt_tokens += response.prompt_length
                self.total_completion_tokens += response.completion_length
            return response
        except Exception:
            return None

    def _estimate_context_tokens(self, messages: list[dict]) -> int:
        """
        Roughly estimate tokens for the current message history.
        Tries tiktoken for the active model; falls back to a character heuristic.
        """
        text_chunks = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_chunks.append(str(part.get("text", "")))
            else:
                text_chunks.append(str(content))

        full_text = "\n".join(text_chunks)

        try:
            import tiktoken

            # encoding = tiktoken.encoding_for_model(self.rollout_engine.model)
            encoding = tiktoken.encoding_for_model("qwen3")
            return len(encoding.encode(full_text, disallowed_special=()))
        except Exception:
            # Heuristic: ~4 chars per token
            return max(1, len(full_text) // 4)

    def get_total_tokens_used(self, messages: list[dict] | None = None) -> int:
        """
        Get total tokens consumed so far from actual API usage.
        If messages are provided, return an estimate based on current context
        (useful after pruning history).

        Returns:
            Total tokens used (prompt + completion)
        """
        if messages is not None:
            return self._estimate_context_tokens(messages)
        return self.total_prompt_tokens + self.total_completion_tokens


    def _prune_failed_tool_turns(self, messages: list[dict]) -> list[dict]:
        """
        Remove assistant/tool-response pairs where the tool failed (error in tool output)
        to reclaim context without losing successful steps.
        """
        if len(messages) <= 2:
            return messages

        pruned = messages[:2]  # preserve system + original user prompt
        i = 2
        while i < len(messages):
            msg = messages[i]

            # ReAct text format: assistant tool_call followed by user tool_response
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and "<tool_call>" in msg.get("content", "")
                and i + 1 < len(messages)
                and isinstance(messages[i + 1], dict)
                and messages[i + 1].get("role") == "user"
                and "<tool_response>" in messages[i + 1].get("content", "")
            ):
                tool_resp = messages[i + 1].get("content", "")
                if "error" in tool_resp.lower():
                    i += 2
                    continue  # drop failed turn
                pruned.extend([msg, messages[i + 1]])
                i += 2
                continue

            # Native function calling: assistant with tool_calls followed by tool role response(s)
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and msg.get("tool_calls")
                and i + 1 < len(messages)
                and isinstance(messages[i + 1], dict)
                and messages[i + 1].get("role") == "tool"
            ):
                tool_resp = messages[i + 1].get("content", "")
                if "error" in str(tool_resp).lower():
                    i += 2
                    continue  # drop failed turn
                pruned.extend([msg, messages[i + 1]])
                i += 2
                continue

            pruned.append(msg)
            i += 1

        return pruned

    async def _run(self, question: str, answer: str = None, images: list = None, **kwargs) -> dict:
        """
        Main reasoning loop adapted from original DeepResearch.

        This is the core ReAct implementation that handles:
        - Multi-turn conversation
        - Tool calling and execution
        - Context length management
        - Termination conditions

        Args:
            question: The research question to answer
            answer: Ground truth answer (for evaluation)
            images: List of image data URLs (base64 encoded)

        Returns:
            Dictionary with results including messages, prediction, and termination reason
        """
        start_time = time.time()

        # Prepare competition-specific output directory and logging
        self.competition_id = extract_competition_id_from_prompt(question)
        run_dir = self._prepare_run_directory(question)
        self.current_run_dir = run_dir
        self._setup_logging(run_dir)

        # Setup system prompt with current date
        system_prompt = self.system_prompt + today_date()
        # print(f"📜 System Prompt:\n{system_prompt}\n")

        # Construct initial user message (multimodal if images present)
        if images:
            # Build multimodal message with images
            user_content = [{"type": "text", "text": question}]
            for image_data in images:
                user_content.append({"type": "image_url", "image_url": {"url": image_data}})
            user_message = {"role": "user", "content": user_content}
        else:
            # Plain text message
            user_message = {"role": "user", "content": question}

        messages = [
            {"role": "system", "content": system_prompt},
            user_message,
        ]

        num_llm_calls_available = self.max_llm_calls
        round = 0
        termination = None
        prediction = ""

        # Truncate question for display
        q_display = str(question).replace("\n", " ").strip()
        if len(q_display) > 200:
            q_display = q_display[:200] + "..."
        # print(f"🔍 Starting DeepResearch for question: {q_display}")

        while num_llm_calls_available > 0:
            # Check time limit (150 minutes)
            if time.time() - start_time > self.max_time:
                prediction = f"No answer found after {self.max_time} seconds"
                print(f"{self.competition_id}: [DeepResearch] Time limit exceeded ({self.max_time} seconds). Terminating.")
                termination = f"timeout"
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                }
                return result

            round += 1
            num_llm_calls_available -= 1

            # Get model response (ModelOutput with text and tool_calls)
            print(f"{self.competition_id}: [DeepResearch] Starting LLM call {round}.")
            response = await self.call_server(messages)
            print(f"{self.competition_id}: [DeepResearch] Round {round} LLM call completed.")
            if response is None:
                prediction = "No answer found due to call_server timeout/None response"
                termination = f"timeout"
                print(f"{self.competition_id}: [DeepResearch] call_server returned None or timed out.")
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                }
                return result

            # Extract text content (may be None for pure function calling)
            content = response.text if hasattr(response, "text") and response.text else ""

            # remove the reasoning part in the content
            if "<think>" in content and "</think>" in content:
                start_idx = content.find("<think>")
                end_idx = content.find("</think>") + len("</think>")
                content = content[:start_idx] + content[end_idx:]

            # Debug: Print raw model response to see format
            # if round == 1:
                # print(f"[DEBUG] Raw model response (first 500 chars): {content[:500]}")
                # if hasattr(response, "tool_calls") and response.tool_calls:
                #     print(f"[DEBUG] Native tool_calls detected: {len(response.tool_calls)} call(s)")

            # Print concise round info with truncation
            MAX_PRINT_LENGTH = 200

            # Simple truncation for all prints
            def truncate(text, max_len=MAX_PRINT_LENGTH):
                text = str(text).replace("\n", " ").strip()
                # Special handling for base64 images
                if "data:image" in text or ";base64," in text:
                    # Find the base64 part and truncate it
                    if "base64," in text:
                        parts = text.split("base64,", 1)
                        return parts[0] + "base64,[truncated]"
                    return "[base64 image data]"
                if len(text) > max_len:
                    return text[:max_len] + "..."
                return text

            # Print round info based on content type
            if "<tool_call>" in content:
                # Extract tool name for display
                if "python" in content.lower() or "<code>" in content.lower():
                    print(f"{self.competition_id}: [DeepResearch] Round {round}: 🐍 Executing Python code")
                    # print(f"Model response: {content}")
                elif '"name":' in content:
                    try:
                        import json5

                        tool_text = content.split("<tool_call>")[1].split("</tool_call>")[0]
                        tool_text = tool_text[:1000]  # Limit for parsing
                        tool_data = json5.loads(tool_text)
                        tool_name = tool_data.get("name", "Unknown")
                        if "arguments" in tool_data:
                            args_str = truncate(str(tool_data["arguments"]), 100)
                            print(f"{self.competition_id}: [DeepResearch] Round {round}: 🔧 Calling {tool_name} with args: {args_str}")
                        else:
                            print(f"{self.competition_id}: [DeepResearch] Round {round}: 🔧 Calling {tool_name}")
                    except Exception as e:
                        print(f"{self.competition_id}: [DeepResearch] Error parsing tool call: {e}")
                        print(f"{self.competition_id}: [DeepResearch] Round {round}: 🔧 Tool call")
                else:
                    print(f"{self.competition_id}: [DeepResearch] Round {round}: 🔧 Tool call, Content snippet: {truncate(content, 100)}")
            elif "<answer>" in content:
                # Final answer
                answer_preview = content.split("<answer>")[1].split("</answer>")[0]
                print(f"{self.competition_id}: [DeepResearch] Round {round}: ✅ Final answer: {content}")
            else:
                # Show internal reasoning if available, otherwise show content
                if hasattr(response, "reasoning") and response.reasoning:
                    reasoning_preview = truncate(response.reasoning, 300)
                    print(f"{self.competition_id}: [DeepResearch] Round {round}: 💭 [Internal] {reasoning_preview}")
                elif content:
                    content_preview = truncate(content, 300)
                    print(f"{self.competition_id}: [DeepResearch] Round {round}: 💭 Reasoning: {content_preview}")

            # Clean up content if it contains tool_response
            if "<tool_response>" in content:
                pos = content.find("<tool_response>")
                content = content[:pos]

            # HYBRID MODE: Handle both native tool_calls and ReAct text format

            # # Priority 1: Check for native function calling (o3, gpt-4-turbo)
            # if hasattr(response, "tool_calls") and response.tool_calls:
            #     # Native function calling path - build ALL messages first, then append atomically
            #     tool_calls_formatted = []
            #     tool_responses = []

            #     for tool_call in response.tool_calls:
            #         try:
            #             # Follow strands.py tolerant extraction of function/name/arguments
            #             try:
            #                 function = tool_call.get("function", {}) if isinstance(tool_call, dict) else getattr(tool_call, "function", {})
            #             except Exception:
            #                 function = tool_call

            #             tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", "unknown")
            #             tool_name = function.get("name") if isinstance(function, dict) else getattr(function, "name", "")
            #             arguments_raw = function.get("arguments") if isinstance(function, dict) else getattr(function, "arguments", "{}")

            #             # Parse arguments if provided as JSON string
            #             tool_args = json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw

            #             # Print tool call with arguments (for consistency with ReAct format)
            #             def truncate(text, max_len=100):
            #                 text = str(text).replace("\n", " ").strip()
            #                 if len(text) > max_len:
            #                     return text[:max_len] + "..."
            #                 return text

            #             args_str = truncate(str(tool_args), 100)
            #             print(f"Round {round}: 🔧 [Native] Calling {tool_name} with args: {args_str}")

            #             # Execute tool
            #             result = await self.custom_call_tool(tool_name, tool_args)

            #             # Collect tool call and response (don't append yet)
            #             tool_calls_formatted.append(
            #                 {
            #                     "id": tool_id,
            #                     "type": "function",
            #                     "function": {
            #                         "name": tool_name,
            #                         "arguments": arguments_raw,
            #                     },
            #                 }
            #             )
            #             tool_responses.append({"role": "tool", "tool_call_id": tool_id, "content": result})
            #             print(f"Round {round}: 🛠️ [Native] Tool {tool_name} returned: {str(result)}")

            #         except Exception as e:
            #             print(f"Error processing native tool call: {e}")
            #             # On error, append error message and skip this tool call
            #             messages.append({"role": "assistant", "content": content.strip()})
            #             messages.append({"role": "user", "content": f"Tool call error: {e}"})
            #             continue

            #     # Only append to messages if we have successful tool calls
            #     if tool_calls_formatted:
            #         # Add assistant message with ALL tool calls at once
            #         messages.append(
            #             {
            #                 "role": "assistant",
            #                 "content": content or "",  # May be empty for pure function calling
            #                 "tool_calls": tool_calls_formatted,
            #             }
            #         )
            #         # Add all tool responses
            #         messages.extend(tool_responses)

            # Priority 2: Check for ReAct text format (gpt-4o, Claude)
            if "<tool_call>" in content and "</tool_call>" in content:
                # ReAct text format path
                messages.append({"role": "assistant", "content": content.strip()})

                tool_call_text = content.split("<tool_call>")[1].split("</tool_call>")[0]
                try:
                    # Special handling for Python code (match original logic)
                    if "python" in tool_call_text.lower() or "<code>" in tool_call_text.lower():
                        try:
                            # Extract code from the original content (not just tool_call_text)
                            code_blocks = re.findall(r"<code>(.*?)</code>", content, flags=re.DOTALL | re.IGNORECASE)
                            if not code_blocks:
                                raise ValueError("No <code> blocks found in tool call.")
                            merged_code = "\n\n".join(block.strip() for block in code_blocks if block.strip())
                            result = await self.execute_python(merged_code)
                            print(f"{self.competition_id}: [DeepResearch] Round {round}: 🐍 Python execution finished")
                        except Exception:
                            result = "[Python Interpreter Error]: Formatting error. You must wrap your code within <code></code> tags."
                    else:
                        # Parse JSON tool call
                        tool_call = json5.loads(tool_call_text)
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("arguments", {})
                        result = await self.custom_call_tool(tool_name, tool_args)
                        print(f"{self.competition_id}: [DeepResearch] Round {round}: 🛠️ Tool {tool_name} returned: {str(result)}")

                except Exception:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'

                # Add tool response in ReAct format
                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                # print(f"Round {round}: Tool response: {tool_response}")
                messages.append({"role": "user", "content": tool_response})

            # Priority 3: No tool call, just reasoning or answer
            else:
                if "<answer>" in content and "</answer>" in content:
                    content += f"\n<output>{self.current_run_dir}</output>"  # Ensure output tag for final answer
                messages.append({"role": "assistant", "content": content.strip()})

            # Check for final answer AFTER processing tools
            # This allows o3 to execute tools even when it includes answer in same message
            if "<answer>" in content and "</answer>" in content:
                prediction = content.split("<answer>")[1].split("</answer>")[0].strip()
                termination = "answer"
                break

            # Check if we've exceeded call limit
            if num_llm_calls_available <= 0 and "<answer>" not in content:
                # Handle both message formats
                if isinstance(messages[-1], dict) and "content" in messages[-1]:
                    messages[-1]["content"] = "Sorry, the number of llm calls exceeds the limit."

            # Handle context length limit using current message context
            total_tokens_used = self.get_total_tokens_used(messages)

            if total_tokens_used > self.max_context_tokens:
                before = len(messages)
                messages = self._prune_failed_tool_turns(messages)
                self.total_prompt_tokens = self._estimate_context_tokens(messages)
                self.total_completion_tokens = 0
                removed = before - len(messages)
                if removed > 0:
                    print(f"{self.competition_id}: [DeepResearch] Round {round}: 🧹 Pruned {removed} message(s) with failed tool outputs to save context.")
                    continue  # retry loop with cleaned messages

                # Fallback: keep system, original user, and last few exchanges
                if len(messages) > 2:
                    messages = messages[:2] + messages[-2:]
                    self.total_prompt_tokens = self._estimate_context_tokens(messages)
                    self.total_completion_tokens = 0
                    # print(f"Round {round}: ⚠️ Context still high; keeping recent history only.")
                    continue

            #     # Instead of replacing the last message, add a clear instruction
            #     final_instruction = {
            #         "role": "user",
            #         "content": "You have reached the maximum context length. Based on all the information above, please provide your best answer now in the format: <think>your final thinking</think>\n<answer>your answer</answer>",
            #     }

            #     # Truncate conversation history to make room for final answer
            #     # Keep system prompt, original question, and recent context
            #     if len(messages) > 4:  # system + user + at least 2 exchanges
            #         # Keep first 2 messages (system + original question) and last 2 meaningful exchanges
            #         truncated_messages = messages[:2]  # system + original question
            #         recent_messages = messages[-4:]  # last 4 messages for context
            #         truncated_messages.extend(recent_messages)
            #         messages = truncated_messages

            #     messages.append(final_instruction)

            #     # Note: After truncation, we'll let the next API call handle any remaining limits
            #     print(f"Round {round + 1}: ⚠️ Context limit reached, requesting final answer")

            #     response = await self.call_server(messages)
            #     content = response.text if hasattr(response, "text") and response.text else ""
            #     messages.append({"role": "assistant", "content": content.strip()})

            #     if "<answer>" in content and "</answer>" in content:
            #         prediction = content.split("<answer>")[1].split("</answer>")[0].strip()
            #         termination = "answer generated due to token limit"
            #     else:
            #         prediction = content.strip()
            #         termination = "response generated due to token limit (no answer format)"

            #     result = {
            #         "question": question,
            #         "answer": answer,
            #         "messages": messages,
            #         "prediction": prediction,
            #         "termination": termination,
            #     }
            #     return result

        # Final validation logic from original Tongyi implementation
        # Handle both native function calling and ReAct text format
        last_message_content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if last_message_content and "<answer>" in last_message_content:
            prediction = last_message_content.split("<answer>")[1].split("</answer>")[0]
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = "answer not found"
            if num_llm_calls_available == 0:
                termination = "token_limit_no_answer"

        # Final result
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "rounds": round,
            "time_taken": time.time() - start_time,
        }

        # print("\n🏁 DeepResearch completed:")
        # print(f"   Rounds: {round}")
        # print(f"   Time: {result['time_taken']:.1f}s")
        # print(f"   Termination: {termination}")
        # Truncate prediction for display
        pred_display = str(prediction).replace("\n", " ").strip()
        if len(pred_display) > 200:
            pred_display = pred_display[:200] + "..."
        # print(f"   Prediction: {pred_display}")

        return result

    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs) -> str:
        """
        Execute tool calls with the available tools.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool

        Returns:
            Tool execution result as string
        """
        if tool_name in self.tools:
            try:
                # Call the tool
                if hasattr(self.tools[tool_name], "call"):
                    # Async tool
                    call_kwargs = dict(tool_args)
                    # Pass run_dir hint for tools that accept it (PythonInterpreter, Score)
                    if getattr(self, "current_run_dir", None) is not None:
                        call_kwargs.setdefault("run_dir", self.current_run_dir)

                    if asyncio.iscoroutinefunction(self.tools[tool_name].call):
                        result = await self.tools[tool_name].call(**call_kwargs)
                    else:
                        result = self.tools[tool_name].call(**call_kwargs)
                elif callable(self.tools[tool_name]):
                    # Direct callable
                    result = self.tools[tool_name](**tool_args)
                else:
                    result = f"Tool {tool_name} is not callable"

                return str(result)

            except Exception as e:
                return f"Error calling tool {tool_name}: {e}"
        else:
            available_tools = list(self.tools.keys())
            return f"Tool {tool_name} not found. Available tools: {available_tools}"

    async def execute_python(self, code: str) -> str:
        """
        Execute Python code using the PythonInterpreter tool.

        Args:
            code: Python code to execute

        Returns:
            Execution result as string
        """
        timeout_s = 60
        if "PythonInterpreter" in self.tools:
            try:
                # Use the PythonInterpreter tool
                tool = self.tools["PythonInterpreter"]
                if hasattr(tool, "call"):
                    task = asyncio.create_task(
                        tool.call(code=code, run_dir=getattr(self, "current_run_dir", None))
                    )
                    try:
                        result = await asyncio.wait_for(task, timeout=timeout_s)
                    except asyncio.TimeoutError:
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=2)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                        print(f"{self.competition_id}: [DeepResearch] Python execution timed out after {timeout_s}s.")
                        return f"[Timeout] Exceeded {timeout_s}s."
                    return str(result)
                else:
                    return "PythonInterpreter tool is not callable"
            except Exception as e:
                return f"Python execution error: {e}"
        else:
            return "PythonInterpreter tool not available"

    def reset(self):
        """Reset the agent state (for compatibility with rLLM workflow)."""
        # Reset token counters for each new task
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def run(self, question: str, answer: str = None, **kwargs) -> dict:
        """
        Public interface for running the agent.

        Args:
            question: Research question to answer
            answer: Ground truth answer (optional, for evaluation)

        Returns:
            Result dictionary
        """
        # Reset token counters for each new run
        self.reset()
        return await self._run(question, answer, **kwargs)
