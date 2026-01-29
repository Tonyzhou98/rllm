"""
DeepResearch Tools - Production-ready implementations

This module provides tool implementations for the DeepResearch agent, with real
functionality ported from Tongyi's original implementations where possible.

Now supports both:
- ReAct text format (for gpt-4o, Claude, etc.)
- OpenAI native function calling (for o3, o3-mini, etc.)
"""

import http.client
import asyncio
import json
import os
import random
import subprocess
import signal
import logging
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import httpx

from rllm.tools.tool_base import Tool as RLLMTool


logging.getLogger("httpx").setLevel(logging.WARNING)

SRUN_API_URL = os.environ.get("SRUN_API_URL", "http://127.0.0.1:9000")
DEFAULT_TOOL_TIMEOUT = 60  # seconds for the tool-level timeout
POLL_INTERVAL = 2.0  # seconds between status polls to SRUN API

async def submit_script_to_srun_api(
    script_path: Path,
    *,
    conda_env: str = "algoevolve",
    time: str = "02:00:00",
    cpus: int = 8,
    mem: str = "32G",
    gres_gpus: str = "gpu:1",
    pre_cmds: Optional[str] = None,
    tool_timeout: int = 60,
    sruns_api_url: str = SRUN_API_URL,
    job_name: Optional[str] = None,
) -> str:
    """
    Submit existing script_path to SRUN API and poll until completion or the tool-level timeout.
    Returns a result string matching the old tool format:
     - "[Output]\\n{stdout_tail}"
     - "[Error] {stderr}" and path to logs
     - "[Timeout] ..." etc.
    """
    if not script_path.exists() or not script_path.is_file():
        return f"[Error] script not found: {script_path}"

    payload = {
        "script_path": str(script_path),
        "time": time,
        "cpus": cpus,
        "mem": mem,
        "gres_gpus": gres_gpus,
        "conda_env": conda_env,
        "pre_cmds": pre_cmds,
        "job_name": job_name or os.environ.get("DEEPRESEARCH_API_JOB_NAME", "deepresearch_api_job"),
        "timeout": 0,  # let API manage process lifetime; we enforce a tool-level timeout here
    }

    # Call SRUN API /run
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"{sruns_api_url.rstrip('/')}/run", json=payload)
        except Exception as e:
            return f"[Error] Failed to contact SRUN API at {sruns_api_url}: {e}"
        if resp.status_code != 200:
            return f"[Error] SRUN API returned {resp.status_code}: {resp.text}"

        meta = resp.json()
        job_id = meta.get("job_id")
        workdir = Path(meta.get("workdir", str(script_path.parent)))
        stdout_path = Path(meta.get("stdout")) if meta.get("stdout") else None
        stderr_path = Path(meta.get("stderr")) if meta.get("stderr") else None

    # Poll for completion and collect log tails
    start = datetime.now(timezone.utc)
    timed_out = False
    stdout_tail = ""
    stderr_tail = ""

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            if tool_timeout and tool_timeout > 0 and elapsed > tool_timeout:
                timed_out = True
                # try to cancel via API
                try:
                    await client.post(f"{sruns_api_url.rstrip('/')}/cancel/{job_id}")
                except Exception:
                    pass
                break

            # get status
            try:
                s = await client.get(f"{sruns_api_url.rstrip('/')}/status/{job_id}")
            except Exception:
                await asyncio.sleep(POLL_INTERVAL)
                continue

            if s.status_code != 200:
                # status may not be available immediately; retry
                await asyncio.sleep(POLL_INTERVAL)
                continue

            st = s.json()
            state = st.get("state", "").upper()

            # fetch logs (tail)
            try:
                l = await client.get(f"{sruns_api_url.rstrip('/')}/logs/{job_id}", params={"tail_lines": 200})
                if l.status_code == 200:
                    logs = l.json()
                    stdout_tail = logs.get("stdout_tail", "") or ""
                    stderr_tail = logs.get("stderr_tail", "") or ""
            except Exception:
                # ignore transient log fetch errors
                pass

            if state in ("RUNNING", "PENDING"):
                await asyncio.sleep(POLL_INTERVAL)
                continue

            # completed or unknown
            break

    # helper truncate function
    def _truncate_block(text: str, max_lines: int = 20) -> str:
        if not text:
            return text
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text
        tail = lines[-max_lines:]
        truncated = len(lines) - max_lines
        return "\n".join(tail + [f"...[truncated {truncated} lines, see log for full output]"])

    stdout_tail = _truncate_block(stdout_tail.strip() if stdout_tail else "")
    stderr_tail = _truncate_block(stderr_tail.strip() if stderr_tail else "")

    # prefer submission.csv if exists
    submission_path = script_path.parent / "submission.csv"

    if timed_out:
        return f"[Timeout] Exceeded {tool_timeout}s. Logs: {stdout_path or script_path.parent}"

    if stdout_tail:
        if stderr_tail:
            if submission_path.exists():
                return f"[Output]\n{stdout_tail}\n\n[Stderr]\n{stderr_tail}\n\nNote: submission.csv found at {submission_path}."
            else:
                return f"[Output]\n{stdout_tail}\n\n[Stderr]\n{stderr_tail}\n\nNote: submission.csv not found."
        else:
            if submission_path.exists():
                return f"[Output]\n{stdout_tail}\n\nNote: no stderr, and submission.csv found at {submission_path}."
            else:
                return f"[Output]\n{stdout_tail}\n\nNote: no stderr, and submission.csv not found."

    if stderr_tail:
        if submission_path.exists():
            return f"No stdout, but submission.csv found at {submission_path}. Stderr:\n{stderr_tail}"
        return f"No stdout and submission.csv not found. Stderr:\n{stderr_tail}"

    if submission_path.exists():
        return f"No stdout, but submission.csv found at {submission_path}."

    # fallback: if API provided stdout file, try to read last lines
    if stdout_path and stdout_path.exists():
        try:
            text = stdout_path.read_text(errors="replace").splitlines()[-20:]
            return "[Output]\n" + "\n".join(text)
        except Exception:
            pass

    return "No stdout and submission.csv not found."


class DeepResearchTool(RLLMTool, ABC):
    """
    Base class for all DeepResearch tools.

    Inherits from rLLM's Tool to support OpenAI native function calling,
    while maintaining compatibility with ReAct text format.
    """

    def __init__(self, name: str, description: str, parameters: dict | None = None):
        """
        Initialize DeepResearch tool with OpenAI function calling support.

        Args:
            name: Tool name
            description: Tool description
            parameters: OpenAI-style parameter schema (optional)
        """
        # Set _json BEFORE calling super().__init__
        # because the parent's __init__ may access self.json
        self._json = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters or {"type": "object", "properties": {}, "required": []},
            },
        }

        super().__init__(name=name, description=description)

    @abstractmethod
    async def call(self, **kwargs) -> str:
        """Execute the tool with given arguments."""
        pass

    async def async_forward(self, **kwargs):
        """rLLM Tool interface - delegates to call()"""
        from rllm.tools.tool_base import ToolOutput

        try:
            result = await self.call(**kwargs)
            return ToolOutput(name=self.name, output=result)
        except Exception as e:
            return ToolOutput(name=self.name, error=f"{type(e).__name__} - {str(e)}")


class SearchTool(DeepResearchTool):
    """Web search tool using Serper API (ported from Tongyi)."""

    def __init__(self):
        super().__init__(
            name="Search",
            description="Performs web searches using Google via Serper API",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string",
                    }
                },
                "required": ["query"],
            },
        )

    def contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _google_search_fallback(self, query: str | list) -> str:
        """Use Google Custom Search API as fallback."""
        try:
            import requests

            google_key = os.getenv("GOOGLE_SEARCH_SECRET_KEY")
            engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

            queries = [query] if isinstance(query, str) else query
            all_results = []

            for q in queries:
                params = {"key": google_key, "cx": engine_id, "q": q, "num": 10}

                response = requests.get(
                    "https://customsearch.googleapis.com/customsearch/v1",
                    params=params,
                    timeout=5,
                )

                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])

                    web_snippets = []
                    for idx, item in enumerate(items[:10], 1):
                        title = item.get("title", "")
                        link = item.get("link", "")
                        snippet = item.get("snippet", "")
                        entry = f"{idx}. [{title}]({link})\n   {snippet}"
                        web_snippets.append(entry)

                    result = f"Google search for '{q}' found {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
                    all_results.append(result)
                else:
                    all_results.append(f"Google search error for '{q}': {response.status_code}")

            return "\n=======\n".join(all_results) if len(all_results) > 1 else all_results[0]

        except Exception as e:
            return f"Google search fallback error: {e}"

    async def call(self, query: str | list, **kwargs) -> str:
        """
        Search the web using Serper API or Google Custom Search.

        Args:
            query: Search query string or list of queries

        Returns:
            Formatted search results
        """
        api_key = os.getenv("SERPER_API_KEY")

        # Try Google Custom Search as fallback if no Serper key
        if not api_key:
            google_key = os.getenv("GOOGLE_SEARCH_SECRET_KEY")
            google_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

            if google_key and google_engine_id:
                return self._google_search_fallback(query)

            return f"""[Search - API Key Required]

To enable real web search, use one of these options:

Option 1 - Serper (Recommended, simpler):
1. Get a free API key from https://serper.dev (2500 searches/month free)
2. Add to .env: SERPER_API_KEY=your_key_here

Option 2 - Google Custom Search:
1. Set up at https://developers.google.com/custom-search
2. Add to .env:
   GOOGLE_SEARCH_SECRET_KEY=your_key
   GOOGLE_SEARCH_ENGINE_ID=your_engine_id

Placeholder results for '{query}'..."""

        # Handle single query or list
        queries = [query] if isinstance(query, str) else query
        all_results = []

        for q in queries:
            try:
                conn = http.client.HTTPSConnection("google.serper.dev")

                # Localize for Chinese queries
                if self.contains_chinese(q):
                    payload = json.dumps({"q": q, "location": "China", "gl": "cn", "hl": "zh-cn"})
                else:
                    payload = json.dumps({"q": q, "location": "United States", "gl": "us", "hl": "en"})

                headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

                # Retry logic
                for i in range(5):
                    try:
                        conn.request("POST", "/search", payload, headers)
                        res = conn.getresponse()
                        break
                    except Exception:
                        if i == 4:
                            all_results.append(f"Google search timeout for '{q}'")
                            continue

                data = res.read()
                results = json.loads(data.decode("utf-8"))

                if "organic" not in results:
                    all_results.append(f"No results found for '{q}'")
                    continue

                web_snippets = []
                for idx, page in enumerate(results.get("organic", [])[:10], 1):
                    date_published = f"\nDate: {page['date']}" if "date" in page else ""
                    source = f"\nSource: {page['source']}" if "source" in page else ""
                    snippet = f"\n{page['snippet']}" if "snippet" in page else ""

                    entry = f"{idx}. [{page.get('title', 'Untitled')}]({page.get('link', '')}){date_published}{source}{snippet}"
                    web_snippets.append(entry)

                content = f"Google search for '{q}' found {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
                all_results.append(content)

            except Exception as e:
                all_results.append(f"Search error for '{q}': {e}")

        return "\n=======\n".join(all_results) if len(all_results) > 1 else all_results[0]


class ScholarTool(DeepResearchTool):
    """Google Scholar search using Serper API (ported from Tongyi)."""

    def __init__(self):
        super().__init__(
            name="Scholar",
            description="Search Google Scholar for academic papers",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The academic search query",
                    }
                },
                "required": ["query"],
            },
        )

    async def call(self, query: str | list, **kwargs) -> str:
        """
        Search Google Scholar using Serper API.

        Args:
            query: Search query string or list of queries

        Returns:
            Academic search results
        """
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return """[Scholar - API Key Required]

To enable Google Scholar search, configure SERPER_API_KEY in your .env file."""

        queries = [query] if isinstance(query, str) else query
        all_results = []

        for q in queries:
            try:
                conn = http.client.HTTPSConnection("google.serper.dev")
                payload = json.dumps({"q": q, "type": "scholar", "num": 10})
                headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

                conn.request("POST", "/scholar", payload, headers)
                res = conn.getresponse()
                data = res.read()
                results = json.loads(data.decode("utf-8"))

                if "organic" not in results:
                    all_results.append(f"No scholar results for '{q}'")
                    continue

                papers = []
                for idx, paper in enumerate(results.get("organic", [])[:10], 1):
                    title = paper.get("title", "Untitled")
                    link = paper.get("link", "")
                    snippet = paper.get("snippet", "")
                    publication = paper.get("publication", "")
                    year = paper.get("year", "")
                    cited_by = paper.get("citedBy", {}).get("value", 0)

                    entry = f"{idx}. [{title}]({link})"
                    if publication:
                        entry += f"\n   Publication: {publication}"
                    if year:
                        entry += f" ({year})"
                    if cited_by:
                        entry += f"\n   Cited by: {cited_by}"
                    if snippet:
                        entry += f"\n   {snippet}"

                    papers.append(entry)

                result_text = f"Google Scholar search for '{q}':\n\n" + "\n\n".join(papers)
                all_results.append(result_text)

            except Exception as e:
                all_results.append(f"Scholar search error for '{q}': {e}")

        return "\n=======\n".join(all_results) if len(all_results) > 1 else all_results[0]


class VisitTool(DeepResearchTool):
    """Web page visiting with content extraction."""

    def __init__(self):
        super().__init__(
            name="Visit",
            description="Visit and extract content from web pages",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to visit"},
                    "goal": {
                        "type": "string",
                        "description": "Optional goal for the visit",
                    },
                },
                "required": ["url"],
            },
        )

    async def call(self, url: str | list, goal: str = "", **kwargs) -> str:
        """
        Visit web pages and extract content.

        Args:
            url: URL string or list of URLs
            goal: Optional goal for the visit

        Returns:
            Extracted webpage content
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return """[Visit Tool - Dependencies Required]

To enable webpage visiting:
pip install requests beautifulsoup4

Then the tool will fetch and parse webpage content."""

        import re
        from urllib.parse import urlparse

        urls = [url] if isinstance(url, str) else url
        all_results = []

        for target_url in urls[:5]:  # Limit to 5 URLs
            try:
                # Validate and normalize URL
                parsed = urlparse(target_url)
                if not parsed.scheme:
                    target_url = f"https://{target_url}"

                # Fetch webpage
                headers = {"User-Agent": "Mozilla/5.0 (compatible; DeepResearch/1.0)"}
                response = requests.get(target_url, headers=headers, timeout=10)
                response.raise_for_status()

                # Parse HTML
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    element.decompose()

                # Extract title
                title = soup.title.string if soup.title else "No Title"

                # Extract main content
                content = ""
                for selector in ["main", "article", ".content", "#content", ".post"]:
                    element = soup.select_one(selector)
                    if element:
                        content = element.get_text(separator="\n", strip=True)
                        break

                if not content:
                    body = soup.find("body")
                    if body:
                        content = body.get_text(separator="\n", strip=True)

                # Clean up text
                content = re.sub(r"\n{3,}", "\n\n", content)
                content = re.sub(r" {2,}", " ", content)

                # Limit length
                if len(content) > 5000:
                    content = content[:5000] + "\n[Content truncated...]"

                # Format result
                result = f"[Webpage: {target_url}]\nTitle: {title}"
                if goal:
                    result += f"\nGoal: {goal}"
                result += f"\n\nContent:\n{content}"

                all_results.append(result)

            except Exception as e:
                all_results.append(f"[Error visiting {target_url}]: {e}")

        return "\n\n=======\n\n".join(all_results)


class FileParserTool(DeepResearchTool):
    """Enhanced file parsing for multiple formats."""

    def __init__(self):
        super().__init__(
            name="FileParser",
            description="Parse files: TXT, JSON, CSV, PDF, DOCX, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "string",
                        "description": "File path or list of file paths to parse",
                    }
                },
                "required": ["files"],
            },
        )

    async def call(self, files: str | list, **kwargs) -> str:
        """
        Parse files and extract content.

        Args:
            files: File path string or list of paths

        Returns:
            Extracted file content
        """
        import csv
        from pathlib import Path

        file_paths = [files] if isinstance(files, str) else files
        all_results = []

        for file_path in file_paths[:10]:  # Limit to 10 files
            if not os.path.exists(file_path):
                all_results.append(f"Error: File not found at {file_path}")
                continue

            try:
                file_ext = Path(file_path).suffix.lower()
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)

                content = ""

                # Text files
                if file_ext in [
                    ".txt",
                    ".md",
                    ".log",
                    ".py",
                    ".js",
                    ".java",
                    ".cpp",
                    ".c",
                    ".h",
                ]:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                # JSON files
                elif file_ext == ".json":
                    with open(file_path, encoding="utf-8") as f:
                        data = json.load(f)
                        content = json.dumps(data, indent=2, ensure_ascii=False)

                # CSV files
                elif file_ext == ".csv":
                    rows = []
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        reader = csv.reader(f)
                        for i, row in enumerate(reader):
                            if i >= 100:
                                rows.append("[... truncated ...]")
                                break
                            rows.append(", ".join(row))
                    content = "\n".join(rows)

                # PDF files
                elif file_ext == ".pdf":
                    try:
                        import PyPDF2

                        with open(file_path, "rb") as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            pages = []
                            for i in range(min(len(pdf_reader.pages), 10)):
                                page = pdf_reader.pages[i]
                                pages.append(f"Page {i + 1}:\n{page.extract_text()}")
                            content = "\n\n".join(pages)
                    except ImportError:
                        content = "[PDF parsing requires: pip install PyPDF2]"

                # Word documents
                elif file_ext in [".docx", ".doc"]:
                    try:
                        from docx import Document

                        doc = Document(file_path)
                        paragraphs = []
                        for i, para in enumerate(doc.paragraphs):
                            if i >= 100:
                                paragraphs.append("[... truncated ...]")
                                break
                            if para.text.strip():
                                paragraphs.append(para.text)
                        content = "\n\n".join(paragraphs)
                    except ImportError:
                        content = "[DOCX parsing requires: pip install python-docx]"

                # Default: try as text
                else:
                    try:
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                    except Exception:
                        content = f"[Cannot parse file type: {file_ext}]"

                # Limit content
                if len(content) > 10000:
                    content = content[:10000] + "\n[Content truncated...]"

                result = f"[File: {file_name}]\nType: {file_ext}\nSize: {file_size:,} bytes\n\nContent:\n{content}"
                all_results.append(result)

            except Exception as e:
                all_results.append(f"Error parsing {file_path}: {e}")

        return "\n\n=======\n\n".join(all_results)


class ScoreTool(DeepResearchTool):
    """Evaluate submission.csv with mlebench grader."""

    def __init__(self):
        super().__init__(
            name="Score",
            description="Grade submission.csv for Spaceship Titanic using mlebench",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional path to submission.csv; defaults to DEEPRESEARCH_OUTPUT_DIR/submission.csv",
                    }
                },
                "required": [],
            },
        )

    async def call(self, competition_id: str, path: str | None = None, run_dir: str | Path | None = None, **kwargs) -> str:
        """Run mlebench grader and return metrics as a JSON string."""
        # Locate submission
        output_dir = Path(run_dir) if run_dir else Path(os.environ.get("DEEPRESEARCH_OUTPUT_DIR", Path.cwd()))
        submission_path = Path(path) if path else output_dir / "submission.csv"
        if not submission_path.exists():
            return f"[Error] submission file not found at {submission_path}"

        cmd = [
            "mlebench",
            "grade-sample",
            str(submission_path),
            competition_id,
            "--data-dir",
            "/fsx/zyhang/mle-bench-data/",
        ]

        try:
            grade_proc = await asyncio.create_task(
                asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                )
            )
        except FileNotFoundError as e:
            return f"[Error] mlebench not found: {e}"
        except Exception as e:
            return f"[Error] Failed to start grading: {e}"

        output = (grade_proc.stderr or "") + "\n" + (grade_proc.stdout or "")
        output = output.strip()

        if grade_proc.returncode != 0:
            return f"[Error] Grading failed with code {grade_proc.returncode}: {output}"

        metrics = {}

        def parse_number(key: str, text: str):
            if key in text:
                try:
                    return float(text.split(f'"{key}": ')[-1].split(",")[0].strip())
                except Exception:
                    return None
            return None

        score = parse_number("score", output)
        metrics["score_primary (main competition metric for current code)"] = score

        # is_lower_better
        if '"is_lower_better":' in output:
            val = output.split('"is_lower_better": ')[-1].split(",")[0].strip().lower()
            metrics["metric_lower_is_better (true means lower score is better)"] = val == "true"

        # Thresholds: minimum score needed to reach each medal/median tier
        metrics["threshold_gold (score needed for gold tier)"] = parse_number("gold_threshold", output)
        metrics["threshold_silver (score needed for silver tier)"] = parse_number("silver_threshold", output)
        metrics["threshold_bronze (score needed for bronze tier)"] = parse_number("bronze_threshold", output)
        metrics["threshold_median (median submission score)"] = parse_number("median_threshold", output)

        # metrics["raw_output_text"] = output
        metrics["submission_path"] = str(submission_path)

        # Build human-friendly prefix when score is missing; surface grader output directly
        if score is None:
            return f"Submission csv is invalid. Detailed issue: {output}"

        return f"Submission OK. Details: {json.dumps(metrics)}"


class SynScoreTool(DeepResearchTool):
    """Evaluate submission.csv with a local evaluator.py for synthetic datasets."""

    def __init__(self):
        super().__init__(
            name="SynScore",
            description="Grade submission.csv using evaluator.py in the output directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional path to submission.csv; defaults to DEEPRESEARCH_OUTPUT_DIR/submission.csv",
                    }
                },
                "required": [],
            },
        )

    async def call(self, competition_id: str, path: str | None = None, run_dir: str | Path | None = None, **kwargs) -> str:
        """Run evaluator.py from the output directory and return metrics as a JSON string."""
        output_dir = Path(run_dir) if run_dir else Path(os.environ.get("DEEPRESEARCH_OUTPUT_DIR", Path.cwd()))
        submission_path = Path(path) if path else output_dir / "submission.csv"
        evaluator_path = Path("/fsx/zyhang/mle-bench-syn") / competition_id / "prepared" / "public" / "evaluator.py"
        evaluator_dir = evaluator_path.parent

        if submission_path != output_dir / "submission.csv":
            return f"[Error] submission.csv must be located at {output_dir / 'submission.csv'}"
        if not evaluator_path.exists():
            return f"[Error] evaluator.py not found at {evaluator_path}"
        if not submission_path.exists():
            return f"[Error] submission file not found at {submission_path}"

        cmd = ["python", "evaluator.py", "--submission_path", str(submission_path)]

        try:
            eval_proc = await asyncio.create_task(
                asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=evaluator_dir,
                )
            )
        except FileNotFoundError as e:
            return f"[Error] python not found: {e}"
        except Exception as e:
            return f"[Error] Failed to start evaluator: {e}"

        stdout = (eval_proc.stdout or "").strip()
        stderr = (eval_proc.stderr or "").strip()

        if eval_proc.returncode != 0:
            output = "\n".join([line for line in [stderr, stdout] if line])
            return f"[Error] Evaluation failed with code {eval_proc.returncode}: {output}"

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            output = "\n".join([line for line in [stderr, stdout] if line])
            return f"[Error] Failed to parse evaluator output as JSON: {e}. Output: {output}"

        metrics = {}

        def parse_number(key: str):
            value = data.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except Exception:
                return None

        def parse_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            if isinstance(value, str):
                return value.strip().lower() in {"true", "1", "yes"}
            return None

        score = parse_number("score")
        metrics["score_primary (main competition metric for current code)"] = score

        is_lower_better = parse_bool(data.get("is_lower_better"))
        if is_lower_better is not None:
            metrics["metric_lower_is_better (true means lower score is better)"] = is_lower_better

        metrics["threshold_gold (score needed for gold tier)"] = parse_number("gold_threshold")
        metrics["threshold_silver (score needed for silver tier)"] = parse_number("silver_threshold")
        metrics["threshold_bronze (score needed for bronze tier)"] = parse_number("bronze_threshold")
        metrics["threshold_median (median submission score)"] = parse_number("median_threshold")
        metrics["submission_path"] = str(submission_path)

        if score is None:
            output = "\n".join([line for line in [stderr, stdout] if line])
            return f"Submission csv is invalid. Detailed issue: {output}"

        return f"Submission OK. Details: {json.dumps(metrics)}"


class PythonInterpreterTool(DeepResearchTool):
    """Safe Python code execution (from existing implementation)."""

    def __init__(self, timeout: int = 120, job_name: str | None = None):
        super().__init__(
            name="PythonInterpreter",
            description="Execute Python code for calculations and analysis",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                "required": ["code"],
            },
        )
        self.timeout = timeout  # Default timeout in seconds
        self.job_name = job_name

    async def call(self, code: str, timeout: int = None, run_dir: str | Path | None = None, **kwargs) -> str:
        """
        Execute Python code by writing it to main.py and launching via python or srun.
        Captures stdout/stderr to both memory and a log file under the per-run output dir.
        """
        timeout = timeout or self.timeout

        # Resolve run directory under DEEPRESEARCH_OUTPUT_DIR so outputs align with submission/logs
        run_dir = Path(run_dir) if run_dir else Path(os.environ.get("DEEPRESEARCH_OUTPUT_DIR", Path.cwd()))
        run_dir.mkdir(parents=True, exist_ok=True)

        # Use timestamped filenames to avoid collisions while keeping them in the same folder
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        script_filename = f"main_{ts}.py"
        script_path = run_dir / script_filename
        log_path = run_dir / f"main_{ts}.log"

        script_path.write_text(code, encoding="utf-8")
        conda_env = os.environ.get("DEEPRESEARCH_CONDA_ENV", "algoevolve")

        result = await submit_script_to_srun_api(
            script_path,
            conda_env=conda_env,
            time="02:00:00",
            cpus=8,
            mem="32G",
            gres_gpus="gpu:1",
            pre_cmds=None,
            tool_timeout=timeout,
            sruns_api_url=SRUN_API_URL,
            job_name=self.job_name,
        )
        return result

        # bash_cmd = f"source ~/miniconda3/bin/activate && conda activate {conda_env} && python -u {script_filename} && conda deactivate && conda activate rllm"
        # use_srun = False
        # try:
        #     gpu_check = subprocess.run(
        #         ["nvidia-smi"],
        #         stdout=subprocess.DEVNULL,
        #         stderr=subprocess.DEVNULL,
        #         check=False,
        #     )
        #     use_srun = gpu_check.returncode != 0
        # except FileNotFoundError:
        #     use_srun = True
        
        # # print(f"Do we use srun? {use_srun}")

        # if use_srun:
        #     cmd = [
        #         "srun",
        #         "--gres=gpu:1",
        #         "--ntasks=1",
        #         "--cpus-per-task=64",
        #         "--time=2-00:00:00",
        #         "bash",
        #         "-lc",
        #         bash_cmd,
        #     ]
        # else:
        #     cmd = ["bash", "-lc", bash_cmd]

        # # Pin each run to a random GPU when none is specified to spread load across 8-GPU node.
        # env = os.environ.copy()
        # if not env.get("CUDA_VISIBLE_DEVICES"):
        #     try:
        #         gpu_count = int(env.get("DEEPRESEARCH_GPU_COUNT", "8"))
        #         env["CUDA_VISIBLE_DEVICES"] = str(random.randrange(gpu_count))
        #     except Exception:
        #         pass

        # async def _stream_output(stream, log_fp, buf: deque, prefix: str = ""):
        #     while True:
        #         line = await stream.readline()
        #         if not line:
        #             break
        #         text = prefix + line.decode(errors="replace")
        #         log_fp.write(text)
        #         log_fp.flush()
        #         # Stream live to user
        #         # print(text, end="", flush=True)
        #         # Skip progress-bar style carriage-return updates in the returned tail to avoid bloating LLM context.
        #         if "\r" in text:
        #             continue
        #         buf.append(text)

        # try:
        #     proc = await asyncio.create_subprocess_exec(
        #         *cmd,
        #         cwd=str(run_dir),
        #         stdout=asyncio.subprocess.PIPE,
        #         stderr=asyncio.subprocess.PIPE,
        #         env=env,
        #         start_new_session=True,
        #     )
        # except FileNotFoundError as e:
        #     missing = "srun" if use_srun else "bash"
        #     return f"[Error] {missing} not found: {e}"
        # except Exception as e:
        #     launcher = "srun" if use_srun else "bash"
        #     return f"[Error] Failed to start {launcher}: {e}"

        # stdout_buf = deque(maxlen=200)
        # stderr_buf = deque(maxlen=200)
        # timed_out = False

        # def _truncate_block(text: str, max_lines: int = 20) -> str:
        #     """
        #     Truncate long text blocks by keeping only the last few lines for the LLM while preserving full logs on disk.
        #     """
        #     if not text:
        #         return text
        #     lines = text.splitlines()
        #     if len(lines) <= max_lines:
        #         return text
        #     tail = lines[-max_lines:]
        #     truncated = len(lines) - max_lines
        #     return "\n".join(tail + [f"...[truncated {truncated} lines, see log for full output]"])

        # with open(log_path, "w", encoding="utf-8") as log_fp:
        #     stdout_task = asyncio.create_task(_stream_output(proc.stdout, log_fp, stdout_buf))
        #     stderr_task = asyncio.create_task(_stream_output(proc.stderr, log_fp, stderr_buf, prefix="[stderr] "))

        #     try:
        #         returncode = await asyncio.wait_for(proc.wait(), timeout=timeout)
        #     except asyncio.TimeoutError:
        #         print("Process timed out, terminating...")
        #         timed_out = True
        #         os.killpg(proc.pid, signal.SIGKILL)
        #         returncode = await proc.wait()
        #     except asyncio.CancelledError:
        #         os.killpg(proc.pid, signal.SIGKILL)
        #         await proc.wait()
        #         raise
        #     finally:
        #         await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

        # stdout_tail = "".join(stdout_buf).strip()
        # stderr_tail = "".join(stderr_buf).strip()
        # stdout_tail = _truncate_block(stdout_tail)

        # # Remove noisy srun launcher lines before truncating
        # if stderr_tail and use_srun:
        #     filtered = [ln for ln in stderr_tail.split("\n") if "srun" not in ln]
        #     stderr_tail = "\n".join(filtered)
        #     stderr_tail = _truncate_block(stderr_tail)

        # if timed_out:
        #     return f"[Timeout] Exceeded {timeout}s. Logs: {log_path}"

        # if returncode != 0:
        #     err_msg = stderr_tail or f"Process exited with code {returncode}"
        #     return f"[Error] {err_msg}\nLogs: {log_path}"

        # submission_path = run_dir / "submission.csv"

        # if stdout_tail:
        #     return f"[Output]\n{stdout_tail}"
        # if stderr_tail:
        #     # if no stdout, check if submission.csv was created
        #     if submission_path.exists():
        #         return f"No stdout, but submission.csv found at {submission_path}. Stderr:\n{stderr_tail}"
        #     return f"No stdout and submission.csv not found. Stderr:\n{stderr_tail}"

        # if submission_path.exists():
        #     return f"No stdout, but submission.csv found at {submission_path}."
        # return f"No stdout and submission.csv not found."


# Tool registry
DEEPRESEARCH_TOOLS = {
    "Search": SearchTool(),
    "Scholar": ScholarTool(),
    "Visit": VisitTool(),
    "FileParser": FileParserTool(),
    "Score": ScoreTool(),
    "SynScore": SynScoreTool(),
    "PythonInterpreter": PythonInterpreterTool(),
}


def get_tool(name: str) -> DeepResearchTool:
    """Get a tool by name."""
    return DEEPRESEARCH_TOOLS.get(name)


def get_all_tools() -> dict[str, DeepResearchTool]:
    """Get all available tools."""
    return DEEPRESEARCH_TOOLS.copy()
