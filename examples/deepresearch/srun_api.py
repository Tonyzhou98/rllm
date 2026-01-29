import asyncio
import os
import uuid
import shlex
import signal
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# ---------- Configuration ----------
CONDA_BASE = os.environ.get("CONDA_BASE", str(Path.home() / "miniconda3"))
DEFAULT_CONDA_ENV = os.environ.get("DEEPRESEARCH_CONDA_ENV", "algoevolve")

MAX_CONCURRENT = int(os.environ.get("SRUN_API_MAX_CONCURRENT", "32"))
DEFAULT_CPUS = int(os.environ.get("SRUN_API_CPUS", "8"))
DEFAULT_TIME = os.environ.get("SRUN_API_TIME", "02:00:00")
DEFAULT_MEM = os.environ.get("SRUN_API_MEM", "32G")
DEFAULT_GRES = os.environ.get("SRUN_API_GRES", "gpu:1")

# Regex to capture "job 60061" from srun stderr
JOB_ID_RE = re.compile(r"\bjob\s+(\d+)\b")

app = FastAPI(title="SRUN Python Executor API")

# ---------- Job store ----------
class JobInfo:
    def __init__(
        self,
        job_id: str,
        workdir: Path,
        proc: asyncio.subprocess.Process,
        start_time: datetime,
        stdout_path: Path,
        stderr_path: Path,
    ):
        self.job_id = job_id
        self.workdir = workdir
        self.proc = proc
        self.start_time = start_time
        self.end_time: Optional[datetime] = None
        self.returncode: Optional[int] = None
        self.timed_out: bool = False

        # Exact log paths for this job
        self.stdout_path: Path = stdout_path
        self.stderr_path: Path = stderr_path

        # Slurm job id parsed from srun stderr (for reliable scancel)
        self.slurm_job_id: Optional[str] = None


JOB_STORE: Dict[str, JobInfo] = {}
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT)

# ---------- Request schema ----------
class RunRequest(BaseModel):
    script_path: str
    time: Optional[str] = DEFAULT_TIME
    cpus: Optional[int] = DEFAULT_CPUS
    mem: Optional[str] = DEFAULT_MEM
    gres_gpus: Optional[str] = DEFAULT_GRES
    conda_env: Optional[str] = DEFAULT_CONDA_ENV
    pre_cmds: Optional[str] = None
    job_name: Optional[str] = "deepresearch_api_job"
    # API-level timeout (0 => no API timeout; caller can cancel via /cancel)
    timeout: Optional[int] = 0


# ---------- Internal: cancel helper ----------
def _scancel(slurm_job_id: str) -> bool:
    """
    Cancel a Slurm job by ID. Returns True if scancel was invoked successfully.
    """
    try:
        proc = subprocess.run(
            ["scancel", str(slurm_job_id)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        # scancel not in PATH
        return False
    except Exception:
        return False


def _kill_process_group(pid: Optional[int]) -> None:
    """
    Kill the local srun process group as a fallback.
    """
    if not pid:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        pass


# ---------- Endpoint: submit ----------
@app.post("/run")
async def run(req: RunRequest = Body(..., example={"script_path": "/path/to/main.py"})):
    # Validate script_path and compute workdir
    script_path = Path(req.script_path).expanduser().resolve()
    if not script_path.exists():
        raise HTTPException(status_code=400, detail=f"script not found: {script_path}")
    if not script_path.is_file():
        raise HTTPException(status_code=400, detail=f"script_path is not a file: {script_path}")

    workdir = script_path.parent
    script_name = script_path.name

    # build inner command that runs inside the srun step (in the script directory)
    conda_activate = f"source ~/miniconda3/bin/activate && conda activate algoevolve"
    conda_deactivate = "conda deactivate || true"

    inner_cmd_parts = []
    if req.pre_cmds:
        inner_cmd_parts.append(req.pre_cmds)
    inner_cmd_parts.append(conda_activate)
    inner_cmd_parts.append(f"python -u {shlex.quote(script_name)}")
    inner_cmd_parts.append(conda_deactivate)
    inner_cmd = " && ".join(inner_cmd_parts)

    # build srun command
    srun_cmd = [
        "srun",
        "--gres=gpu:1",
        "--ntasks=1",
        "--cpus-per-task=64",
        "--mem=32G",
        "--time=2-00:00:00",
        f"--job-name={req.job_name or 'deepresearch_api_job'}",
        "bash",
        "-lc",
        inner_cmd,
    ]

    # Create API-level job id
    job_uuid = uuid.uuid4().hex[:12]

    # Log files next to script (same folder as your run artifacts)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stdout_path = workdir / f"{script_name}.{ts}.out"
    stderr_path = workdir / f"{script_name}.{ts}.err"

    # Limit concurrent active srun processes from the API
    await SEMAPHORE.acquire()
    try:
        # Ensure Slurm controls CUDA_VISIBLE_DEVICES for the step
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)

        proc = await asyncio.create_subprocess_exec(
            *srun_cmd,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
    except FileNotFoundError as e:
        SEMAPHORE.release()
        raise HTTPException(status_code=500, detail=f"[Error] srun not found: {e}")
    except Exception as e:
        SEMAPHORE.release()
        raise HTTPException(status_code=500, detail=f"[Error] Failed to start srun: {e}")

    job = JobInfo(job_uuid, workdir, proc, datetime.utcnow(), stdout_path, stderr_path)
    JOB_STORE[job_uuid] = job

    # Stream output and wait in background
    asyncio.create_task(_stream_and_wait(job, req.timeout or 0))

    return {
        "job_id": job_uuid,
        "workdir": str(workdir),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


# ---------- Background: stream and wait ----------
async def _stream_and_wait(job: JobInfo, api_timeout: int):
    """
    Streams stdout/stderr to the job's log files, parses slurm job id from stderr,
    waits for completion, and cancels via scancel if api_timeout triggers.
    Always releases SEMAPHORE when done.
    """
    proc = job.proc
    try:
        with job.stdout_path.open("w", encoding="utf-8") as out_fp, job.stderr_path.open("w", encoding="utf-8") as err_fp:

            async def _read_stdout():
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    text = line.decode(errors="replace")
                    out_fp.write(text)
                    out_fp.flush()

            async def _read_stderr():
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break
                    text = line.decode(errors="replace")
                    err_fp.write(text)
                    err_fp.flush()

                    # Parse slurm job id if present
                    if job.slurm_job_id is None:
                        m = JOB_ID_RE.search(text)
                        if m:
                            job.slurm_job_id = m.group(1)

            readers = [
                asyncio.create_task(_read_stdout()),
                asyncio.create_task(_read_stderr()),
            ]

            if api_timeout and api_timeout > 0:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=api_timeout)
                except asyncio.TimeoutError:
                    job.timed_out = True

                    # Prefer scancel if we captured the slurm job id
                    cancelled = False
                    if job.slurm_job_id:
                        cancelled = _scancel(job.slurm_job_id)

                    # Fallback: kill local srun process group
                    if not cancelled:
                        _kill_process_group(proc.pid)

                    # Ensure process exits
                    try:
                        await proc.wait()
                    except Exception:
                        pass
            else:
                await proc.wait()

            # Ensure readers drain
            await asyncio.gather(*readers, return_exceptions=True)

            job.returncode = proc.returncode
            job.end_time = datetime.utcnow()
    finally:
        SEMAPHORE.release()


# ---------- Status ----------
@app.get("/status/{job_id}")
async def status(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    proc = job.proc
    if proc.returncode is None:
        state = "RUNNING" if proc.pid else "PENDING"
    else:
        state = "COMPLETED"

    return {
        "job_id": job_id,
        "state": state,
        "pid": proc.pid,
        "returncode": job.returncode,
        "timed_out": job.timed_out,
        "slurm_job_id": job.slurm_job_id,
        "start_time": job.start_time.isoformat(),
        "end_time": job.end_time.isoformat() if job.end_time else None,
        "workdir": str(job.workdir),
        "stdout": str(job.stdout_path),
        "stderr": str(job.stderr_path),
    }


# ---------- Logs ----------
@app.get("/logs/{job_id}")
async def logs(job_id: str, tail_lines: int = 200):
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    def tail(path: Path, n=tail_lines) -> str:
        if not path.exists():
            return ""
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and len(data.splitlines()) <= n:
                read_sz = min(block, size)
                f.seek(size - read_sz)
                data = f.read(read_sz) + data
                size -= read_sz
                if size == 0:
                    break
            lines = data.splitlines()[-n:]
            return b"\n".join(lines).decode(errors="replace")

    return {
        "stdout_tail": tail(job.stdout_path),
        "stderr_tail": tail(job.stderr_path),
        "stdout_path": str(job.stdout_path),
        "stderr_path": str(job.stderr_path),
    }


# ---------- Cancel ----------
@app.post("/cancel/{job_id}")
async def cancel(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    proc = job.proc
    if proc.returncode is not None:
        return {
            "job_id": job_id,
            "cancelled": False,
            "reason": "already finished",
            "slurm_job_id": job.slurm_job_id,
        }

    # Prefer scancel if we have the slurm id
    cancelled = False
    if job.slurm_job_id:
        cancelled = _scancel(job.slurm_job_id)

    # Fallback: kill local srun group if no slurm id or scancel failed
    if not cancelled:
        _kill_process_group(proc.pid)

    # Best-effort wait for local proc to exit
    try:
        await proc.wait()
    except Exception:
        pass

    return {
        "job_id": job_id,
        "cancelled": True,
        "slurm_job_id": job.slurm_job_id,
        "used_scancel": bool(job.slurm_job_id),
    }
