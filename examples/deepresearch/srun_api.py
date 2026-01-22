# srun_api.py
import asyncio
import os
import uuid
import shlex
import signal
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
# Optional partition to force jobs away from inference partition:
DEFAULT_PARTITION = os.environ.get("SRUN_API_PARTITION")  # e.g., "training"

# ---------- App & simple in-memory store ----------
app = FastAPI(title="SRUN Python Executor API")

class JobInfo:
    def __init__(self, job_id: str, workdir: Path, proc: asyncio.subprocess.Process, start_time: datetime):
        self.job_id = job_id
        self.workdir = workdir
        self.proc = proc
        self.start_time = start_time
        self.end_time: Optional[datetime] = None
        self.returncode: Optional[int] = None
        self.timed_out: bool = False

JOB_STORE: Dict[str, JobInfo] = {}
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT)

# ---------- Request schema (script_path required) ----------
class RunRequest(BaseModel):
    script_path: str
    time: Optional[str] = DEFAULT_TIME
    cpus: Optional[int] = DEFAULT_CPUS
    mem: Optional[str] = DEFAULT_MEM
    gres_gpus: Optional[str] = DEFAULT_GRES
    conda_env: Optional[str] = DEFAULT_CONDA_ENV
    pre_cmds: Optional[str] = None
    timeout: Optional[int] = 0  # seconds; 0 => no API-level timeout (wait until job exit)

# ---------- Endpoint: submit job ---------
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

    # create unique job id
    job_uuid = uuid.uuid4().hex[:12]

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
        "bash",
        "-lc",
        inner_cmd,
    ]
    if DEFAULT_PARTITION:
        # insert partition argument after srun
        srun_cmd.insert(1, f"--partition={DEFAULT_PARTITION}")

    # acquire semaphore to bound concurrency in API
    await SEMAPHORE.acquire()
    try:
        # Do not pass CUDA_VISIBLE_DEVICES in env; let Slurm set it for the job step
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        # Start subprocess with cwd = script's parent directory
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

    job = JobInfo(job_uuid, workdir, proc, datetime.utcnow())
    JOB_STORE[job_uuid] = job

    # log file paths next to the script
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stdout_path = workdir / f"{script_name}.{ts}.out"
    stderr_path = workdir / f"{script_name}.{ts}.err"

    # background task streams and waits
    asyncio.create_task(_stream_and_wait(job, stdout_path, stderr_path, req.timeout))
    return {"job_id": job_uuid, "workdir": str(workdir), "stdout": str(stdout_path), "stderr": str(stderr_path)}

# ---------- background streamer & waiter ----------
async def _stream_and_wait(job: JobInfo, stdout_path: Path, stderr_path: Path, timeout: int):
    proc = job.proc
    workdir = job.workdir
    try:
        with stdout_path.open("w", encoding="utf-8") as out_fp, stderr_path.open("w", encoding="utf-8") as err_fp:
            async def _reader(stream, fp):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode(errors="replace")
                    fp.write(text)
                    fp.flush()
            readers = [
                asyncio.create_task(_reader(proc.stdout, out_fp)),
                asyncio.create_task(_reader(proc.stderr, err_fp)),
            ]
            if timeout and timeout > 0:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    job.timed_out = True
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    await proc.wait()
            else:
                await proc.wait()
            # ensure readers done
            await asyncio.gather(*readers, return_exceptions=True)
            job.returncode = proc.returncode
            job.end_time = datetime.utcnow()
    finally:
        SEMAPHORE.release()

# ---------- check status ----------
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
        "start_time": job.start_time.isoformat(),
        "end_time": job.end_time.isoformat() if job.end_time else None,
        "workdir": str(job.workdir),
    }

# ---------- return logs tail ----------
@app.get("/logs/{job_id}")
async def logs(job_id: str, tail_lines: int = 200):
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    # find most recent matching stdout/stderr created by this job in the job.workdir
    stdout = next(iter(sorted(job.workdir.glob("*.out"), key=lambda p: p.stat().st_mtime, reverse=True)), None)
    stderr = next(iter(sorted(job.workdir.glob("*.err"), key=lambda p: p.stat().st_mtime, reverse=True)), None)


    def tail(path: Optional[Path], n=tail_lines):
        if not path or not path.exists():
            return ""
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 1024
            data = b""
            while size > 0 and len(data.splitlines()) <= n:
                read_sz = min(block, size)
                f.seek(size - read_sz)
                data = f.read(read_sz) + data
                size -= read_sz
                if size == 0:
                    break
            return b"\n".join(data.splitlines()[-n:]).decode(errors="replace")
    return {"stdout_tail": tail(stdout), "stderr_tail": tail(stderr)}
