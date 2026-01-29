import time
import requests

API = "http://10.136.73.30:9000"

SCRIPT_PATH = (
    "/fsx/zyhang/rllm/examples/deepresearch/output/train-20260121-171606/"
    "botanicavision-ultra-fine-grained-plant-species-recognition-from-field-photography/"
    "20260121-172057-847495/main_20260121-172125-397564.py"
)

# -------- submit job --------
print("Submitting job...")
resp = requests.post(
    f"{API}/run",
    json={
        "script_path": SCRIPT_PATH,
        "time": "00:10:00",
        "cpus": 8,
        "mem": "32G",
        "gres_gpus": "gpu:1",
        "conda_env": "algoevolve",
    },
    timeout=30,
)
resp.raise_for_status()
job = resp.json()

job_id = job["job_id"]
stdout_path = job["stdout"]
stderr_path = job["stderr"]

print(f"Job submitted: {job_id}")
print(f"stdout: {stdout_path}")
print(f"stderr: {stderr_path}")
print("-" * 60)

# -------- poll until finished --------
while True:
    s = requests.get(f"{API}/status/{job_id}", timeout=30)
    s.raise_for_status()
    status = s.json()

    state = status.get("state")
    print(f"[status] {state}")

    # fetch logs tail every poll
    logs = requests.get(
        f"{API}/logs/{job_id}",
        params={"tail_lines": 40},
        timeout=30,
    ).json()

    stdout_tail = logs.get("stdout_tail", "")
    stderr_tail = logs.get("stderr_tail", "")

    if stdout_tail:
        print("\n--- stdout (tail) ---")
        print(stdout_tail)

    if stderr_tail:
        print("\n--- stderr (tail) ---")
        print(stderr_tail)

    if state not in ("RUNNING", "PENDING"):
        print("\nJob finished.")
        break

    print("-" * 60)
    time.sleep(5)

print("\nFinal log locations:")
print("stdout:", stdout_path)
print("stderr:", stderr_path)
