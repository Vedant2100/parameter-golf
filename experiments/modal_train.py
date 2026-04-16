import os
import modal

# -----------------------------
# CONFIGURATION
# -----------------------------
# Name of the Modal app
APP_NAME = "parameter-golf"
# Path where we will mount our persistent storage inside the container
VOLUME_PATH = "/data"
# Choice of GPU for the training speedrun
# L40S is cost-effective; switch to H100:8 for final leaderboard runs.
TRAIN_GPU = "L40S:8"

# 1. Define the persistence volume for datasets and tokenizers
volume = modal.Volume.from_name("pg-data-vol", create_if_missing=True)

# 2. Define the container environment
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "curl")
    .pip_install(
        "numpy",
        "tqdm",
        "torch",
        "huggingface-hub",
        "setuptools",
        "typing-extensions==4.15.0",
        "datasets",
        "tiktoken",
        "sentencepiece",
    )
    .add_local_dir(".", remote_path="/workspace", ignore=["data/datasets", "data/tokenizers", ".venv", ".git"])
)

app = modal.App(APP_NAME, image=image)

# -----------------------------
# TASKS
# -----------------------------

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=3600, # 1 hour
)
def download_data(shards: int = 20):
    """
    Downloads the FineWeb dataset shards to the persistent volume.
    Run this once before your first training job.
    """
    import subprocess
    
    print(f"🚀 Starting download of {shards} shards to {VOLUME_PATH}...")
    
    cmd = [
        "python3", "data/cached_challenge_fineweb.py",
        "--variant", "sp1024",
        "--train-shards", str(shards)
    ]
    
    # We override the default data paths so they save directly into the Volume
    env = os.environ.copy()
    env["DATASETS_PATH"] = os.path.join(VOLUME_PATH, "datasets")
    env["TOKENIZERS_PATH"] = os.path.join(VOLUME_PATH, "tokenizers")
    
    subprocess.run(cmd, check=True, cwd="/workspace", env=env)
    
    print("✅ Download complete. Committing volume...")
    volume.commit()


@app.function(
    gpu=TRAIN_GPU,
    volumes={VOLUME_PATH: volume},
    timeout=1200, # 20 mins (to allow for setup + 10 min train)
)
def train(iterations: int = 20000, run_id: str = None):
    """
    Executes the multi-GPU speedrun on the Modal cluster.
    """
    import subprocess
    import torch
    
    # 1. Sync the volume to get the latest data
    volume.reload()
    
    num_gpus = torch.cuda.device_count()
    print(f"🔥 Starting training on {num_gpus} {TRAIN_GPU.split(':')[0]} GPUs...")

    # 2. Construct the torchrun command
    # We use standalone mode to manage the distributed coordination automatically
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        "train_gpt.py"
    ]
    
    # 3. Setup environment variables for the training script
    env = os.environ.copy()
    env["DATA_PATH"] = os.path.join(VOLUME_PATH, f"datasets/fineweb10B_sp1024")
    env["TOKENIZER_PATH"] = os.path.join(VOLUME_PATH, "tokenizers/fineweb_1024_bpe.model")
    env["ITERATIONS"] = str(iterations)
    if run_id:
        env["RUN_ID"] = run_id
    
    # Set PyTorch optimization flags
    env["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    
    # 4. Execute
    # We don't capture output here so that it streams live to your terminal
    subprocess.run(cmd, check=True, cwd="/workspace", env=env)
    
    print("✅ Training complete. Results saved to logs/ and committed to Volume.")
    volume.commit()

@app.local_entrypoint()
def main(download: bool = False, shards: int = 20, train_run: bool = False):
    if download:
        download_data.remote(shards=shards)
    if train_run:
        train.remote()
