import shutil
import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv

from ml_training.utils import BASE_DIR


def check_env() -> str:
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        print("Running on Kaggle")
        return "kaggle"
    else:
        print("Running locally")
        return "local"


def download_from_hf(repo_id: str, filename: str, local_dir: Path, token: str = None, force: bool = False) -> Path:
    """Download a file from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        local_dir=local_dir,
        force_download=force
    )
    return Path(path)


def needs_download(hf_repo: str, hf_token: str, local_version_path: Path) -> tuple[bool, str]:
    """Compare HF version with local version. Returns (needs_download, hf_version)."""
    import tempfile

    # Download HF version to temp dir to avoid overwriting local
    with tempfile.TemporaryDirectory() as tmp_dir:
        hf_version_path = download_from_hf(hf_repo, "version.txt", Path(tmp_dir), hf_token, force=True)
        with open(hf_version_path, "r") as f:
            hf_version = f.read().strip()

    if not local_version_path.exists():
        return True, hf_version

    with open(local_version_path, "r") as f:
        local_version = f.read().strip()

    return hf_version > local_version, hf_version


def prepare_dataset() -> Path:
    """Download and extract dataset, or use local version if it's up to date."""
    environment = check_env()

    if environment == "kaggle":
        return Path("/kaggle/input/ka-ocr")

    load_dotenv(BASE_DIR / ".env")

    data_dir = BASE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    hf_repo = os.getenv("HF_DATASET_REPO")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_repo:
        raise ValueError("HF_DATASET_REPO not set in .env")

    local_version_path = data_dir / "version.txt"

    print("Downloading version file from HuggingFace for comparison...")
    should_download, hf_version = needs_download(hf_repo, hf_token, local_version_path)

    if should_download:
        print(f"dataset needs updating (HF version: {hf_version})")

        # Clear old data first
        print("Clearing old dataset...")
        for item in data_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        # Download zip and extract
        print("downloading...")
        zip_path = download_from_hf(hf_repo, "ka-ocr.zip", data_dir, hf_token, force=True)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Delete zip
        zip_path.unlink()
        print("Extraction complete, zip deleted")

        # Save the new version file locally
        with open(local_version_path, "w") as f:
            f.write(hf_version)

        return data_dir

    print(f"Local dataset is the newest version already: {hf_version}. No update needed.")
    return data_dir
