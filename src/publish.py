"""Publish the trained model and its reports to the Hugging Face Hub.

Authentication
--------------
Before running this script, log in once with your Hugging Face account:

    hf auth login

It asks for an access token. Create one at https://huggingface.co/settings/tokens
with the "Write" permission and paste it. The token is saved locally, and
huggingface_hub uses it automatically. Check it with `hf auth whoami`.

(Older versions of huggingface_hub used `huggingface-cli login` instead; that
command no longer exists in huggingface_hub 1.0+.)

Usage
-----
    python -m src.publish --repo your-username/credit-card-fraud-rf
    python -m src.publish --repo your-username/credit-card-fraud-rf --private
"""

import argparse

from huggingface_hub import HfApi

from src.config import MODELS_DIR, REPORTS_DIR, ROOT_DIR

# Local file -> path inside the Hugging Face repository.
FILES_TO_UPLOAD = {
    ROOT_DIR / "README.md": "README.md",
    ROOT_DIR / "LICENSE": "LICENSE",
    MODELS_DIR / "model.joblib": "model.joblib",
    REPORTS_DIR / "metrics.json": "reports/metrics.json",
    REPORTS_DIR / "pr_curve.png": "reports/pr_curve.png",
    ROOT_DIR / "requirements.txt": "requirements.txt",
}


def main() -> None:
    """Create the model repository (if needed) and upload the project files."""
    parser = argparse.ArgumentParser(description="Publish the model to the Hugging Face Hub.")
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository id, for example: your-username/credit-card-fraud-rf",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private (default: public).",
    )
    args = parser.parse_args()

    # Check all files first, so we never upload only part of them.
    missing = [str(path) for path in FILES_TO_UPLOAD if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files: {missing}. Run `python -m src.train` and "
            "`python -m src.evaluate` first."
        )

    api = HfApi()
    # Note: `private` only applies when the repository is created. If it already
    # exists, its visibility is not changed.
    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)

    for local_path, path_in_repo in FILES_TO_UPLOAD.items():
        print(f"Uploading {local_path.name} -> {path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=args.repo,
            repo_type="model",
        )

    print(f"Done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
