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
    python -m src.publish --repo your-username/credit-card-fraud-rf --split random --no-hour

--split and --no-hour pick which run configuration to upload (default: temporal,
with Hour). The files keep their names, e.g. model_temporal.joblib, so several
configurations can live in the same repository.
"""

import argparse

from huggingface_hub import HfApi

from src.config import ROOT_DIR, add_run_args, metrics_path, model_path, pr_curve_path


def files_to_upload(split: str, use_hour: bool) -> dict:
    """Local file -> path inside the Hugging Face repository, for one run configuration."""
    model = model_path(split, use_hour)
    metrics = metrics_path(split, use_hour)
    curve = pr_curve_path(split, use_hour)
    return {
        ROOT_DIR / "README.md": "README.md",
        ROOT_DIR / "LICENSE": "LICENSE",
        model: model.name,
        metrics: f"reports/{metrics.name}",
        curve: f"reports/{curve.name}",
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
    add_run_args(parser)
    args = parser.parse_args()
    files = files_to_upload(args.split, use_hour=not args.no_hour)

    # Check all files first, so we never upload only part of them.
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files: {missing}. Run `python -m src.train` and "
            "`python -m src.evaluate` first, with the same --split/--no-hour options."
        )

    api = HfApi()
    # Note: `private` only applies when the repository is created. If it already
    # exists, its visibility is not changed.
    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)

    for local_path, path_in_repo in files.items():
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
