"""Load model artifacts for inference: local dir, or fetch from MODEL_ARTIFACT_URI (S3, local file, tar.gz)."""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import joblib

MODEL_BASENAMES = ("sample_model.joblib", "regression_model.joblib")
MODEL_BASENAME = MODEL_BASENAMES[0]
VERSION_BASENAME = "model_version.txt"


def _is_tarball(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name.lower()
    return name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar")


def _expected_model_names() -> str:
    return " or ".join(MODEL_BASENAMES)


def _find_model_path(search_root: Path) -> Path:
    for basename in MODEL_BASENAMES:
        direct = search_root / basename
        if direct.is_file():
            return direct
    for basename in MODEL_BASENAMES:
        for p in search_root.rglob(basename):
            if p.is_file():
                return p
    raise FileNotFoundError(
        f"{_expected_model_names()} not found under {search_root} (after extract or download)"
    )


def _find_model_root(search_root: Path) -> Path:
    return _find_model_path(search_root).parent


def _extract_tarball(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        if sys.version_info >= (3, 12):
            tf.extractall(dest, filter="data")
        else:
            tf.extractall(dest)
    return _find_model_root(dest)


def _download_s3_to_dir(uri: str, dest_dir: Path) -> Path:
    import boto3

    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid s3 URI: {uri!r}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 key missing in URI: {uri!r}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = key.rsplit("/", 1)[-1] or "artifact"
    local_path = dest_dir / filename

    client = boto3.client("s3")
    client.download_file(bucket, key, str(local_path))
    return local_path


def _materialize_uri_to_model_root(uri: str, model_dir: Path) -> Path:
    """Download or unpack artifacts into model_dir and return directory containing the joblib."""
    model_dir.mkdir(parents=True, exist_ok=True)

    if uri.startswith("s3://"):
        local_file = _download_s3_to_dir(uri, model_dir)
        if _is_tarball(local_file):
            extract_root = model_dir / "_extracted"
            if extract_root.exists():
                shutil.rmtree(extract_root)
            result = _extract_tarball(local_file, extract_root)
            local_file.unlink(missing_ok=True)
            return result
        if local_file.name in MODEL_BASENAMES:
            return local_file.parent
        raise FileNotFoundError(
            f"After S3 download, expected {_expected_model_names()} or a tarball, "
            f"got {local_file.name}"
        )

    raw = uri.replace("file://", "").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.is_dir():
        return _find_model_root(path)

    if not path.is_file():
        raise FileNotFoundError(f"MODEL_ARTIFACT_URI not found: {path}")

    if _is_tarball(path):
        extract_root = model_dir / "_extracted"
        if extract_root.exists():
            shutil.rmtree(extract_root)
        return _extract_tarball(path, extract_root)

    if path.name == MODEL_BASENAME or path.suffix == ".joblib":
        return path.parent

    raise ValueError(f"Unsupported artifact file type: {path}")


def resolve_model_directory() -> Path:
    """Directory containing sample_model.joblib (after optional fetch/extract)."""
    uri = (os.environ.get("MODEL_ARTIFACT_URI") or "").strip()
    base = Path(os.environ.get("MODEL_DIR", "runs/artifacts/latest"))

    if not uri:
        return base

    return _materialize_uri_to_model_root(uri, base)


def load_model(artifacts_dir: str):
    base = Path(artifacts_dir)
    model_path = _find_model_path(base)

    model = joblib.load(model_path)
    version_path = base / VERSION_BASENAME

    override = (os.environ.get("MODEL_VERSION") or "").strip()
    if override:
        model_version = override
    elif version_path.is_file():
        model_version = version_path.read_text(encoding="utf-8").strip()
    else:
        raise FileNotFoundError(
            f"{VERSION_BASENAME} not found under {base} and MODEL_VERSION is not set"
        )
    return model, model_version


def load_model_bundle():
    """Resolve artifacts (optional MODEL_ARTIFACT_URI) and load model + version string."""
    resolved = resolve_model_directory()
    return load_model(str(resolved))
