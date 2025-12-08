#!/usr/bin/env python3
# Copyright ...
import fnmatch
import json
import logging
import os
import posixpath
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional
from urllib.parse import quote

import draccus
import fsspec
from huggingface_hub import HfApi, hf_hub_download
from tqdm_loggable.auto import tqdm

logger = logging.getLogger("download_hf_manual")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_provenance_json(output_path: str, metadata: dict[str, Any]) -> None:
    metadata = dict(metadata)
    metadata["access_time"] = _now_iso()
    with fsspec.open(posixpath.join(output_path, "provenance.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    logger.info("[*] Wrote provenance.json")


def ensure_fsspec_path_writable(output_path: str) -> None:
    fs, _ = fsspec.core.url_to_fs(output_path)
    try:
        fs.mkdirs(output_path, exist_ok=True)
        test_path = posixpath.join(output_path, ".write_test")
        with fs.open(test_path, "w") as f:
            f.write("ok")
        fs.rm(test_path)
    except Exception as e:
        raise ValueError(f"No write access to {output_path}: {e}") from e


def construct_hf_url(dataset_id: str, revision: str, file_path: str) -> str:
    encoded = quote(file_path)
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{encoded}"


def _match_any(path: str, patterns: Iterable[str]) -> bool:
    path = path.lstrip("/")  # repo paths are POSIX-like
    for p in patterns:
        if fnmatch.fnmatchcase(path, p):
            return True
    return False


@dataclass(frozen=True)
class DownloadConfig:
    # Required
    hf_dataset_id: str  # e.g. "EleutherAI/proof-pile-2"
    revision: str  # e.g. "main" or a commit hash
    gcs_output_path: str  # any fsspec path (gs://, s3://, file://, etc.)

    # Optional selection
    include: List[str] = field(default_factory=list)  # globs to include; empty => include all
    exclude: List[str] = field(default_factory=list)  # globs to exclude (applied after include)
    max_files: Optional[int] = None  # hard cap on number of files to transfer

    # Execution
    max_workers: int = 16
    # If True, try the hf-transfer downloader (requires `pip install hf-transfer`)
    use_hf_transfer: bool = True


def _select_files(
    all_files: List[str],
    include: List[str],
    exclude: List[str],
    max_files: Optional[int],
) -> List[str]:
    if not include:
        selected = list(all_files)  # include all by default
    else:
        selected = [f for f in all_files if _match_any(f, include)]

    if exclude:
        selected = [f for f in selected if not _match_any(f, exclude)]

    selected.sort()
    if max_files is not None and max_files >= 0:
        selected = selected[:max_files]
    return selected


def _fs_put(fs: Any, src_path: str, dst_path: str) -> None:
    # Ensure parent dirs exist
    parent = posixpath.dirname(dst_path)
    if parent:
        fs.mkdirs(parent, exist_ok=True)
    fs.put(src_path, dst_path)


def _require_token() -> str:
    tok = os.environ.get("HF_TOKEN") or ""
    if not tok:
        raise RuntimeError("HF_TOKEN is not set in the environment; set it to your HF access token.")
    return tok


def _maybe_enable_hf_transfer(enable: bool) -> None:
    if enable:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def download_and_upload_to_store(cfg: DownloadConfig) -> None:
    _maybe_enable_hf_transfer(cfg.use_hf_transfer)
    token = _require_token()

    # Prepare output FS
    ensure_fsspec_path_writable(cfg.gcs_output_path)
    versioned_out = posixpath.join(cfg.gcs_output_path, cfg.revision)
    ensure_fsspec_path_writable(versioned_out)
    fs, _ = fsspec.core.url_to_fs(versioned_out)

    # List repo files
    api = HfApi(token=token)
    logger.info(f"Listing files for {cfg.hf_dataset_id}@{cfg.revision} (repo_type=dataset)")
    all_files = api.list_repo_files(repo_id=cfg.hf_dataset_id, revision=cfg.revision, repo_type="dataset")
    logger.info(f"Total files in repo: {len(all_files)}")

    # Select subset
    selected = _select_files(all_files, cfg.include, cfg.exclude, cfg.max_files)
    if not selected:
        logger.warning("No files matched your selection. " 'Tip: use --include "**" (quotes are important in shells).')
        return

    logger.info(f"Selected {len(selected)} files to transfer.")
    logger.debug("First 10 selected files:\n  " + "\n  ".join(selected[:10]))

    # Prepare provenance links
    links = [construct_hf_url(cfg.hf_dataset_id, cfg.revision, f) for f in selected]

    # Transfer
    pbar = tqdm(total=len(selected), desc="Transferring")
    futures = []
    errors = 0

    def worker(temp_dir: str, rel_path: str) -> str:
        # rel_path is the repo-relative filename expected by hf_hub_download
        local = hf_hub_download(
            repo_id=cfg.hf_dataset_id,
            filename=rel_path,
            revision=cfg.revision,
            token=token,
            local_dir=temp_dir,
            local_dir_use_symlinks=False,
            repo_type="dataset",
        )
        dst = posixpath.join(versioned_out, rel_path)
        _fs_put(fs, local, dst)
        return rel_path

    with tempfile.TemporaryDirectory() as tmp:
        with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as pool:
            for rel in selected:
                futures.append(pool.submit(worker, tmp, rel))

            for fut in as_completed(futures):
                try:
                    rel_path = fut.result()
                    logger.debug(f"Uploaded {rel_path}")
                except Exception as e:
                    errors += 1
                    logger.exception(f"Transfer failed: {e}")
                finally:
                    pbar.update(1)

    pbar.close()

    # Provenance
    write_provenance_json(
        versioned_out,
        metadata={
            "dataset": cfg.hf_dataset_id,
            "revision": cfg.revision,
            "files": selected,
            "links": links,
            "errors": errors,
        },
    )

    if errors:
        logger.warning(f"Completed with {errors} failures. See logs for details.")
    else:
        logger.info("Completed successfully.")


@draccus.wrap()
def main(cfg: DownloadConfig) -> None:
    download_and_upload_to_store(cfg)


if __name__ == "__main__":
    main()
