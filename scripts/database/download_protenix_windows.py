#!/usr/bin/env python3

"""Download the Protenix inference assets needed on the workstation.

This script is intended to be run on a Windows laptop with Python 3.10+.
It downloads:

- All published Protenix inference checkpoints needed to support every model
- Shared ESM sidecar checkpoints required by the ESM/ISM variants
- The always-used common inference files
- Template metadata for both supported data versions
- The seqres FASTA used by local template search

It intentionally does NOT download the large local MSA/RNA-MSA/template-search
databases, because the target workstation will use external MSA/template
services and can fetch template mmCIF files from PDBe at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_ROOT = Path("protenix_offline_assets")
CHUNK_SIZE = 8 * 1024 * 1024

DEFAULT_BASE_URL = "https://protenix.tos-cn-beijing.volces.com"
V20260101_BASE_URL = "https://protenix.tos-cn-beijing.volces.com/protenix.20260101"

CHECKPOINT_FILES = [
    "protenix-v2.pt",
    "protenix_base_default_v1.0.0.pt",
    "protenix_base_20250630_v1.0.0.pt",
    "protenix_base_default_v0.5.0.pt",
    "protenix_base_constraint_v0.5.0.pt",
    "protenix_mini_default_v0.5.0.pt",
    "protenix_tiny_default_v0.5.0.pt",
    "protenix_mini_esm_v0.5.0.pt",
    "protenix_mini_ism_v0.5.0.pt",
    "esm2_t36_3B_UR50D.pt",
    "esm2_t36_3B_UR50D-contact-regression.pt",
    "esm2_t36_3B_UR50D_ism.pt",
    "esm2_t36_3B_UR50D_ism-contact-regression.pt",
]

COMMON_ALWAYS_USED = [
    "components.cif",
    "components.cif.rdkit_mol.pkl",
    "clusters-by-entity-40.txt",
    "obsolete_release_date.csv",
]

TEMPLATE_METADATA = [
    "release_date_cache.json",
    "obsolete_to_successor.json",
]

TEMPLATE_SEARCH_DATABASE_FILES = [
    "pdb_seqres_2022_09_28.fasta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all offline Protenix inference assets for workstation upload."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root output directory on the local machine.",
    )
    parser.add_argument(
        "--skip-archives",
        action="store_true",
        help=(
            "Skip downloading the versioned common.tar.gz archives. "
            "The script will still download the individual common files it knows about."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    return parser.parse_args()


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, force: bool = False) -> None:
    ensure_dir(destination.parent)

    if destination.exists() and not force:
        print(f"[skip] {destination}")
        return

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "protenix-offline-downloader/1.0"})

    print(f"[get ] {url}")
    try:
        with urlopen(request, timeout=120) as response, tmp_path.open("wb") as out_handle:
            shutil.copyfileobj(response, out_handle, length=CHUNK_SIZE)
    except (HTTPError, URLError, TimeoutError) as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    tmp_path.replace(destination)
    print(f"[ ok ] {destination} sha256={sha256_of_file(destination)}")


def extract_selected_members(archive_path: Path, output_dir: Path, wanted_names: Iterable[str]) -> None:
    wanted = set(wanted_names)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        matched = []
        for member in members:
            member_name = Path(member.name).name
            if member.isfile() and member_name in wanted:
                matched.append((member, member_name))

        found_names = {name for _, name in matched}
        missing = sorted(wanted - found_names)
        if missing:
            print(f"[warn] {archive_path.name} did not contain: {', '.join(missing)}")

        for member, member_name in matched:
            target = output_dir / member_name
            ensure_dir(target.parent)
            with archive.extractfile(member) as src, target.open("wb") as dst:
                if src is None:
                    raise RuntimeError(f"Could not extract {member.name} from {archive_path}")
                shutil.copyfileobj(src, dst, length=CHUNK_SIZE)
            print(f"[ ok ] extracted {target} sha256={sha256_of_file(target)}")


def download_archived_common_version(
    version_name: str,
    base_url: str,
    output_root: Path,
    force: bool,
) -> None:
    version_dir = output_root / version_name
    common_dir = version_dir / "common"
    ensure_dir(common_dir)

    archive_name = "common.tar.gz"
    archive_path = version_dir / archive_name
    download_file(f"{base_url}/{archive_name}", archive_path, force=force)
    extract_selected_members(
        archive_path,
        common_dir,
        COMMON_ALWAYS_USED + TEMPLATE_METADATA,
    )


def download_individual_common_files(version_name: str, base_url: str, output_root: Path, force: bool) -> None:
    common_dir = output_root / version_name / "common"
    ensure_dir(common_dir)
    for filename in COMMON_ALWAYS_USED + TEMPLATE_METADATA:
        download_file(f"{base_url}/common/{filename}", common_dir / filename, force=force)


def download_checkpoints(output_root: Path, force: bool) -> None:
    checkpoint_dir = output_root / "shared" / "checkpoint"
    ensure_dir(checkpoint_dir)
    for filename in CHECKPOINT_FILES:
        download_file(
            f"{DEFAULT_BASE_URL}/checkpoint/{filename}",
            checkpoint_dir / filename,
            force=force,
        )


def download_template_search_database(output_root: Path, force: bool) -> None:
    search_database_dir = output_root / "shared" / "search_database"
    ensure_dir(search_database_dir)
    for filename in TEMPLATE_SEARCH_DATABASE_FILES:
        download_file(
            f"{DEFAULT_BASE_URL}/search_database/{filename}",
            search_database_dir / filename,
            force=force,
        )


def write_manifest(output_root: Path) -> None:
    manifest = output_root / "README.txt"
    manifest.write_text(
        """
This directory was generated by download_protenix_windows.py.

Layout:
- shared/checkpoint/
    All Protenix model checkpoints plus ESM sidecar checkpoints.
- shared/search_database/
    Shared seqres FASTA used by local template search.
- original/common/
    Common inference files and template metadata from the original data release.
- v20260101/common/
    Common inference files and template metadata from the 20260101 data release.

Recommended workstation mapping:
- Copy shared/checkpoint/* into PROTENIX_ROOT_DIR/checkpoint/
- Copy shared/search_database/* into PROTENIX_ROOT_DIR/search_database/
- For the original data release, copy original/common/* into PROTENIX_ROOT_DIR/common/
- For the 20260101 data release, keep a separate copy of v20260101/common/
  and switch them into PROTENIX_ROOT_DIR/common/ when you want that dataset version.

Because the workstation can access PDBe at runtime, this bundle does not include:
- mmcif/
- mmcif_msa_template/
- rna_msa/
- the rest of search_database/
""".strip()
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_root = args.out_dir.resolve()
    ensure_dir(output_root)

    print(f"Output root: {output_root}")

    download_checkpoints(output_root, force=args.force)
    download_template_search_database(output_root, force=args.force)

    version_specs = [
        ("original", DEFAULT_BASE_URL),
        ("v20260101", V20260101_BASE_URL),
    ]

    for version_name, base_url in version_specs:
        if args.skip_archives:
            download_individual_common_files(version_name, base_url, output_root, args.force)
        else:
            try:
                download_archived_common_version(version_name, base_url, output_root, args.force)
            except Exception as exc:
                print(f"[warn] archive download failed for {version_name}: {exc}")
                print(f"[warn] falling back to individual file downloads for {version_name}")
                download_individual_common_files(version_name, base_url, output_root, args.force)

    write_manifest(output_root)
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
