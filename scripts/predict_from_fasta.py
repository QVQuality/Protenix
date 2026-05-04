#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse, urlunparse


DEFAULT_MODEL_NAME = "protenix_base_20250630_v1.0.0"
DEFAULT_MSA_SERVER_URL = "http://localhost:8080/msaserver"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Protenix JSON from a multi-sequence FASTA file and run "
            "prediction in the Protenix Docker Compose service using a local "
            "ColabFold-style MSA server plus Protenix template search."
        )
    )
    parser.add_argument(
        "fasta",
        type=Path,
        help="Path to an input FASTA file containing one or more protein sequences.",
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=None,
        help=(
            "Directory for generated JSON, intermediate files, and prediction "
            "outputs. Defaults to outputs/tools/protenix/<fasta_stem>."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Job name to store in the generated Protenix JSON.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Protenix model checkpoint name.",
    )
    parser.add_argument(
        "--msa-server-url",
        type=str,
        default=DEFAULT_MSA_SERVER_URL,
        help="ColabFold-style MSA server base URL.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="101",
        help="Comma-separated model seeds passed to Protenix.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of samples passed to Protenix.",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=10,
        help="Pairformer cycle count passed to Protenix.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=200,
        help="Diffusion step count passed to Protenix.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="Inference dtype passed to Protenix.",
    )
    parser.add_argument(
        "--compose-service",
        type=str,
        default="protenix",
        help="Docker Compose service name to run.",
    )
    parser.add_argument(
        "--compose-project-dir",
        type=Path,
        default=None,
        help="Directory containing docker-compose.yaml. Defaults to the repo root.",
    )
    parser.add_argument(
        "--use-default-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --use_default_params to Protenix.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run docker compose build for the Protenix service before prediction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated command and exit without running it.",
    )
    return parser.parse_args()


def detect_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "docker-compose.yaml").exists():
            return candidate
    raise RuntimeError(
        "Could not find the repository root from the script location. "
        "Pass --compose-project-dir explicitly."
    )


def load_fasta_records(fasta_path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_header: str | None = None
    current_sequence: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequence = "".join(current_sequence).strip().upper()
                    if sequence:
                        records.append((current_header, sequence))
                current_header = line[1:].strip()
                current_sequence = []
                continue
            current_sequence.append(line)

    if current_header is not None:
        sequence = "".join(current_sequence).strip().upper()
        if sequence:
            records.append((current_header, sequence))

    if not records:
        raise ValueError(f"No FASTA sequences found in {fasta_path}")

    return records


def excel_style_chain_id(index: int) -> str:
    label = ""
    current = index
    while True:
        current, remainder = divmod(current, 26)
        label = chr(ord("A") + remainder) + label
        if current == 0:
            return label
        current -= 1


def extract_chain_id(header: str) -> str | None:
    match = re.search(r"(?:^|\|)chain=([^|\s]+)", header)
    if not match:
        return None
    chain_id = match.group(1).strip()
    return chain_id or None


def choose_chain_ids(headers: Iterable[str]) -> list[str]:
    chain_ids: list[str] = []
    seen: set[str] = set()
    for index, header in enumerate(headers):
        chain_id = extract_chain_id(header)
        if chain_id is None or chain_id in seen:
            chain_id = excel_style_chain_id(index)
            while chain_id in seen:
                index += 1
                chain_id = excel_style_chain_id(index)
        chain_ids.append(chain_id)
        seen.add(chain_id)
    return chain_ids


def build_input_payload(job_name: str, records: list[tuple[str, str]]) -> list[dict[str, object]]:
    chain_ids = choose_chain_ids(header for header, _sequence in records)
    sequences = []
    for (_header, sequence), chain_id in zip(records, chain_ids, strict=True):
        sequence_entry: dict[str, object] = {
            "proteinChain": {
                "sequence": sequence,
                "count": 1,
                "id": [chain_id],
            }
        }
        sequences.append(sequence_entry)
    return [{"name": job_name, "sequences": sequences, "covalent_bonds": []}]


def host_cache_checks(repo_root: Path) -> list[str]:
    warnings: list[str] = []
    cache_root = repo_root / "containers" / "Protenix" / "cache"
    if not (cache_root / "checkpoint").exists():
        warnings.append("cache/checkpoint is missing")
    if not (cache_root / "common" / "components.cif").exists():
        warnings.append("cache/common/components.cif is missing")
    if not (cache_root / "search_database" / "pdb_seqres_2022_09_28.fasta").exists():
        warnings.append("cache/search_database/pdb_seqres_2022_09_28.fasta is missing; template search may try to download it inside the container")
    return warnings


def write_json(path: Path, payload: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_container_gateway(compose_project_dir: Path, compose_service: str) -> str:
    python_code = "\n".join(
        [
            "from pathlib import Path",
            "import socket",
            "for line in Path('/proc/net/route').read_text().splitlines()[1:] :",
            "    fields = line.split()",
            "    if len(fields) > 2 and fields[1] == '00000000':",
            "        print(socket.inet_ntoa(bytes.fromhex(fields[2])[::-1]), end='')",
            "        break",
        ]
    )
    command = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-T",
        compose_service,
        "python",
        "-c",
        python_code,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=str(compose_project_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Could not determine the Protenix container gateway for a host-local MSA URL. "
            f"Docker stderr was: {exc.stderr.strip()}"
        ) from exc
    combined_output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    matches = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", combined_output)
    gateway = matches[-1] if matches else ""
    if not gateway:
        raise RuntimeError("Could not determine the container default gateway for local MSA routing")
    return gateway


def container_reachable_msa_url(
    msa_server_url: str, compose_project_dir: Path, compose_service: str
) -> tuple[str, str | None]:
    parsed = urlparse(msa_server_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return msa_server_url, None

    gateway = resolve_container_gateway(compose_project_dir, compose_service)
    rewritten = parsed._replace(netloc=f"{gateway}:{parsed.port}" if parsed.port else gateway)
    return urlunparse(rewritten), gateway


def build_compose_command(args: argparse.Namespace, job_dir: Path, msa_server_url: str) -> list[str]:
    pred_output_dir = "/work/pred_out"
    input_json_path = "/work/input.json"

    command = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-T",
        "-e",
        f"MMSEQS_SERVICE_HOST_URL={msa_server_url}",
        "-v",
        f"{job_dir}:/work:rw",
        args.compose_service,
        "protenix",
        "pred",
        "-i",
        input_json_path,
        "-o",
        pred_output_dir,
        "-s",
        args.seeds,
        "-c",
        str(args.cycle),
        "-p",
        str(args.step),
        "-e",
        str(args.sample),
        "-d",
        args.dtype,
        "-n",
        args.model_name,
        "--use_msa",
        "true",
        "--msa_server_mode",
        "colabfold",
        "--use_template",
        "true",
    ]
    if args.use_default_params:
        command.extend(["--use_default_params", "true"])
    return command


def run_subprocess(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> int:
    args = parse_args()
    repo_root = (
        args.compose_project_dir.resolve()
        if args.compose_project_dir is not None
        else detect_repo_root()
    )
    fasta_path = args.fasta.resolve()
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    records = load_fasta_records(fasta_path)
    job_name = args.name or fasta_path.stem
    job_dir = (
        args.job_dir.resolve()
        if args.job_dir is not None
        else (repo_root / "outputs" / "tools" / "protenix" / fasta_path.stem).resolve()
    )
    job_dir.mkdir(parents=True, exist_ok=True)

    payload = build_input_payload(job_name, records)
    input_json_path = job_dir / "input.json"
    write_json(input_json_path, payload)

    print(f"Wrote input JSON: {input_json_path}")
    print(f"Job directory: {job_dir}")
    print(f"Sequences: {len(records)}")
    print(f"Model: {args.model_name}")
    msa_server_url, gateway = container_reachable_msa_url(
        args.msa_server_url,
        repo_root,
        args.compose_service,
    )
    print(f"MSA server: {msa_server_url}")
    if gateway is not None:
        print(f"Rewrote local host MSA URL through container gateway: {gateway}")

    for warning in host_cache_checks(repo_root):
        print(f"Warning: {warning}", file=sys.stderr)

    if args.build:
        build_command = ["docker", "compose", "build", args.compose_service]
        print("Running:", " ".join(build_command))
        if not args.dry_run:
            run_subprocess(build_command, cwd=repo_root)

    command = build_compose_command(args, job_dir, msa_server_url)
    print("Running:", " ".join(command))
    if args.dry_run:
        return 0

    run_subprocess(command, cwd=repo_root)
    print(f"Prediction output directory: {job_dir / 'pred_out'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
