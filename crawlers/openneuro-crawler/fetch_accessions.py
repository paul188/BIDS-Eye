"""
Fetches all dataset accession IDs from the public OpenNeuro S3 bucket
and writes them to accessions.yaml.

Usage:
    python fetch_accessions.py
    python fetch_accessions.py --dry-run   # just print count, don't write
"""

import argparse
import re

import boto3
import yaml
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "openneuro.org"
# OpenNeuro accession IDs look like ds000001, ds000002, ...
_ACCESSION_RE = re.compile(r"^(ds\d{6})/$")


def list_all_accessions() -> list[str]:
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )
    paginator = client.get_paginator("list_objects_v2")
    accessions = []
    print("Listing top-level prefixes in s3://openneuro.org ...")
    for page in paginator.paginate(Bucket=BUCKET, Delimiter="/"):
        for prefix in page.get("CommonPrefixes", []):
            name = prefix["Prefix"]  # e.g. "ds000001/"
            m = _ACCESSION_RE.match(name)
            if m:
                accessions.append(m.group(1))
    return sorted(accessions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print count only, do not write file")
    parser.add_argument("--out", default="accessions.yaml",
                        help="Output YAML file (default: accessions.yaml)")
    args = parser.parse_args()

    accessions = list_all_accessions()
    print(f"Found {len(accessions)} datasets (first: {accessions[0]}, last: {accessions[-1]})")

    if args.dry_run:
        return

    with open(args.out, "w") as f:
        yaml.dump({"accessions": accessions}, f, default_flow_style=False)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
