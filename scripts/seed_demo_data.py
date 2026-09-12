"""Seed a freshly started backend with a handful of well-known molecules so
the frontend's History page has something to show immediately after
`docker compose up -d`, instead of an empty state.

Usage (after the stack is up):

    python scripts/seed_demo_data.py
    python scripts/seed_demo_data.py --base-url http://localhost:8000

Uses only the Python standard library (urllib) - no extra dependency to
install just to run a one-off seed script. Idempotent-ish: POSTing the same
SMILES again just creates another history row with a new id (the backend
has no dedup), so re-running is harmless, just adds duplicates.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Same SMILES as frontend/src/admetConstants.js's EXAMPLE_MOLECULES (Aspirin,
# Thalidomide, Caffeine) plus Ibuprofen, a well-known NSAID not currently in
# the frontend's example chips but a good addition for demo variety.
DEMO_MOLECULES = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("Thalidomide", "C1CC(=O)NC(=O)C1N1C(=O)c2ccccc2C1=O"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
]


def wait_for_health(base_url: str, timeout_s: float = 60.0) -> None:
    """Poll GET /health until the model is loaded, or give up after timeout_s."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                body = json.loads(resp.read())
                if body.get("model_loaded"):
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    raise SystemExit(f"Backend at {base_url} did not become healthy within {timeout_s}s")


def seed(base_url: str) -> None:
    wait_for_health(base_url)
    for label, smiles in DEMO_MOLECULES:
        payload = json.dumps({"smiles": smiles}).encode()
        req = urllib.request.Request(
            f"{base_url}/admet-profile",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                print(f"seeded {label} ({smiles}) -> id={result['id']}")
        except urllib.error.HTTPError as exc:
            print(f"FAILED to seed {label} ({smiles}): {exc.code} {exc.read().decode()}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Backend base URL (default: http://localhost:8000, matching docker-compose.yml's exposed port)",
    )
    args = parser.parse_args()
    seed(args.base_url)
