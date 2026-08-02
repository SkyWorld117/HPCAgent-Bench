"""Submit final benchmark code through the configured remote endpoint.

Use this tool when the implementation is ready to finalize. It sends the final
source code and language to the submit endpoint. The remote service decides how
to store, grade, rank, or finalize the submission.

Fields:
  code: required string with the final source code.
  language: required string such as c, cpp, cuda, hip, fortran, or python.
  kernel: optional benchmark kernel identifier.
  metadata: optional object for run IDs, rank, preset, or other service-specific data.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from http_json import endpoint, post_json


DESCRIPTION = "Submit final code for the benchmark service to record and grade."
INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Full final source code to submit."},
        "language": {"type": "string", "description": "Implementation language, e.g. cpp, cuda, hip."},
        "kernel": {"type": "string", "description": "Optional benchmark kernel identifier."},
        "metadata": {"type": "object", "description": "Optional routing/context data for the benchmark service."},
    },
    "required": ["code", "language"],
}


def run(payload: Dict[str, Any]) -> Dict[str, Any]:
    return post_json(endpoint("submit"), payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--json", help="JSON payload. If omitted, stdin is read.")
    args = parser.parse_args()
    payload = json.loads(args.json if args.json is not None else sys.stdin.read())
    print(json.dumps(run(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
