#!/usr/bin/env python3
"""Clear AGENT_EFFORT on the Kimi arms, and fix the comment that says the ladders differ.

Kimi K2.7 has no reasoning-effort ladder: it always thinks, always preserves thinking, and
Moonshot documents reasoning_effort as a K3-only field. The value was never reaching the model
(its chat template has no reasoning_effort at all), so this is hygiene, not a behaviour fix.

The line is set EMPTY rather than deleted on purpose: agent_driver.py reads
os.environ.get("AGENT_EFFORT", "xhigh"), so a missing line means xhigh, the opposite of intent.
"""
import pathlib
import sys

OLD_COMMENT = """# Top reasoning effort FOR THIS MODEL. The ladders differ: gpt-oss is low/medium/high and
# renders "Reasoning: <value>" verbatim with no guard (so an unknown level is pasted into
# the system prompt, not refused, and an unset one silently means medium), Qwen3.8 is
# low/medium/xhigh and raises on anything else, kimi has no effort mechanism at all.
AGENT_EFFORT=high"""

NEW_COMMENT = """# EMPTY on purpose: Kimi K2.7 has no effort ladder. It always thinks and always preserves
# thinking, and Moonshot documents reasoning_effort as a K3-only field ("do not pass K3's
# reasoning_effort"); the checkpoint's chat template never reads one. Empty, NOT absent --
# agent_driver.py defaults a missing AGENT_EFFORT to xhigh, which is the opposite of this.
AGENT_EFFORT="""


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2] / "containers" / "cluster" / "example-script"
    targets = sorted(root.glob(".env.llr8-kimi27sglang-*"))
    if not targets:
        print("no kimi env files found", file=sys.stderr)
        return 1
    changed = 0
    for path in targets:
        text = path.read_text()
        if NEW_COMMENT in text:
            continue
        if OLD_COMMENT not in text:
            print(f"SKIP (unexpected effort block): {path.name}", file=sys.stderr)
            continue
        path.write_text(text.replace(OLD_COMMENT, NEW_COMMENT))
        changed += 1
    print(f"cleared AGENT_EFFORT in {changed} of {len(targets)} kimi env files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
