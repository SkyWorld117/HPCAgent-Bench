#!/usr/bin/env python3
"""Restore the Kimi arms to the SGLang attention config their passing smokes actually used.

edb3da08 shipped these arms with BOTH SGLANG_USE_AITER=1 and --attention-backend triton, and
that is the pair smokes 604941/604942 passed the long-context accuracy gate under. 2e4a5ce1
(2026-08-27 13:32) dropped the pin on the belief the smokes had run unpinned; they had not.

Unpinned, SGLang selects aiter, whose attention kernels JIT-compile on the FIRST request behind
a cross-process baton lock, and the build outlives SGLang's hard 600 s warmup read timeout: the
server logs "Initialization failed. warmup error" and is killed before serving a token. Three
qwen38 servers died exactly there (610217/610220/610225) with the cache warmer each time; 610229
pinned triton and served 163 tok/s at 9/9.
"""
import pathlib
import re
import sys

PIN = "--attention-backend triton "
ANCHOR = 'SGLANG_EXTRA_ARGS="--trust-remote-code '
OLD_COMMENT = """# No --attention-backend: unpinned, SGLang logs "Attention backend not specified. Use aiter
# backend by default" and that is what the two smokes which passed the long-context accuracy
# gate served (604941/604942). Pinning triton routes MLA away from aiter, and it is the one
# knob SGLANG_USE_AITER=1 cannot override -- so the concurrency curve quoted above, measured
# with the pin in place, is a FLOOR for this configuration and wants a re-measure."""
NEW_COMMENT = """# --attention-backend triton, WITH SGLANG_USE_AITER=1: that pair is what smokes 604941/604942
# passed the long-context accuracy gate under, and the concurrency curve quoted above was
# measured with it. Leaving the backend unpinned lets SGLang pick aiter, whose attention kernels
# JIT-compile on the FIRST request behind a baton lock and outlive SGLang's hard 600 s warmup
# read timeout -- "Initialization failed. warmup error", killed before serving a token. Three
# qwen38 servers died there (610217/610220/610225) with the JIT cache warmer each time and no
# change in outcome; 610229 pinned triton and served 163 tok/s at 9/9."""


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2] / "containers" / "cluster" / "example-script"
    targets = sorted(root.glob(".env.llr8-kimi27sglang-*"))
    if not targets:
        print("no kimi env files found", file=sys.stderr)
        return 1
    pinned = restored = recommented = 0
    for path in targets:
        text = path.read_text()
        line = re.search(r"^SGLANG_EXTRA_ARGS=.*$", text, re.M)
        if line is None:
            print(f"SKIP (no SGLANG_EXTRA_ARGS): {path.name}", file=sys.stderr)
            continue
        if "--attention-backend" not in line.group(0):
            if ANCHOR not in text:
                print(f"SKIP (unexpected args line): {path.name}", file=sys.stderr)
                continue
            text = text.replace(ANCHOR, ANCHOR + PIN, 1)
            pinned += 1
        if "SGLANG_USE_AITER=0" in text:
            text = text.replace("SGLANG_USE_AITER=0", "SGLANG_USE_AITER=1", 1)
            restored += 1
        if OLD_COMMENT in text:
            text = text.replace(OLD_COMMENT, NEW_COMMENT, 1)
            recommented += 1
        path.write_text(text)
    print(f"pinned triton: {pinned}, SGLANG_USE_AITER back to 1: {restored}, "
          f"comment corrected: {recommented}, of {len(targets)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
