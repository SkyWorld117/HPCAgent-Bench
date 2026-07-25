"""Show the emitted C signature next to the binding the harness calls, for one kernel.

    python scripts/abi_check.py srad [more kernels...]

The comparison itself is imported from the corpus gate (``test_abi_corpus_agreement``) rather
than repeated here, so this can never certify agreement the gate rejects. Exit status is 1 if
any kernel disagrees.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "hpcagent_bench", "numpy_translators", "tests"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from _bench_yaml import kir_for  # noqa: E402
from test_abi_corpus_agreement import binding_abi, emitted_abi  # noqa: E402

from hpcagent_bench.spec import BenchSpec  # noqa: E402


def report(short: str) -> bool:
    """Print both sides for ``short``; True when they agree exactly."""
    emitted = emitted_abi(kir_for(short, do_lower=True))
    binding = binding_abi(BenchSpec.load(short))
    print(f"=== {short}")
    if emitted == binding:
        print(f"  AGREE ({len(emitted)} args)")
        return True
    print(f"  emitted({len(emitted)}): {emitted}")
    print(f"  binding({len(binding)}): {binding}")

    emitted_names = [n for n, _ in emitted]
    binding_names = [n for n, _ in binding]
    if emitted_names != binding_names:
        # A missing/extra/duplicated name shifts every following positional slot.
        print("  NAMES differ -- the positional call is SHIFTED")
        print(f"    emitted-only: {sorted(set(emitted_names) - set(binding_names))}")
        print(f"    binding-only: {sorted(set(binding_names) - set(emitted_names))}")
        duplicated = sorted({n for n in emitted_names if emitted_names.count(n) > 1})
        if duplicated:
            print(f"    DUPLICATED in emitted (one value is silently dropped): {duplicated}")
    else:
        # Names line up, so a slot is merely misread -- but INTEGER and SSE arguments come from
        # independent register sequences, so an int/float disagreement reads a different register.
        by_name = dict(binding)
        print("  DTYPE differs (name, emitted, binding):")
        for name, dtype in emitted:
            if by_name[name] != dtype:
                print(f"    {name}: emitted={dtype} binding={by_name[name]}")
    return False


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    ok = True
    for short in sys.argv[1:]:
        try:
            ok = report(short) and ok
        except Exception as exc:  # a kernel that cannot lower is itself a failure to report
            print(f"=== {short}\n  ERROR {type(exc).__name__}: {exc}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
