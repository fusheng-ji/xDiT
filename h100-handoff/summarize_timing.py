"""Median latency and peak memory per config from run_timing.sh: main (passes a, c) against the PR (b)."""

import json
import re
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
CONFIGS = {"Z1p": "Z-Image, one prompt", "Z2p": "Z-Image, two prompts", "Q2p": "Qwen-Image, two prompts"}
runs = {}
for d in sorted(p for p in out.iterdir() if p.is_dir()):
    m = re.fullmatch(r"(\w+?)_u(\d)_(\w+)_([abc])_[\w-]+", d.name)
    if not m:
        continue
    log = (d / "log.txt").read_text(errors="replace")
    try:
        t = statistics.median(json.loads((d / "timings.json").read_text()))
    except (OSError, ValueError):
        t = None
    peaks = [float(x) for x in re.findall(r"max_memory_allocated ([\d.]+) GiB", log)]
    backend = re.search(r"Using (\w+) as attention backend", log)
    runs[m.groups()] = (t, max(peaks) if peaks else None, backend.group(1) if backend else "?")

print("| Batch | Backend | GPUs | main a / PR / main c (s) | PR vs main | main spread | peak GiB main / PR |")
print("| --- | --- | ---: | --- | ---: | ---: | --- |")
for (cfg, u, be) in sorted({k[:3] for k in runs}, key=lambda k: (list(CONFIGS).index(k[0]), k[2], k[1])):
    a, b, c = (runs.get((cfg, u, be, p), (None, None, "?")) for p in "abc")
    if None in (a[0], b[0], c[0]):
        print(f"| {CONFIGS[cfg]} | {be} | {u} | missing: {a[0]} / {b[0]} / {c[0]} | | | |")
        continue
    main = (a[0] + c[0]) / 2
    used = {a[2], b[2], c[2]}
    note = "" if used == {be} else f" (logged {'/'.join(sorted(used))})"
    print(f"| {CONFIGS[cfg]} | {be}{note} | {u} | {a[0]:.3f} / {b[0]:.3f} / {c[0]:.3f} | {(b[0] / main - 1) * 100:+.1f}% "
          f"| {abs(a[0] - c[0]) / main * 100:.2f}% | {a[1]} / {b[1]} |")
