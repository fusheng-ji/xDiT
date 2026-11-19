"""Tabulate run_856.sh: python summarize_856.py <dir>. Reads the JSON report each parity log ends with."""

import json
import re
import sys
from pathlib import Path


def report(log):
    text = log.read_text(errors="replace")
    start = text.rfind("\n{")
    if start < 0:
        error = re.findall(r"^\w*(?:Error|Exception)\b.*$", text, re.M)
        return None, (error[-1][:120] if error else "no report")
    try:
        return json.loads(text[start + 1 :]), None
    except json.JSONDecodeError:
        return None, "unreadable report"


rows = [
    "| Backend | Model | Ulysses | Tree | Same-input max error | Sample PSNR (dB) |",
    "| --- | --- | ---: | --- | ---: | --- |",
]
for log in sorted(Path(sys.argv[1]).glob("e2e-*.log")):
    tree, model, ulysses, backend = re.match(r"e2e-(\w+)-([\w.-]+)-u(\d)-(\w+)\.log", log.name).groups()
    data, error = report(log)
    if data is None:
        rows.append(f"| {backend} | {model} | {ulysses} | {tree} | {error} | |")
        continue
    psnr = ", ".join("inf" if p == float("inf") else f"{p:.1f}" for p in data.get("sample_psnr_db", []))
    rows.append(f"| {backend} | {model} | {ulysses} | {tree} | {data['same_input']['max']:.3g} | {psnr} |")
print("\n".join(rows))
