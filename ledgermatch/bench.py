"""Does the reliability layer actually catch anything?

You cannot answer that by staring at correct invoices. So this harness takes a
valid extraction and corrupts it the way a model realistically misreads a
document -- a transposed digit, a dropped leading digit, an OCR 5/6 confusion, a
tax line copied from the wrong row -- and measures what fraction the validator
blocks.

The corruptions are chosen to be *fluent*: every one of them produces a
well-formed, confident-looking JSON object that a schema check alone would wave
through. Only recomputing the arithmetic catches them.
"""

from __future__ import annotations

import copy
import random
from decimal import Decimal

from .extract import to_invoice
from .validate import validate


def _transpose_digits(s: str) -> str:
    digits = [i for i, c in enumerate(s) if c.isdigit()]
    if len(digits) < 2:
        return s
    i, j = digits[0], digits[1]
    ls = list(s)
    ls[i], ls[j] = ls[j], ls[i]
    return "".join(ls)


CORRUPTIONS = {
    # a misread digit in one line total: line arithmetic breaks
    "line_total_transposed":
        lambda d: _set(d, "line_items.0.line_total",
                       _transpose_digits(str(d["line_items"][0]["line_total"]))),
    # dropped leading digit on a rate: 1250 -> 250
    "rate_leading_digit_dropped":
        lambda d: _set(d, "line_items.0.rate", str(d["line_items"][0]["rate"])[1:]),
    # OCR 5/6 confusion inside the subtotal
    "subtotal_ocr_5_for_6":
        lambda d: _set(d, "subtotal", str(d["subtotal"]).replace("6", "5", 1)),
    # grand total hallucinated: plausible magnitude, not in the document
    "total_invented":
        lambda d: _set(d, "total", str(Decimal(str(d["total"])) + Decimal("1000.00"))),
    # tax amount lifted from the wrong row
    "tax_amount_swapped":
        lambda d: _set(d, "taxes.0.amount",
                       str(Decimal(str(d["taxes"][0]["amount"])) * Decimal("1.5"))
                       if d.get("taxes") else None),
    # CGST/SGST asymmetry -- one side misread
    "cgst_sgst_asymmetry":
        lambda d: _set(d, "taxes.1.amount",
                       str(Decimal(str(d["taxes"][1]["amount"])) + Decimal("450.00"))
                       if len(d.get("taxes", [])) > 1 else None),
    # quantity misread: 200 -> 20
    "quantity_digit_dropped":
        lambda d: _set(d, "line_items.1.quantity",
                       str(d["line_items"][1]["quantity"])[:-1]
                       if len(d.get("line_items", [])) > 1 else None),
}


def _set(d: dict, path: str, value):
    if value is None:
        return None
    out = copy.deepcopy(d)
    node = out
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[int(p)] if p.isdigit() else node[p]
    key = parts[-1]
    node[int(key) if key.isdigit() else key] = value
    return out


def run_benchmark(clean_payload: dict, raw_text: str, seed: int = 3) -> dict:
    random.seed(seed)
    results = {}

    base = validate(to_invoice(clean_payload), raw_text)
    results["clean_invoice_passes"] = base.ok

    caught = 0
    total = 0
    detail = {}
    for name, fn in CORRUPTIONS.items():
        payload = fn(clean_payload)
        if payload is None:
            continue
        total += 1
        try:
            inv = to_invoice(payload)
        except Exception:
            caught += 1
            detail[name] = "blocked at schema"
            continue
        rep = validate(inv, raw_text)
        if not rep.ok:
            caught += 1
            detail[name] = "blocked: " + ", ".join(sorted({f.code for f in rep.errors}))
        else:
            detail[name] = "MISSED"
    results["corruptions_tested"] = total
    results["corruptions_caught"] = caught
    results["catch_rate_pct"] = round(caught / total * 100, 1) if total else 0.0
    results["detail"] = detail
    return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    from .extract import RegexExtractor

    raw = Path("samples/INV-2026-0042.txt").read_text()
    payload = RegexExtractor().extract(raw)
    print(json.dumps(run_benchmark(payload, raw), indent=2))
