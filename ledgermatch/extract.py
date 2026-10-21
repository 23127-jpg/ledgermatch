"""Extraction: OCR, then an LLM, then the door in schema.py.

The extractor is deliberately behind an interface with a deterministic
implementation alongside the LLM one. That is not just for tests -- it means the
validation and matching layers can be developed, benchmarked and demonstrated
with no API key and no network, and it makes the boundary obvious: everything
downstream of `ExtractedInvoice` treats its input identically regardless of which
extractor produced it.

The LLM is asked for JSON only, at temperature 0, with the schema in the prompt.
It is still not trusted -- see validate.py.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from .schema import ExtractedInvoice

SYSTEM_PROMPT = """You extract structured data from invoice text.

Return ONLY a JSON object. No prose, no markdown fences, no explanation.

Rules:
- Copy values exactly as printed. Do not compute, correct or infer any number.
- If a field is genuinely absent, use null. Never guess a plausible value.
- Amounts as plain decimal strings without currency symbols or thousands separators.
- Dates as YYYY-MM-DD.

Schema:
{
  "vendor_name": str, "vendor_gstin": str|null, "invoice_no": str,
  "invoice_date": "YYYY-MM-DD", "po_number": str|null, "currency": "INR",
  "line_items": [{"description": str, "hsn_sac": str|null, "quantity": str,
                  "unit": str|null, "rate": str, "line_total": str}],
  "subtotal": str,
  "taxes": [{"kind": "CGST|SGST|IGST|UTGST|CESS", "rate_pct": str, "amount": str}],
  "total": str
}
"""


class Extractor(Protocol):
    def extract(self, raw_text: str) -> dict: ...


class ExtractionError(RuntimeError):
    pass


# ---- OCR --------------------------------------------------------------
def ocr(path: str | Path) -> str:
    """Scanned page -> text. Falls back to a sidecar .txt when OCR is absent,
    so the pipeline is runnable without a Tesseract install."""
    p = Path(path)
    sidecar = p.with_suffix(".txt")
    if sidecar.exists():
        return sidecar.read_text()
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise ExtractionError(
            f"no OCR backend and no sidecar text at {sidecar}: {e}") from e
    return pytesseract.image_to_string(Image.open(p))


# ---- LLM extractor ----------------------------------------------------
class LLMExtractor:
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def extract(self, raw_text: str) -> dict:
        if not self.api_key:
            raise ExtractionError("ANTHROPIC_API_KEY not set")
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=self.model, max_tokens=2000, temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": raw_text}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _parse_json_strict(text)


def _parse_json_strict(text: str) -> dict:
    """Models sometimes wrap JSON in fences despite instructions. Strip those,
    but do not attempt to repair malformed JSON -- a model that cannot emit
    valid JSON has not understood the document either."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"model did not return valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ExtractionError("model returned JSON that is not an object")
    return obj


# ---- deterministic extractor (no API, no network) --------------------
class RegexExtractor:
    """A plain parser for the sample invoice layout.

    Useful as a baseline: if the LLM cannot beat a regex on structured layouts,
    it is not earning its cost or its latency.
    """

    def extract(self, raw_text: str) -> dict:
        def find(pattern, default=None, flags=re.I | re.M):
            m = re.search(pattern, raw_text, flags)
            return m.group(1).strip() if m else default

        items = []
        for m in re.finditer(
            r"^\s*(?P<desc>[A-Za-z][A-Za-z0-9 ,./()-]+?)\s{2,}"
            r"(?P<hsn>\d{4,8})?\s*"
            r"(?P<qty>[\d,]+(?:\.\d+)?)\s+(?P<unit>[A-Za-z]{2,6})?\s*"
            r"(?P<rate>[\d,]+\.\d{2})\s+(?P<total>[\d,]+\.\d{2})\s*$",
            raw_text, re.M,
        ):
            d = m.groupdict()
            items.append({
                "description": d["desc"].strip(), "hsn_sac": d["hsn"],
                "quantity": _num(d["qty"]), "unit": d["unit"],
                "rate": _num(d["rate"]), "line_total": _num(d["total"]),
            })

        taxes = []
        for m in re.finditer(r"\b(CGST|SGST|IGST|UTGST|CESS)\s*@?\s*"
                             r"([\d.]+)\s*%\s*[:\s]\s*([\d,]+\.\d{2})", raw_text, re.I):
            taxes.append({"kind": m.group(1).upper(), "rate_pct": m.group(2),
                          "amount": _num(m.group(3))})

        return {
            "vendor_name": find(r"^\s*(?:From|Vendor)\s*:\s*(.+)$"),
            "vendor_gstin": find(r"GSTIN\s*:?\s*([0-9A-Z]{15})"),
            "invoice_no": find(r"Invoice\s*(?:No|#)\s*:?\s*([A-Za-z0-9/\-]+)"),
            "invoice_date": _date(find(r"(?:Invoice\s*)?Date\s*:?\s*([\d]{2,4}[-/][\d]{2}[-/][\d]{2,4})")),
            "po_number": find(r"\bPO\s*(?:No|#)?\s*:?\s*([A-Za-z0-9/\-]+)"),
            "currency": "INR",
            "line_items": items,
            "subtotal": _num(find(r"Sub\s*-?\s*total\s*:?\s*([\d,]+\.\d{2})", "0")),
            "taxes": taxes,
            # anchored so "Sub-total" cannot be mistaken for the grand total --
            # an unanchored \bTotal\b matches the subtotal line first and silently
            # under-states the invoice by the entire tax amount
            "total": _num(find(r"(?:Grand\s*Total|^[ \t]*Total)\s*:?\s*"
                               r"(?:INR\s*)?([\d,]+\.\d{2})", "0")),
        }


def _num(s: str | None) -> str | None:
    return s.replace(",", "").strip() if s else s


def _date(s: str | None) -> str | None:
    if not s:
        return None
    s = s.replace("/", "-")
    parts = s.split("-")
    if len(parts[0]) == 2:  # DD-MM-YYYY
        parts = parts[::-1]
    return "-".join(parts)


# ---- the door ---------------------------------------------------------
def to_invoice(payload: dict) -> ExtractedInvoice:
    """Validate raw extractor output against the schema.

    A ValidationError here is the system working, not failing: the model
    produced something that cannot represent an invoice, and it is being stopped
    at the boundary rather than three layers deeper.
    """
    try:
        return ExtractedInvoice.model_validate(payload)
    except ValidationError as e:
        raise ExtractionError(
            f"extractor output rejected by schema: {e.error_count()} problem(s)\n{e}"
        ) from e
