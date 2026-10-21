"""The reliability layer.

The model reads the document. It does not get to do arithmetic, and it does not
get the last word on any number that will hit the ledger.

Every quantitative relationship on an invoice is redundant -- qty x rate should
equal the line total, the line totals should equal the subtotal, the tax should
equal the subtotal times the rate, and the sum should equal the printed total.
An LLM that misreads a digit will usually break one of those identities without
breaking the others, so recomputing them in Python catches exactly the failure
mode that model confidence cannot see: a fluent, well-formed, wrong number.

Where the recomputed figure disagrees with the claimed one, the recomputed figure
wins and the claim is recorded as a finding. Where they cannot be reconciled at
all, the invoice is blocked.
"""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP

from .schema import ExtractedInvoice, Finding, Severity, ValidationReport

# A rupee of slack absorbs legitimate per-line rounding on a printed invoice.
# Anything wider than this is a real disagreement, not a rounding artefact.
LINE_TOLERANCE = Decimal("1.00")
TOTAL_TOLERANCE = Decimal("1.00")
TAX_TOLERANCE = Decimal("1.00")

GST_SLABS = {Decimal(x) for x in ("0", "0.1", "0.25", "1", "3", "5", "12", "18", "28")}
HALF_SLABS = {s / 2 for s in GST_SLABS}


def _q(v: Decimal) -> Decimal:
    return Decimal(v).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def validate(inv: ExtractedInvoice, raw_text: str = "") -> ValidationReport:
    findings: list[Finding] = []
    conf: dict[str, float] = {}

    # ---- 1. line arithmetic -----------------------------------------
    computed_subtotal = Decimal("0")
    for i, li in enumerate(inv.line_items):
        expected = _q(li.quantity * li.rate)
        if abs(expected - _q(li.line_total)) > LINE_TOLERANCE:
            findings.append(Finding(
                code="LINE_ARITHMETIC", severity=Severity.ERROR,
                field=f"line_items[{i}].line_total",
                detail=f"qty x rate does not equal the printed line total (line {i + 1})",
                claimed=str(_q(li.line_total)), computed=str(expected)))
            computed_subtotal += expected  # trust the arithmetic, not the print
        else:
            computed_subtotal += _q(li.line_total)
    computed_subtotal = _q(computed_subtotal)

    # ---- 2. subtotal ------------------------------------------------
    if abs(computed_subtotal - _q(inv.subtotal)) > TOTAL_TOLERANCE:
        findings.append(Finding(
            code="SUBTOTAL_MISMATCH", severity=Severity.ERROR, field="subtotal",
            detail="line items do not sum to the claimed subtotal",
            claimed=str(_q(inv.subtotal)), computed=str(computed_subtotal)))

    # ---- 3. tax ------------------------------------------------------
    computed_tax = Decimal("0")
    for t in inv.taxes:
        expected = _q(computed_subtotal * t.rate_pct / Decimal(100))
        computed_tax += _q(t.amount)
        # CGST and SGST each carry half the slab -- an 18% supply is 9% + 9% --
        # so the valid rate set differs by tax kind.
        valid = HALF_SLABS if t.kind in ("CGST", "SGST", "UTGST") else GST_SLABS
        if t.rate_pct not in valid:
            findings.append(Finding(
                code="TAX_RATE_NOT_A_SLAB", severity=Severity.WARNING,
                field=f"taxes.{t.kind}",
                detail=f"{t.rate_pct}% is not a valid {t.kind} rate",
                claimed=str(t.rate_pct)))
        if abs(expected - _q(t.amount)) > TAX_TOLERANCE:
            findings.append(Finding(
                code="TAX_ARITHMETIC", severity=Severity.ERROR,
                field=f"taxes.{t.kind}",
                detail=f"{t.kind} at {t.rate_pct}% does not match the taxable value",
                claimed=str(_q(t.amount)), computed=str(expected)))

    kinds = [t.kind for t in inv.taxes]
    if "IGST" in kinds and ("CGST" in kinds or "SGST" in kinds):
        findings.append(Finding(
            code="IGST_WITH_CGST_SGST", severity=Severity.ERROR, field="taxes",
            detail="a supply is either inter-state or intra-state, never both"))
    if "CGST" in kinds and "SGST" in kinds:
        c = next(t for t in inv.taxes if t.kind == "CGST")
        s = next(t for t in inv.taxes if t.kind == "SGST")
        if abs(_q(c.amount) - _q(s.amount)) > TAX_TOLERANCE:
            findings.append(Finding(
                code="CGST_SGST_ASYMMETRY", severity=Severity.ERROR, field="taxes",
                detail="CGST and SGST must be equal on an intra-state supply",
                claimed=f"CGST {_q(c.amount)} / SGST {_q(s.amount)}"))

    # ---- 4. grand total ---------------------------------------------
    computed_total = _q(computed_subtotal + computed_tax)
    total_ok = abs(computed_total - _q(inv.total)) <= TOTAL_TOLERANCE
    if not total_ok:
        findings.append(Finding(
            code="TOTAL_MISMATCH", severity=Severity.ERROR, field="total",
            detail="subtotal plus tax does not equal the claimed invoice total",
            claimed=str(_q(inv.total)), computed=str(computed_total)))

    # ---- 5. cross-check the total against the raw OCR text ----------
    # Independent evidence: if the model's total never appears in the scanned
    # text at all, the model did not read it -- it produced it.
    #
    # Severity is conditional on purpose. The arithmetic check and the source
    # check are independent, and neither is perfect on its own: OCR mangles
    # printed totals, and arithmetic cannot see a coherently fabricated one.
    # Failing both means the number has no support from any direction, which
    # blocks. Failing only this one is a flag, not a rejection -- otherwise every
    # invoice printed as "Rs. 1,05,798/-" would be refused for a formatting quirk.
    if raw_text:
        if not _appears_in_text(_q(inv.total), raw_text):
            findings.append(Finding(
                code="TOTAL_NOT_IN_SOURCE",
                severity=Severity.ERROR if not total_ok else Severity.WARNING,
                field="total",
                detail="claimed total does not appear anywhere in the OCR text",
                claimed=str(_q(inv.total))))
        if inv.invoice_no.lower() not in raw_text.lower():
            findings.append(Finding(
                code="INVOICE_NO_NOT_IN_SOURCE", severity=Severity.WARNING,
                field="invoice_no",
                detail="claimed invoice number not found in the OCR text",
                claimed=inv.invoice_no))

    # ---- 6. per-field confidence ------------------------------------
    bad_fields = {f.field for f in findings if f.severity == Severity.ERROR}
    for field in ("vendor_name", "vendor_gstin", "invoice_no", "invoice_date",
                  "subtotal", "total", "po_number"):
        base = 0.99 if field not in bad_fields else 0.30
        if field == "vendor_gstin" and inv.vendor_gstin is None:
            base = 0.0
        if field == "po_number" and inv.po_number is None:
            base = 0.0
        if raw_text and field in ("invoice_no",) and inv.invoice_no.lower() not in raw_text.lower():
            base = min(base, 0.45)
        conf[field] = base
    conf["total"] = 0.99 if total_ok else 0.25

    errors = [f for f in findings if f.severity == Severity.ERROR]
    return ValidationReport(
        ok=not errors, findings=findings, field_confidence=conf,
        corrected_total=computed_total if not total_ok else None,
    )


def _appears_in_text(amount: Decimal, text: str) -> bool:
    """Look for the amount in the raw text, tolerating comma grouping."""
    plain = f"{amount:.2f}"
    grouped = f"{amount:,.2f}"
    indian = _indian_grouping(amount)
    haystack = text.replace(" ", "")
    return any(c.replace(" ", "") in haystack for c in (plain, grouped, indian))


def _indian_grouping(amount: Decimal) -> str:
    """1234567.89 -> 12,34,567.89 (lakh/crore grouping used on Indian invoices)."""
    s = f"{amount:.2f}"
    whole, frac = s.split(".")
    neg = whole.startswith("-")
    whole = whole.lstrip("-")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = re.sub(r"(?<=\d)(?=(\d\d)+$)", ",", head)
        whole = f"{head},{tail}"
    return ("-" if neg else "") + whole + "." + frac
