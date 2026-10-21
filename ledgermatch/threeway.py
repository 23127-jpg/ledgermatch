"""Three-way match: purchase order, goods receipt, invoice.

Validation proves an invoice is internally consistent. It says nothing about
whether we ordered the goods, or received them. A perfectly-formed invoice for
400 slide cassettes we never ordered will pass every arithmetic check in
validate.py -- catching that is this module's job.

The three documents answer three different questions:

    PO       what we agreed to buy, and at what price
    GRN      what actually arrived
    Invoice  what we are being asked to pay for

Any disagreement between them is an exception. Tolerances exist because real
deliveries are imperfect, but they are per-line and configurable, never a blanket
"close enough".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

from .schema import ExtractedInvoice, Finding, Severity


def _q(v) -> Decimal:
    return Decimal(v).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass
class POLine:
    line_no: int
    sku: str
    description: str
    quantity: Decimal
    rate: Decimal


@dataclass
class PurchaseOrder:
    po_number: str
    vendor_name: str
    vendor_gstin: str | None
    currency: str
    lines: list[POLine]


@dataclass
class GRNLine:
    po_line_no: int
    sku: str
    quantity_received: Decimal


@dataclass
class GoodsReceipt:
    grn_number: str
    po_number: str
    lines: list[GRNLine]


@dataclass
class Tolerance:
    price_pct: Decimal = Decimal("2")     # unit price may drift 2%
    quantity_units: Decimal = Decimal("1")  # short/over delivery of 1 unit
    total_abs: Decimal = Decimal("50")    # absolute slack on the invoice total


@dataclass
class MatchOutcome:
    matched: bool
    findings: list[Finding] = field(default_factory=list)
    matched_lines: int = 0
    unmatched_lines: int = 0

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]


def three_way_match(inv: ExtractedInvoice, po: PurchaseOrder | None,
                    grn: GoodsReceipt | None, tol: Tolerance | None = None,
                    seen_invoice_keys: set[tuple[str, str]] | None = None) -> MatchOutcome:
    tol = tol or Tolerance()
    findings: list[Finding] = []

    # ---- duplicate invoice ------------------------------------------
    # The single most common AP loss: paying the same invoice twice because it
    # arrived by email and again on paper. Keyed on vendor + number, not number
    # alone, since numbering restarts every year across vendors.
    if seen_invoice_keys is not None:
        key = (inv.vendor_name.strip().lower(), inv.invoice_no.strip().lower())
        if key in seen_invoice_keys:
            findings.append(Finding(
                code="DUPLICATE_INVOICE", severity=Severity.ERROR, field="invoice_no",
                detail="this vendor and invoice number have already been processed",
                claimed=inv.invoice_no))

    # ---- PO exists ---------------------------------------------------
    if po is None:
        findings.append(Finding(
            code="NO_PURCHASE_ORDER", severity=Severity.ERROR, field="po_number",
            detail="no purchase order found for this invoice",
            claimed=inv.po_number or "(none quoted)"))
        return MatchOutcome(False, findings, 0, len(inv.line_items))

    if inv.po_number and inv.po_number.strip().upper() != po.po_number.strip().upper():
        findings.append(Finding(
            code="PO_NUMBER_MISMATCH", severity=Severity.ERROR, field="po_number",
            detail="invoice quotes a different PO than the one supplied",
            claimed=inv.po_number, computed=po.po_number))

    # ---- vendor identity ---------------------------------------------
    if _norm(inv.vendor_name) != _norm(po.vendor_name):
        findings.append(Finding(
            code="VENDOR_MISMATCH", severity=Severity.ERROR, field="vendor_name",
            detail="invoice vendor differs from the PO vendor",
            claimed=inv.vendor_name, computed=po.vendor_name))
    if inv.vendor_gstin and po.vendor_gstin and inv.vendor_gstin != po.vendor_gstin:
        findings.append(Finding(
            code="GSTIN_MISMATCH", severity=Severity.ERROR, field="vendor_gstin",
            detail="invoice GSTIN differs from the GSTIN on the PO",
            claimed=inv.vendor_gstin, computed=po.vendor_gstin))
    if inv.currency != po.currency:
        findings.append(Finding(
            code="CURRENCY_MISMATCH", severity=Severity.ERROR, field="currency",
            detail="invoice currency differs from the PO currency",
            claimed=inv.currency, computed=po.currency))

    # ---- line-level match --------------------------------------------
    received: dict[int, Decimal] = {}
    if grn:
        if grn.po_number.strip().upper() != po.po_number.strip().upper():
            findings.append(Finding(
                code="GRN_PO_MISMATCH", severity=Severity.ERROR, field="grn",
                detail="goods receipt belongs to a different purchase order",
                claimed=grn.po_number, computed=po.po_number))
        for gl in grn.lines:
            received[gl.po_line_no] = received.get(gl.po_line_no, Decimal(0)) + gl.quantity_received

    po_by_desc = {_norm(l.description): l for l in po.lines}
    matched = 0

    for i, li in enumerate(inv.line_items):
        pol = po_by_desc.get(_norm(li.description))
        if pol is None:
            findings.append(Finding(
                code="LINE_NOT_ON_PO", severity=Severity.ERROR,
                field=f"line_items[{i}]",
                detail="invoiced item does not appear on the purchase order",
                claimed=li.description))
            continue

        # price
        if pol.rate > 0:
            drift = abs(_q(li.rate) - _q(pol.rate)) / _q(pol.rate) * Decimal(100)
            if drift > tol.price_pct:
                findings.append(Finding(
                    code="PRICE_VARIANCE", severity=Severity.ERROR,
                    field=f"line_items[{i}].rate",
                    detail=f"unit price is {drift:.1f}% off the PO "
                           f"(tolerance {tol.price_pct}%)",
                    claimed=str(_q(li.rate)), computed=str(_q(pol.rate))))

        # quantity vs PO -- over-billing
        if li.quantity - pol.quantity > tol.quantity_units:
            findings.append(Finding(
                code="OVER_BILLED_VS_PO", severity=Severity.ERROR,
                field=f"line_items[{i}].quantity",
                detail="invoiced quantity exceeds the ordered quantity",
                claimed=str(li.quantity), computed=str(pol.quantity)))

        # quantity vs GRN -- billed for goods not received
        if grn is not None:
            got = received.get(pol.line_no, Decimal(0))
            if li.quantity - got > tol.quantity_units:
                findings.append(Finding(
                    code="BILLED_ABOVE_RECEIPT", severity=Severity.ERROR,
                    field=f"line_items[{i}].quantity",
                    detail="invoiced quantity exceeds what was actually received",
                    claimed=str(li.quantity), computed=str(got)))
            elif got - li.quantity > tol.quantity_units:
                findings.append(Finding(
                    code="SHORT_BILLED", severity=Severity.WARNING,
                    field=f"line_items[{i}].quantity",
                    detail="received more than invoiced; expect a further invoice",
                    claimed=str(li.quantity), computed=str(got)))
        matched += 1

    if grn is None:
        findings.append(Finding(
            code="NO_GOODS_RECEIPT", severity=Severity.WARNING, field="grn",
            detail="no goods receipt yet; two-way matched against the PO only"))

    # ---- invoice value against the PO commitment ---------------------
    po_value = _q(sum(l.quantity * l.rate for l in po.lines))
    if _q(inv.subtotal) - po_value > tol.total_abs:
        findings.append(Finding(
            code="INVOICE_EXCEEDS_PO_VALUE", severity=Severity.ERROR, field="subtotal",
            detail="invoice value exceeds the total committed on the PO",
            claimed=str(_q(inv.subtotal)), computed=str(po_value)))

    errors = [f for f in findings if f.severity == Severity.ERROR]
    return MatchOutcome(
        matched=not errors, findings=findings,
        matched_lines=matched, unmatched_lines=len(inv.line_items) - matched)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())
