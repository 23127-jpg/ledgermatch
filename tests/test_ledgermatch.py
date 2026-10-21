"""Tests for the reliability layer and the three-way match.

The load-bearing tests here are the ones asserting that a *fluent, well-formed,
wrong* extraction gets blocked. Anyone can validate a good invoice.
"""

from datetime import date
from decimal import Decimal

import pytest

from ledgermatch.extract import ExtractionError, RegexExtractor, to_invoice
from ledgermatch.schema import ExtractedInvoice, Severity
from ledgermatch.threeway import (GoodsReceipt, GRNLine, POLine, PurchaseOrder,
                                  Tolerance, three_way_match)
from ledgermatch.validate import validate

RAW = open("samples/INV-2026-0042.txt").read()


def clean_payload():
    return RegexExtractor().extract(RAW)


def clean_invoice() -> ExtractedInvoice:
    return to_invoice(clean_payload())


def po(qty1="40", rate1="1250.00") -> PurchaseOrder:
    return PurchaseOrder(
        po_number="PO-2026-0311", vendor_name="Meridian Scientific Supplies Pvt Ltd",
        vendor_gstin="29ABCDE1234F1Z5", currency="INR",
        lines=[
            POLine(1, "MB-LP-40", "Microtome Blades Low Profile", Decimal(qty1), Decimal(rate1)),
            POLine(2, "SC-WH-200", "Slide Cassettes White", Decimal("200"), Decimal("145.50")),
            POLine(3, "XY-RG-5L", "Xylene Reagent Grade 5L", Decimal("12"), Decimal("880.00")),
        ])


def grn(q1="40") -> GoodsReceipt:
    return GoodsReceipt("GRN-2026-0455", "PO-2026-0311", [
        GRNLine(1, "MB-LP-40", Decimal(q1)),
        GRNLine(2, "SC-WH-200", Decimal("200")),
        GRNLine(3, "XY-RG-5L", Decimal("12")),
    ])


# ---- schema boundary -------------------------------------------------
def test_clean_invoice_extracts_and_validates():
    rep = validate(clean_invoice(), RAW)
    assert rep.ok and not rep.findings


def test_schema_rejects_missing_required_field():
    d = clean_payload()
    d["vendor_name"] = None
    with pytest.raises(ExtractionError):
        to_invoice(d)


def test_schema_rejects_unknown_keys():
    """extra='forbid' -- a model inventing a field is a signal, not a bonus."""
    d = clean_payload()
    d["confidence_score"] = 0.98
    with pytest.raises(ExtractionError):
        to_invoice(d)


def test_schema_rejects_negative_quantity():
    d = clean_payload()
    d["line_items"][0]["quantity"] = "-5"
    with pytest.raises(ExtractionError):
        to_invoice(d)


def test_schema_rejects_malformed_gstin():
    d = clean_payload()
    d["vendor_gstin"] = "NOTAGSTIN"
    with pytest.raises(ExtractionError):
        to_invoice(d)


# ---- arithmetic recomputation ----------------------------------------
def test_line_arithmetic_error_is_blocking():
    d = clean_payload()
    d["line_items"][0]["line_total"] = "5000.00"  # was 50000.00
    rep = validate(to_invoice(d), RAW)
    assert not rep.ok
    assert "LINE_ARITHMETIC" in {f.code for f in rep.errors}


def test_subtotal_that_does_not_sum_is_blocking():
    d = clean_payload()
    d["subtotal"] = "89560.00"
    rep = validate(to_invoice(d), RAW)
    assert "SUBTOTAL_MISMATCH" in {f.code for f in rep.errors}


def test_tax_that_does_not_match_taxable_value_is_blocking():
    d = clean_payload()
    d["taxes"][0]["amount"] = "9069.40"
    rep = validate(to_invoice(d), RAW)
    assert "TAX_ARITHMETIC" in {f.code for f in rep.errors}


def test_cgst_sgst_must_be_symmetric():
    d = clean_payload()
    d["taxes"][1]["amount"] = "8519.40"
    rep = validate(to_invoice(d), RAW)
    assert "CGST_SGST_ASYMMETRY" in {f.code for f in rep.errors}


def test_igst_cannot_coexist_with_cgst():
    d = clean_payload()
    d["taxes"].append({"kind": "IGST", "rate_pct": "18", "amount": "16138.80"})
    rep = validate(to_invoice(d), RAW)
    assert "IGST_WITH_CGST_SGST" in {f.code for f in rep.errors}


def test_hallucinated_total_fails_both_checks_and_is_blocked():
    """The decisive test. A total that neither reconciles nor appears anywhere in
    the document has no support from any direction, so it blocks."""
    d = clean_payload()
    d["total"] = "115798.80"
    rep = validate(to_invoice(d), RAW)
    assert not rep.ok
    codes = {f.code for f in rep.errors}
    assert "TOTAL_MISMATCH" in codes and "TOTAL_NOT_IN_SOURCE" in codes


def test_total_absent_from_source_but_arithmetically_sound_is_only_flagged():
    """Rounded up by 40 paise: the arithmetic still ties, so this is a flag for a
    human rather than a rejection."""
    d = clean_payload()
    d["total"] = str(Decimal(d["total"]) + Decimal("0.40"))
    rep = validate(to_invoice(d), RAW)
    assert rep.ok
    assert "TOTAL_NOT_IN_SOURCE" in {f.code for f in rep.warnings}


def test_confidence_drops_on_the_field_that_failed():
    d = clean_payload()
    d["total"] = "99999.00"
    rep = validate(to_invoice(d), RAW)
    assert rep.field_confidence["total"] < 0.5
    assert rep.corrected_total is not None


# ---- three-way match --------------------------------------------------
def test_clean_three_way_match_passes():
    out = three_way_match(clean_invoice(), po(), grn())
    assert out.matched and out.matched_lines == 3


def test_invoice_without_a_po_is_blocked():
    out = three_way_match(clean_invoice(), None, None)
    assert not out.matched
    assert "NO_PURCHASE_ORDER" in {f.code for f in out.blocking}


def test_price_above_tolerance_is_blocked():
    out = three_way_match(clean_invoice(), po(rate1="1000.00"), grn())
    assert "PRICE_VARIANCE" in {f.code for f in out.blocking}


def test_price_within_tolerance_passes():
    out = three_way_match(clean_invoice(), po(rate1="1235.00"), grn(),
                          Tolerance(price_pct=Decimal("2")))
    assert "PRICE_VARIANCE" not in {f.code for f in out.blocking}


def test_billing_for_more_than_was_received_is_blocked():
    out = three_way_match(clean_invoice(), po(), grn(q1="30"))
    assert "BILLED_ABOVE_RECEIPT" in {f.code for f in out.blocking}


def test_over_billing_against_the_po_is_blocked():
    out = three_way_match(clean_invoice(), po(qty1="20"), grn())
    assert "OVER_BILLED_VS_PO" in {f.code for f in out.blocking}


def test_wrong_vendor_is_blocked():
    p = po()
    p.vendor_name = "Someone Else Pvt Ltd"
    out = three_way_match(clean_invoice(), p, grn())
    assert "VENDOR_MISMATCH" in {f.code for f in out.blocking}


def test_duplicate_invoice_is_blocked():
    inv = clean_invoice()
    seen = {(inv.vendor_name.strip().lower(), inv.invoice_no.strip().lower())}
    out = three_way_match(inv, po(), grn(), seen_invoice_keys=seen)
    assert "DUPLICATE_INVOICE" in {f.code for f in out.blocking}


def test_missing_grn_is_a_warning_not_a_block():
    """Two-way matching is a legitimate state -- the invoice arrived before the
    receipt was booked. It should be flagged, not rejected."""
    out = three_way_match(clean_invoice(), po(), None)
    codes = {f.code for f in out.findings}
    assert "NO_GOODS_RECEIPT" in codes
    assert not any(f.code == "NO_GOODS_RECEIPT" and f.severity == Severity.ERROR
                   for f in out.findings)


def test_item_not_on_the_po_is_blocked():
    d = clean_payload()
    d["line_items"].append({"description": "Consulting Services", "hsn_sac": None,
                            "quantity": "1", "unit": None, "rate": "25000.00",
                            "line_total": "25000.00"})
    d["subtotal"] = str(Decimal(d["subtotal"]) + Decimal("25000.00"))
    out = three_way_match(to_invoice(d), po(), grn())
    assert "LINE_NOT_ON_PO" in {f.code for f in out.blocking}


# ---- benchmark --------------------------------------------------------
def test_corruption_benchmark_catches_everything():
    from ledgermatch.bench import run_benchmark
    r = run_benchmark(clean_payload(), RAW)
    assert r["clean_invoice_passes"]
    assert r["catch_rate_pct"] == 100.0, r["detail"]
