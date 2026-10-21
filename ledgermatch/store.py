"""Persistence and the PO/GRN master data.

Kept deliberately small -- SQLite, plain SQL, no ORM. The interesting logic in
this project is in validate.py and threeway.py, and burying it under a data
access layer would not make it better.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .schema import ExtractedInvoice, ValidationReport
from .threeway import GoodsReceipt, GRNLine, MatchOutcome, POLine, PurchaseOrder

SCHEMA = """
CREATE TABLE IF NOT EXISTS invoices (
    invoice_id TEXT PRIMARY KEY,
    created_utc TEXT NOT NULL,
    filename TEXT,
    vendor_name TEXT,
    invoice_no TEXT,
    po_number TEXT,
    total TEXT,
    status TEXT NOT NULL,
    extracted_json TEXT,
    validation_json TEXT,
    threeway_json TEXT,
    raw_text TEXT
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id TEXT NOT NULL,
    ts_utc TEXT NOT NULL,
    actor TEXT NOT NULL,
    note TEXT
);
"""

# ---- seed master data -------------------------------------------------
SEED_POS = [
    PurchaseOrder(
        po_number="PO-2026-0311", vendor_name="Meridian Scientific Supplies Pvt Ltd",
        vendor_gstin="29ABCDE1234F1Z5", currency="INR",
        lines=[
            POLine(1, "MB-LP-40", "Microtome Blades Low Profile", Decimal("40"), Decimal("1250.00")),
            POLine(2, "SC-WH-200", "Slide Cassettes White", Decimal("200"), Decimal("145.50")),
            POLine(3, "XY-RG-5L", "Xylene Reagent Grade 5L", Decimal("12"), Decimal("880.00")),
        ]),
]
SEED_GRNS = [
    GoodsReceipt("GRN-2026-0455", "PO-2026-0311", [
        GRNLine(1, "MB-LP-40", Decimal("40")),
        GRNLine(2, "SC-WH-200", Decimal("200")),
        GRNLine(3, "XY-RG-5L", Decimal("12")),
    ]),
]


class Store:
    def __init__(self, db_path: str | Path = "out/ledgermatch.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._pos = {p.po_number: p for p in SEED_POS}
        self._grns = {g.po_number: g for g in SEED_GRNS}

    # ---- master data --------------------------------------------------
    def get_po(self, po_number: str | None) -> PurchaseOrder | None:
        return self._pos.get((po_number or "").strip().upper())

    def get_grn(self, po_number: str | None) -> GoodsReceipt | None:
        return self._grns.get((po_number or "").strip().upper())

    def add_po(self, po: PurchaseOrder) -> None:
        self._pos[po.po_number] = po

    def add_grn(self, grn: GoodsReceipt) -> None:
        self._grns[grn.po_number] = grn

    # ---- invoices -----------------------------------------------------
    @staticmethod
    def status_for(report: ValidationReport, outcome: MatchOutcome) -> str:
        if report.errors or outcome.blocking:
            return "BLOCKED"
        if report.warnings or outcome.findings:
            return "REVIEW"
        return "READY"

    def save(self, inv: ExtractedInvoice, report: ValidationReport,
             outcome: MatchOutcome, raw_text: str, filename: str | None) -> str:
        inv_id = f"INV-{uuid.uuid4().hex[:10]}"
        self.conn.execute(
            "INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (inv_id, datetime.now(timezone.utc).isoformat(), filename,
             inv.vendor_name, inv.invoice_no, inv.po_number, str(inv.total),
             self.status_for(report, outcome),
             inv.model_dump_json(),
             report.model_dump_json(),
             json.dumps({"matched": outcome.matched,
                         "findings": [f.model_dump(mode="json") for f in outcome.findings]}),
             raw_text))
        self.conn.commit()
        return inv_id

    def save_rejected(self, raw_text: str, error: str, filename: str | None) -> str:
        inv_id = f"INV-{uuid.uuid4().hex[:10]}"
        self.conn.execute(
            "INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (inv_id, datetime.now(timezone.utc).isoformat(), filename, None, None,
             None, None, "REJECTED", None, json.dumps({"error": error}), None, raw_text))
        self.conn.commit()
        return inv_id

    def get(self, invoice_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM invoices WHERE invoice_id=?",
                              (invoice_id,)).fetchone()
        return dict(r) if r else None

    def list_by_status(self, status: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT invoice_id, vendor_name, invoice_no, po_number, total, status"
            " FROM invoices WHERE status=? ORDER BY created_utc DESC", (status,)).fetchall()
        return [dict(r) for r in rows]

    def seen_invoice_keys(self) -> set[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT vendor_name, invoice_no FROM invoices"
            " WHERE vendor_name IS NOT NULL AND status != 'REJECTED'").fetchall()
        return {(r["vendor_name"].strip().lower(), r["invoice_no"].strip().lower())
                for r in rows if r["invoice_no"]}

    def approve(self, invoice_id: str, actor: str, note: str) -> None:
        self.conn.execute(
            "INSERT INTO approvals (invoice_id, ts_utc, actor, note) VALUES (?,?,?,?)",
            (invoice_id, datetime.now(timezone.utc).isoformat(), actor, note))
        self.conn.execute("UPDATE invoices SET status='POSTED' WHERE invoice_id=?",
                          (invoice_id,))
        self.conn.commit()
