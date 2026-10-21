"""HTTP surface.

Note what the endpoints do *not* offer: there is no way to post an invoice to
the ledger directly from an extraction. Extraction produces a proposal; the
proposal has to survive validation and the three-way match; and anything with a
blocking finding requires an explicit human approval that is recorded against a
named user. The API shape enforces the control, rather than documenting it.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

from .extract import ExtractionError, RegexExtractor, to_invoice
from .schema import ExtractedInvoice
from .store import Store
from .threeway import Tolerance, three_way_match
from .validate import validate

app = FastAPI(title="LedgerMatch", version="1.0",
              description="AI invoice capture with a deterministic validation layer")
store = Store()


class IngestRequest(BaseModel):
    raw_text: str
    filename: str | None = None


class ApprovalRequest(BaseModel):
    actor: str
    note: str = ""


@app.post("/invoices", status_code=201)
def ingest(req: IngestRequest = Body(...)):
    """OCR text in, validated proposal out. Nothing is posted here."""
    try:
        payload = RegexExtractor().extract(req.raw_text)
        inv = to_invoice(payload)
    except ExtractionError as e:
        # A rejection at the schema boundary is a normal outcome, not a 500.
        inv_id = store.save_rejected(req.raw_text, str(e), req.filename)
        raise HTTPException(422, {"invoice_id": inv_id, "stage": "extraction",
                                  "detail": str(e)})

    report = validate(inv, req.raw_text)
    po = store.get_po(inv.po_number) if inv.po_number else None
    grn = store.get_grn(inv.po_number) if inv.po_number else None
    outcome = three_way_match(inv, po, grn, Tolerance(),
                              seen_invoice_keys=store.seen_invoice_keys())

    inv_id = store.save(inv, report, outcome, req.raw_text, req.filename)
    return {
        "invoice_id": inv_id,
        "status": store.status_for(report, outcome),
        "validation": report.model_dump(mode="json"),
        "three_way": {
            "matched": outcome.matched,
            "matched_lines": outcome.matched_lines,
            "findings": [f.model_dump(mode="json") for f in outcome.findings],
        },
    }


@app.get("/invoices/{invoice_id}")
def get_invoice(invoice_id: str):
    row = store.get(invoice_id)
    if not row:
        raise HTTPException(404, "unknown invoice")
    return row


@app.get("/exceptions")
def exceptions(status: str = "BLOCKED"):
    return {"status": status, "items": store.list_by_status(status)}


@app.post("/invoices/{invoice_id}/approve")
def approve(invoice_id: str, req: ApprovalRequest):
    """Override a blocking finding. Requires a named actor and is logged."""
    row = store.get(invoice_id)
    if not row:
        raise HTTPException(404, "unknown invoice")
    if row["status"] == "POSTED":
        raise HTTPException(409, "already posted")
    store.approve(invoice_id, req.actor, req.note)
    return {"invoice_id": invoice_id, "status": "POSTED", "approved_by": req.actor}


@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}
