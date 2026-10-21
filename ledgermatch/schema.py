"""The contract between the model and the ledger.

Everything an LLM returns is untrusted input. This module is the only door it
can come through, and the door is narrow: strict types, no coercion from junk,
no extra keys, and money as Decimal rather than float.

A parse failure here is a *rejection*, not a warning. An invoice that cannot be
represented in this schema does not get a degraded posting -- it goes to a human.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Severity(str, Enum):
    ERROR = "ERROR"      # blocks posting
    WARNING = "WARNING"  # posts, but flagged for review


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False)

    description: str = Field(min_length=1, max_length=300)
    hsn_sac: str | None = Field(default=None, max_length=12)
    quantity: Decimal = Field(gt=0)
    unit: str | None = Field(default=None, max_length=16)
    rate: Decimal = Field(ge=0)
    line_total: Decimal = Field(ge=0)

    @field_validator("quantity", "rate", "line_total", mode="before")
    @classmethod
    def _no_commas(cls, v):
        if isinstance(v, str):
            return v.replace(",", "").replace("\u20b9", "").strip()
        return v


class TaxLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(pattern=r"^(CGST|SGST|IGST|UTGST|CESS)$")
    rate_pct: Decimal = Field(ge=0, le=50)
    amount: Decimal = Field(ge=0)


class ExtractedInvoice(BaseModel):
    """What the model claims the document says. Not yet believed."""

    model_config = ConfigDict(extra="forbid")

    vendor_name: str = Field(min_length=2, max_length=200)
    vendor_gstin: str | None = Field(default=None, max_length=15)
    invoice_no: str = Field(min_length=1, max_length=64)
    invoice_date: date
    po_number: str | None = Field(default=None, max_length=64)
    currency: str = Field(default="INR", pattern=r"^[A-Z]{3}$")

    line_items: list[LineItem] = Field(min_length=1)
    subtotal: Decimal = Field(ge=0)
    taxes: list[TaxLine] = Field(default_factory=list)
    total: Decimal = Field(ge=0)

    @field_validator("vendor_gstin")
    @classmethod
    def _gstin_shape(cls, v):
        """Shape check only. A structurally valid GSTIN is not a real one --
        registration status needs the GST portal, which is out of scope here."""
        if v is None:
            return v
        v = v.strip().upper()
        if len(v) != 15:
            raise ValueError("GSTIN must be 15 characters")
        if not (v[:2].isdigit() and v[2:12].isalnum()):
            raise ValueError("GSTIN state code / PAN block malformed")
        return v


class Finding(BaseModel):
    """One thing the validator objected to."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Severity
    field: str | None = None
    detail: str
    claimed: str | None = None
    computed: str | None = None


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    findings: list[Finding] = Field(default_factory=list)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    corrected_total: Decimal | None = None

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]
