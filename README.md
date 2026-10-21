# LedgerMatch

AI invoice capture with a deterministic validation layer, plus PO/GRN/invoice three-way matching.

The model reads the document. It does not do arithmetic, and it does not get the last word on any
number that will hit the ledger.

That constraint is the entire project. Handing an LLM a scanned invoice and posting whatever JSON
comes back is a demo, not a control. The failure mode that matters is not a crash or a refusal —
it is a **fluent, well-formed, confident, wrong number**, which no amount of model confidence will
flag and no schema check will catch.

**Measured:** the validation layer blocks **7 of 7** realistic extraction corruptions — transposed
digits, dropped leading digits, OCR 5/6 confusion, a fabricated total, tax lifted from the wrong
row, CGST/SGST asymmetry, a misread quantity. Every one of those produces valid JSON that passes
schema validation. **24 tests** cover the validator, the matcher and the API.

---

## How the numbers are checked

Every quantitative relationship on an invoice is redundant, and an LLM that misreads a digit will
usually break one identity without breaking the others. So each one is recomputed in Python:

```
qty x rate           == line_total          per line
sum(line_totals)     == subtotal
subtotal x tax_rate  == tax_amount          per tax line
subtotal + sum(tax)  == grand_total
CGST                 == SGST                on an intra-state supply
IGST                 XOR (CGST + SGST)      a supply is inter-state or intra-state, never both
```

Where the recomputed figure disagrees with the claimed one, **the recomputed figure wins** and the
claim is recorded as a finding. A rupee of per-line slack absorbs genuine printing rounding; wider
than that is a real disagreement.

### Two independent checks, not one

Arithmetic cannot see a coherently fabricated total. So there is a second, independent check: does
the claimed total actually appear anywhere in the OCR text? (Comma grouping and Indian lakh/crore
grouping both handled.)

Severity is conditional, on purpose:

| Arithmetic | In source text | Outcome |
|---|---|---|
| ties | found | clean |
| ties | absent | **warning** — flag for review, don't reject over a `Rs. 1,05,798/-` formatting quirk |
| fails | found | **blocked** |
| fails | absent | **blocked** — no support from any direction |

Neither check is reliable alone. Requiring both to fail before hard-blocking is what keeps the tool
from being so noisy that people start clicking through it.

### The schema is a door, not a suggestion

`schema.py` is the only way an extraction enters the system. Strict Pydantic: `extra="forbid"`,
money as `Decimal` never `float`, quantities `gt=0`, GSTIN shape-checked, dates real dates. A
`ValidationError` here is a **rejection**, not a warning — an invoice that cannot be represented
does not get a degraded posting, it goes to a human.

This caught a bug in this repo's own regex extractor during development: an unanchored `Total`
pattern matched the `Sub-total` line first and silently under-stated an invoice by the entire tax
amount. The door stopped it. That is the mechanism working on real code, not a hypothetical.

---

## Three-way match

Validation proves an invoice is internally consistent. It says nothing about whether we ordered the
goods or received them — a perfectly-formed invoice for 400 cassettes nobody ordered passes every
arithmetic check above.

```
PO       what we agreed to buy, and at what price
GRN      what actually arrived
Invoice  what we are being asked to pay for
```

| Finding | Severity | Catches |
|---|---|---|
| `DUPLICATE_INVOICE` | error | the same invoice arriving by email and again on paper |
| `NO_PURCHASE_ORDER` | error | unauthorised spend |
| `PRICE_VARIANCE` | error | billed above the agreed rate, beyond tolerance |
| `OVER_BILLED_VS_PO` | error | invoiced more units than ordered |
| `BILLED_ABOVE_RECEIPT` | error | billed for goods never received |
| `SHORT_BILLED` | warning | received more than invoiced; another invoice is coming |
| `VENDOR_MISMATCH` / `GSTIN_MISMATCH` | error | payment redirection |
| `LINE_NOT_ON_PO` | error | items added to an approved order |
| `NO_GOODS_RECEIPT` | warning | two-way matched only; legitimate, but flagged |

Duplicates are keyed on **vendor + invoice number**, not the number alone — numbering restarts every
year and collides across vendors. Tolerances are per-line and configurable, never a blanket
"close enough".

## Control design in the API

There is deliberately no endpoint that posts an extraction to the ledger. Extraction produces a
*proposal*; the proposal must survive validation and the three-way match; anything with a blocking
finding requires an explicit approval recorded against a named actor. The API shape enforces the
control rather than documenting it.

```
POST /invoices                    OCR text in, validated proposal out. Posts nothing.
GET  /invoices/{id}               extraction + full validation report
GET  /exceptions?status=BLOCKED   the review queue
POST /invoices/{id}/approve       named override, logged
GET  /health
```

Statuses: `READY` (clean) · `REVIEW` (warnings only) · `BLOCKED` (errors) · `REJECTED` (failed the
schema door) · `POSTED` (approved).

## Extractors are swappable

`Extractor` is a Protocol with two implementations. `LLMExtractor` prompts for JSON only at
temperature 0 with the schema inline. `RegexExtractor` is a plain parser for the sample layout —
which means **the validator, matcher and API run and benchmark with no API key and no network**, and
it doubles as a baseline. If the LLM cannot beat a regex on a structured layout, it is not earning
its latency or its cost.

Everything downstream of `ExtractedInvoice` treats its input identically regardless of origin.

---

## Running it

```bash
pip install -r requirements.txt

python -m ledgermatch.bench                    # corruption benchmark: 7/7 caught
python -m pytest tests/ -q                     # 24 tests
uvicorn ledgermatch.api:app --reload           # API on :8000, docs at /docs
```

For real OCR, install `pytesseract` + Tesseract; for the LLM path, set `ANTHROPIC_API_KEY`. Neither
is needed to run the benchmark or the tests — `ocr()` falls back to a sidecar `.txt`.

## Layout

```
ledgermatch/
  schema.py     strict Pydantic contract -- the only door in
  validate.py   recomputes every number; conditional-severity source cross-check
  threeway.py   PO / GRN / invoice matching with per-line tolerances
  extract.py    OCR + LLM extractor, plus a deterministic regex baseline
  bench.py      corruption harness -- does the reliability layer catch anything?
  store.py      SQLite persistence and PO/GRN master data
  api.py        FastAPI surface
tests/          24 tests
samples/        sample invoice
```

## Known limits

- The regex extractor handles one invoice layout. It is a baseline and a test fixture, not a
  general parser — real coverage is the LLM path.
- GSTIN validation is a **shape** check. Confirming a registration is live needs the GST portal.
- Tax logic covers CGST/SGST/IGST/UTGST/CESS on standard slabs. Reverse charge, TCS and
  composition-scheme invoices are not modelled.
- The three-way match keys line items on normalised description. Real deployments should key on a
  vendor SKU, which is stable; descriptions are not.
- No approval-limit hierarchy — a single `approve` endpoint. Real AP needs value-banded authority.
