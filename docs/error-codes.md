# Input-validation error codes

All input-stack prechecks in `disp_s1.main` and `disp_s1.pge_runconfig` raise
`disp_s1._log.InputValidationError` — a `ValueError` subclass carrying a
numeric `.error_code` so the PGE can map failures to a specific cause. These
checks run *before* the expensive `dolphin` workflow starts, so a bad input
list fails in seconds instead of after minutes-to-hours of processing.

See [compressed-slc-operations.md](compressed-slc-operations.md) for the
mechanism behind the compressed-SLC-related checks (1001, 1002, 2001).

## Summary table

| Code | Function | Applies to | Condition |
|---|---|---|---|
| 1000 | `_assert_no_large_temporal_gaps` (`main.py`) | both modes | consecutive real SLC dates in a burst are more than `max_gap_days` (default 2 years) apart |
| 1001 | `_assert_no_compressed_slc_conflicts` (`main.py`) | both modes, whenever a CCSLC is present | a real SLC's date exactly equals its burst's most-recent CCSLC's reference date |
| 1002 | `_assert_no_compressed_slc_conflicts` (`main.py`) | both modes, whenever a CCSLC is present — boundary is mode-dependent | a CCSLC's reference date is later than allowed for its burst (see below) |
| 2000 | `_assert_forward_mode_compressed` (`main.py`) | forward mode only | no CCSLC present at all, or a burst with real SLCs has no CCSLC of its own |
| 2001 | `_create_forward_mode_network` (`pge_runconfig.py`) | forward mode only | fewer than `nearest_n + 1` **real** CSLCs in a burst (compressed SLCs don't count toward this depth) |

Numbering convention: **1xxx** = checks that apply regardless of
`product_type`; **2xxx** = forward-mode-only checks.

## 1000 — temporal gap

```
Temporal gap(s) exceeding the {N}-year limit for input SLCs:
burst {burst_id}: {gap}-day ({gap:.2f}-year) gap between {earlier} and {later}
```

(The limit is rendered in days rather than years when `max_gap_days` is set
below one year, so a short threshold doesn't report itself as a "0-year limit".)

A multi-year gap between consecutive real SLC dates in the same burst
destroys interferometric coherence and almost always indicates a malformed
input list (e.g. missing files, wrong burst/frame). Compressed SLCs are
excluded from this check — only real-SLC-to-real-SLC gaps are measured.

**Fix**: supply the missing intermediate real SLC(s), or confirm the gap is
intentional and raise `max_gap_days` when calling `disp_s1.main.run`.

## 1001 — real SLC overlaps its CCSLC's reference date

```
Input CSLC list has real SLC(s) sharing a date with their own burst's
compressed SLC reference date. The compressed SLC already represents that
epoch, so no real SLC for the same date should also be included:
burst {burst_id}: real SLC date {date} overlaps with its most recent
compressed SLC's reference date
```

The compressed SLC's reference date *is* that epoch already — a real SLC
for the same date is ambiguous with it. Uncaught, this crashes deep inside
`dolphin`'s ministack construction with a `pydantic.ValidationError`
(`dolphin.stack.Ministack` in forward mode; `MiniStackPlanner` in
historical/sequential mode — confirmed by direct reproduction against
`delivery_data_official`'s official archived CCSLC, ref=2017-05-24, paired
with a real SLC for the same date). Historical mode burns real processing
time (mask extraction, PS pixels, EMI setup — observed ~90s) before hitting
it, since ministack construction happens mid-workflow rather than at the
CLI entry point; forward mode fails immediately.

**Known gap**: only checked against the CCSLC's exact reference date, not
its full `[start, end]` coverage range. A real SLC dated strictly *inside*
that range (not equal to the reference date) is not currently caught — see
the operational-cadence note in
[compressed-slc-operations.md](compressed-slc-operations.md).

**Fix**: drop the redundant real SLC for that date, or don't include a CCSLC
that already covers it.

## 1002 — CCSLC reference date later than allowed

Forward mode:

```
Compressed SLC reference date is later than the latest real SLC in the same
burst (the latest acquisition must always be real):
burst {burst_id}: compressed SLC reference date {ref} is later than
the latest real SLC date {boundary}
```

Historical mode:

```
Compressed SLC reference date is later than the earliest real SLC in the same
burst (historical mode outputs a product for every real date, so compressed
SLCs must cover only prior history):
burst {burst_id}: compressed SLC reference date {ref} is later than
the earliest real SLC date {boundary}
```

A compressed SLC must summarize *prior* history — it should never reach
into (or past) the real dates it's paired with in the same run. The
boundary depends on `product_type`, because the two modes produce different
numbers of output products (see
[compressed-slc-operations.md](compressed-slc-operations.md) for why):

- **`DISP_S1_FORWARD`**: the CCSLC's reference date must predate the
  **latest** real SLC date in its burst. Forward mode only ever outputs one
  product (for the latest date), so earlier real dates in the same batch
  are just context.
- **Any other `product_type`** (historical, including the "catch-up"
  scenario where a CCSLC leads a batch of new real dates): the CCSLC's
  reference date must predate the **earliest** real SLC date in its burst.
  Historical mode outputs one product per new real date, so if the CCSLC's
  reference fell between two new dates, the earlier one would effectively
  be double-counted — once as real data, once as part of the CCSLC's
  claimed history.

Uncaught, this reproduces the same crash class as 1001 — confirmed by two
separate cases: the abandoned S3-CCSLC dead end (official CCSLC ref
2017-05-24 with no real SLC past 2017-04-30 in `delivery_data_small`), and
a self-inflicted reproduction using a compressed SLC referenced at the
latest available date.

**Fix**: use a CCSLC referenced earlier in time, or (for historical) confirm
every new real date genuinely postdates the CCSLC's coverage — not just the
newest one.

## 2000 — forward mode requires CCSLC coverage

Two related messages, same code:

```
Forward mode requires at least one compressed SLC (CCSLC) in the input CSLC
list, but none was found.
```
```
Forward mode requires a compressed SLC (CCSLC) for every burst in the input
CSLC list, but no CCSLC was found for burst(s): {burst_ids}.
```

Forward mode is incremental — it must be handed at least one CCSLC to
process against, and since each burst is processed independently, a burst
with real SLCs but no CCSLC of its own would silently produce an empty
compressed-SLC reader for that burst downstream (observed: `IndexError` in
`disp_s1._ps.run_combine`). This check does **not** apply to historical
mode — running with zero CCSLCs is historical mode's normal/default case.

**Fix**: supply at least one CCSLC per burst that has real SLCs in the input
list.

## 2001 — insufficient stack depth for the forward-mode network

```
Forward mode nearest-{N} network requires at least {N+1} real CSLCs
(compressed SLCs don't count toward this depth) in the input stack, but
only {count} were found.
```

Forward mode's manual-index network (nearest-3 or nearest-4, from
`forward_mode_network_size`) reaches back `nearest_n + 1` positions from the
latest date — but only into `dolphin`'s **real-date-only** phase-linked
list (compressed SLCs are globbed separately and never enter it; see
[compressed-slc-operations.md](compressed-slc-operations.md)). So the
required depth is counted in real SLCs alone, regardless of how many
compressed SLCs (0–5) are also present.

This check previously counted *all* CSLCs (real + compressed) toward the
depth, undercounting by the number of compressed SLCs present. That bug
produced an `IndexError` in `dolphin.interferogram._make_ifg_pairs` instead
of failing at input validation — reproduced directly against
`delivery_data_official` (1 CCSLC + 4 real dates passed the old check but
crashed; nearest-4 actually needs 5 *real* dates). Fixed to count only real
SLCs.

**Fix**: include more real trailing dates (`nearest_n + 1` per burst,
compressed SLCs don't count), or lower `forward_mode_network_size` (valid
range: 3–4 only — schema-constrained, `Field(3, ge=3, le=4)`).
