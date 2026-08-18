# Compressed CSLCs: operational logic

This page explains how compressed CSLCs (CCSLCs) are used in forward and
historical processing, and why the two modes have different requirements.
It reflects the behavior of `dolphin` (phase linking, ministack planning,
interferogram formation) and `disp_s1` (input validation, output
re-referencing).

## Why compressed CSLCs exist

A compressed CSLC summarizes a prior "ministack" of real CSLCs as a single
phase-linked reference. Processing a new acquisition against one CCSLC is
far cheaper than re-processing it against every prior real SLC individually
— that's the entire point of the sequential/StBAS algorithm this pipeline
implements.

Under `compressed_slc_plan = LAST_PER_MINISTACK`, a CCSLC's **reference date
always equals its end date**: `MiniStackPlanner.plan` sets
`compressed_reference_idx = -1`, i.e. the last (most recent) date in the
ministack it summarizes. This identity (`ref == end`) is assumed throughout
the input checks described in [error-codes.md](error-codes.md). See
"Filename conventions" below for exactly how `ref`/`start`/`end` appear in a
filename.

`last_per_ministack` is the disp-s1 default: every delivered
`configs/algorithm_parameters_*.yaml`, forward and historical alike, sets it,
and the PGE always runs from one of those. The only way to end up on a
different plan is to build `AlgorithmParameters()` in Python without a
delivered config — `AlgorithmParameters.phase_linking` is `dolphin`'s
`PhaseLinkingOptions` unchanged, whose own pydantic default is
`always_first`. Under `always_first`,
`compressed_reference_idx = max(0, num_ccslc - 1)` puts `ref` at or before
the start of the covered range and `ref == end` no longer holds, so the
checks below would rest on a false premise. Worth knowing when reading
`dolphin` in isolation, or when writing tests that construct the parameters
directly (several in `tests/test_pge_runconfig.py` do).

## Filename conventions

All date/burst parsing in this codebase (`opera_utils.get_dates`,
`group_by_burst`) works on the filename stem alone — no file I/O, no
dependence on which of the conventions below produced it. The `is_compressed`
test used throughout (`"compressed" in str(f).lower()`) matches both CCSLC
conventions: the self-generated form's literal `compressed_` prefix, and the
official form's `COMPRESSED-CSLC` substring.

### Real CSLC (input - the official OPERA format)

```
OPERA_L2_CSLC-S1_{burst_id}_{acquisition_datetime}Z_{processing_datetime}Z_{sensor}_{pol}_v{version}.h5
```

Example: `OPERA_L2_CSLC-S1_T027-056725-IW1_20170605T132754Z_20240626T090220Z_S1A_VV_v1.1.h5`

| Field | Example | Meaning |
|---|---|---|
| `burst_id` | `T027-056725-IW1` | relative orbit (track) – burst – subswath; hyphen-separated, uppercase |
| `acquisition_datetime` | `20170605T132754Z` | Sentinel-1 acquisition start time (UTC) |
| `processing_datetime` | `20240626T090220Z` | when this CSLC product was generated |
| `sensor` | `S1A` | S1A / S1B / S1C / S1D |
| `pol` | `VV` | polarization |
| `version` | `v1.1` | product version |

`get_dates()` finds **2** matches (acquisition + processing datetimes);
`[0]` — the acquisition date — is always what disp_s1 treats as "the date"
for this file. There is no separate "locally generated" convention for real
CSLCs: disp_s1/dolphin only ever consume them as delivered, never rename or
regenerate them.

### Compressed CSLC (CCSLC)
(e.g. `s3://opera-ops-lts-pop1/products/CSLC_S1_COMPRESSED/...`):

```
OPERA_L2_COMPRESSED-CSLC-S1_{frame_id}_{burst_id}_{ref_date}Z_{start_date}Z_{end_date}Z_{processing_datetime}Z_{pol}_v{version}.h5
```

Example: `OPERA_L2_COMPRESSED-CSLC-S1_F07091_T027-056725-IW1_20170524T000000Z_20160716T000000Z_20170524T000000Z_20250412T025708Z_VV_v1.0.h5`

| Field | Example | Meaning |
|---|---|---|
| `frame_id` | `F07091` | OPERA frame identifier |
| `burst_id` | `T027-056725-IW1` | hyphen-separated, uppercase (matches the real-CSLC convention) |
| `ref_date` | `20170524T000000Z` | reference/output date, full datetime at midnight UTC (== `end_date`) |
| `start_date` | `20160716T000000Z` | earliest date summarized |
| `end_date` | `20170524T000000Z` | latest date summarized (== `ref_date`) |
| `processing_datetime` | `20250412T025708Z` | when this compressed product was generated |
| `pol` | `VV` | polarization |
| `version` | `v1.0` | product version |

Same three-date semantics as the self-generated form, just with full
datetimes instead of bare `YYYYMMDD`, plus frame ID / processing time /
polarization / version fields the self-generated form omits. `get_dates()`
finds **4** matches (ref, start, end, processing); `[:3]` still correctly
picks out `(ref, start, end)` since `processing_datetime` sorts last in the
filename.

## Phase-linking's reference is always the most recent CCSLC (if any)

`disp_s1.pge_runconfig._compute_reference_dates` sets
`phase_linking.output_reference_idx`. With the default `LAST_PER_MINISTACK`
plan:

```python
if compressed_slc_plan == CompressedSlcPlan.LAST_PER_MINISTACK:
    output_reference_idx = max(0, num_ccslc - 1)
```

This is **identical logic for forward and historical mode** — it does not
branch on `product_type`. `output_reference_idx` points at the index of the
*most recent* compressed SLC in the burst's date-sorted file list,
regardless of how many real dates trail behind it (1, or 15, or more). If
`num_ccslc == 0` (historical mode's default case — no CCSLC supplied),
`output_reference_idx` falls back to `0`, i.e. the earliest real SLC in the
list — the classic single-master historical reference.

**Practical consequence**: forward mode's own precheck
(`_assert_forward_mode_compressed`, error 2000) requires at least one CCSLC
per burst, so in forward mode the phase-linking base is *always* a
compressed SLC, never a real one. Historical mode has no such requirement,
so its base is conditional on whether a CCSLC was supplied.

## Two interferogram groups get formed per run

`dolphin.workflows.wrapped_phase.create_ifgs` builds two distinct sets of
interferograms whenever compressed SLCs are present (comment from that
function, using its own example — `compressed_1_2_3, slc_4, slc_5, slc_6`,
with the CCSLC referenced to day "1"):

1. **Single-reference ifgs**: every new real date's phase-linked output is
   directly conjugated against the CCSLC's reference date —
   `(1,4), (1,5), (1,6)`. The CCSLC contributes to this group. All of them
   are formed, but how many are *kept* is mode-dependent — see below.
2. **Short-baseline network among the new real dates only**: a second set of
   pairs formed *only* among the new real dates —
   `(4,5), (4,6), (5,6)` — using whichever network style is configured.
   **The CCSLC never enters this group.** `dolphin` builds this list by
   globbing only the real-date phase-linked outputs (`pl_path.glob("2*.tif")`),
   explicitly excluding compressed-SLC outputs
   (`pl_path.glob("compressed*.tif")`).

The network style for group 2 differs by mode:

- **Forward mode**: fixed manual indices (nearest-N, N ∈ {3, 4}), e.g. for
  nearest-4: `(-2,-1), (-3,-1), (-4,-1), (-3,-2), (-4,-2), (-4,-3), (-5,-1),
  (-5,-2), (-5,-3), (-5,-4)`. The deepest index (`-5` for nearest-4) is
  applied to the **real-only** list from group 2 — so it requires
  `nearest_n + 1` real dates, *not counting the CCSLC*. Undercounting this
  (e.g. by including the CCSLC in the depth count) produces an `IndexError`
  deep in `dolphin.interferogram._make_ifg_pairs` instead of failing at
  input validation,
  `_create_forward_mode_network` (error 2001; see
  [error-codes.md](error-codes.md)) accounts for this.
- **Historical mode**: bandwidth-limited (`interferogram_network.max_bandwidth`
  — 3 in `algorithm_parameters_historical_*.yaml`), which caps the *maximum*
  pair separation rather than requiring a fixed depth. With fewer real dates
  than the bandwidth, historical just forms fewer/narrower pairs — it
  degrades gracefully rather than indexing off the end of a list. Historical
  has no equivalent of the 2001 stack-depth check for this reason.

**Which group-1 ifgs are kept.** `create_ifgs` takes the *whole*
single-reference set only when `interferogram_network.reference_idx == 0`,
which operationally it never is — the yaml leaves `reference_idx` null, and
forward mode's network comes from `_create_forward_mode_network`, which
returns a bare `InterferogramNetwork(indexes=...)`. (Both `reference_idx` and
`max_bandwidth` default to `None` there, so the forward yaml's
`max_bandwidth: 4` is discarded along with the rest of that block — it is not
an active setting in forward mode.) What each mode keeps:

- **Historical**: the `max_bandwidth` branch takes `single_ref_ifgs[:max_b]`
  and appends the real-date `Network` pairs — the "same as though we had
  normal SLCs (1, 4, 5, 6)" construction described in that function's
  comment. The first `max_bandwidth` CCSLC-referenced ifgs are kept.
- **Forward**: the `indexes` branch *assigns* `ifg_file_list` from the
  manual-index `Network` alone, and neither other branch runs, so no
  CCSLC-referenced ifg is kept. This is by design, not an oversight: the
  forward product comes from the nearest-N network among real dates,
  inverted and then re-referenced to the second-to-last date (see below), so
  it never needs a pair against the CCSLC. The CCSLC's role in forward mode
  is to be the phase-linking base, not a network node. (The `TODO` about
  `(0, X)` indexes in `create_ifgs` doesn't apply — the manual indices are
  all negative.)

Anything formed but not requested is deleted at the end of `create_ifgs`
(`for p in written_ifgs - requested_ifgs: p.unlink()`). That costs nothing
measurable: `convert_pl_to_ifg` writes a small `.int.vrt` that conjugates
lazily, so the discarded ifgs are XML stubs, never computed pixels.

## Output re-referencing: a separate concept

Don't conflate the phase-linking reference (above) with `main.py`'s
**output** re-reference step, which happens later, after inversion:

```python
*_, (second_to_last_date,), (_last_date,) = datetimes_present
...
if pge_runconfig.primary_executable.product_type == "DISP_S1_FORWARD":
    final_ts_paths, final_residual_paths = _redo_reference(
        ..., second_to_last_date, ...
    )
```

In forward mode, this re-references the already-inverted time series to the
second-to-last date in the batch, for output packaging. It's independent of
whether a CCSLC is involved at all, and independent of which file
phase-linking used as its base.

The `_redo_reference` call is **forward-only**: there is no historical branch
and no `else`, so a historical run packages the time series with whatever
reference the inversion produced. That follows from the product counts in the
next section — forward emits one product, for the newest date, so pinning it
to the second-to-last date makes that product a nearest-neighbor pair;
historical emits one product per new real date, and collapsing them all onto
a single date inside the batch would be meaningless.

Two adjacent details are *not* forward-gated, despite sitting next to code
that is:

- `second_to_last_date` is unpacked from `datetimes_present`
  unconditionally, before any product-type check. Only its two *uses* are
  gated (the `last_processed` consistency check, and `_redo_reference`), so
  the unpack's implicit "at least two distinct dates in the batch"
  requirement applies in historical mode too.
- A second, database-driven output re-reference runs in both modes:
  `_compute_reference_dates` always sets
  `output_options.extra_reference_date` from the reference date database
  json, and `dolphin` shifts every pair whose secondary falls after that
  date (`wrapped_phase.py`, then `timeseries.py`). So "re-reference the
  output" is not uniquely forward — only the *second-to-last-date* one is.

## Why forward's and historical's date-boundary checks differ

Because of the two interferogram groups above, and because of how many
output products each mode produces:

- **Forward mode** produces exactly **one** output product per run — for the
  single latest date in the batch. So it only needs the CCSLC to predate
  that one date; earlier real dates in a forward batch (if any) are only
  ever context, never their own output.
- **Historical mode** produces **one output product per new real date**. If
  the CCSLC's reference date fell after the *earliest* new real date but
  before the *latest*, that earliest date would be claimed twice — once as
  real data, once as (implicitly) part of the CCSLC's summarized history.
  So historical requires the CCSLC to predate *all* new real dates, not just
  the latest.

This asymmetry is implemented in `_assert_no_compressed_slc_conflicts`'s
`product_type` parameter (error 1002) — see
[error-codes.md](error-codes.md) for the exact boundary logic.

## Operational cadence (reference implementation)

`scripts/run_disp.py`'s `run_once_forward` implements one concrete
operational pattern (not the only possible one, but illustrative of how the
"defer, then compress" cadence works):

```python
save_compressed_now = (run_idx + 1) % ms_size == 0
```

With the script's default `ms_size = 15`: a new compressed SLC only gets
saved every 15th forward run. In between, up to 15 real CSLCs accumulate
uncompressed in front of the existing CCSLC(s) — the compressed SLC does
*not* move to stay pinned at "second-to-last" every round; instead, the
distance between it and the newest date grows by one each round until the
next compression event resets it. 

**Does the sliding window re-feed already-compressed dates?** For the real
CSLCs, yes: `get_forward_batch` selects by plain index position
(`cslc_files[run_idx : run_idx + ms_size + 1]`) and slides by one per run, so
consecutive runs share all but one real date, and after a compression event
the window keeps re-including real dates the new CCSLC already summarizes.
What keeps that from producing an invalid input list is the filter in
`latest_k_per_burst`:

```python
valid_files = [f for f in files if opera_utils.get_dates(f)[0] < first_real_date]
```

For a compressed file `get_dates(...)[0]` is the *reference* date, and with
the default `LAST_PER_MINISTACK` plan `ref == end`. So every CCSLC that
survives the filter has its whole `[start, end]` range strictly earlier than
every real date in the batch — a real CSLC can never land inside a supplied
CCSLC's covered range. (The `TODO: decide if we wanna watch for overlap in
time...` comment just above that call is stale; the filter is the watch. The
filter compares against the global earliest real date rather than a per-burst
one, which is the conservative direction.)

Two caveats:

- The guarantee rests entirely on `ref == end`, i.e. on the algorithm
  parameters keeping disp-s1's `compressed_slc_plan: last_per_ministack`.
  Under `always_first`, `ref` sits at or before the
  start of the covered range, so a CCSLC could pass the
  `ref < first_real_date` filter while its `end` still reaches into the
  batch's real dates. Neither error 1001 (exact equality to the reference
  date) nor error 1002 (boundary comparison against the reference date)
  would catch that, since both look only at `ref`. That residual gap is in
  the prechecks, not in this script's operational path. (`first_per_ministack`
  is accepted by the enum but has no branch in `MiniStackPlanner.plan`'s
  `compressed_reference_idx` chain, so it fails with an `UnboundLocalError`
  rather than producing anything.)
- The filter is strict (`<`), so the CCSLC produced by run `k` — whose
  reference date is that batch's *last* real date, `ms_size` indices ahead of
  the window start — is excluded from every subsequent run until the window
  start passes it, which includes the next compression run. The script
  therefore depends on older CCSLCs already present in `bulk_compressed_dir`
  (e.g. seeded from historical runs). If that directory only ever holds this
  chain's own output, a forward run can end up with zero CCSLCs for a burst
  and trip error 2000.
