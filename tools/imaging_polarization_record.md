# Imaging beam-2 HWP polarization optimization -- procedure record (2026-06-28)

> Working record of the 2026-06-28 imaging beam-2 half-waveplate (HWP) polarization sweep, paused for
> hardware intensity stabilization. NOT yet a skill -- keep as a record; promote to an
> `experiment-running` reference runbook later if the procedure proves durable. Tooling lives at
> `pyctrl/tools/hwp_point.py`. All pyctrl, 33x33_feedback9 array.

## Goal / physics
Tune the **imaging beam-2** HWP for **photon-collection efficiency**. The two 399 imaging beams are
`AmpAbsImag` (beam 1) and `Amp399Imag2` (beam 2). Beam 1 was found hardware power-limited earlier the
same day and fixed (see the imaging-recal record); this sweep optimizes beam 2's polarization.

**The discriminating hypothesis:** as the HWP rotates toward the collection optimum,
- the **atom-vs-empty intensity SEPARATION** (`delta-mu = atom_mu - empty_mu`, and per-site `d'`) should
  **INCREASE** -- more scattered photons reach the camera, brighter atom signal; while
- the 399 push-out **spectrum CONTRAST** (survival dip depth) should stay **~CONSTANT** -- the same number
  of photons are *scattered* by the atoms regardless of where the HWP sends them.
So separation up + contrast flat = a pure **collection** gain (not more scattering). That is the signature
we look for.

## Method (per HWP angle)
The user turns the HWP by hand and reports the angle; for each angle:
1. **Scan** -- `Spectrum399Scan` with **beam 2 only**: in `YbScans/Spectrum399Scan.py` set
   `g().Pushout.Blue.Amp1 = 0.0` (beam 1 off), `g().Pushout.Blue.Amp2 = 0.5` (beam 2 on). Freq sweep
   narrowed to `matlab_colon(260, 3, 360)` (34 pts) to sit on the dip. ~100 shots (`--reps 3`) for a
   coarse point, ~400 (`--reps 12`) for a high-rep confirm. `NumImages=2` survival.
   ```
   cd pyctrl && python YbScans/Spectrum399Scan.py --reps 3      # ~100 shots
   cd pyctrl && python YbScans/Spectrum399Scan.py --reps 12     # ~400 shots
   ```
2. **Analyze** -- `pyctrl/tools/hwp_point.py <scan_id> --angle <deg>` (yb_analysis env, from project root)
   computes + saves two PNGs into the scan dir:
   - **single-Lorentzian dip fit** -> center, FWHM, R^2, and **contrast** (= dip depth A in survival =
     baseline `y0` minus on-resonance floor `y0-A`). `fit_spectrum_<fid>.png`.
   - **img1 atom-vs-empty intensity separation histogram** -> pooled `d'`, per-site `d'` median, empty &
     atom `mu`/`sigma`, **delta-mu**. `intensity_separation_<fid>.png`.
   It prints a one-line summary + an `HWP_JSON` line with all metrics.
3. **Log** each angle to the Notion daily page (one entry: spectrum + separation figures + the metric
   line), under a single "beam-2 imaging HWP polarization sweep" section.

## Results (2026-06-28)
Scan range 260:3:360 MHz, beam 2 only, 33x33_feedback9. `empty_mu ~ 200.1`, `empty_sigma ~ 0.6` (tight)
throughout, so `delta-mu` (atom brightness above pedestal) is the sensitive collection metric; `d'` barely
moves because the empty width dominates.

| HWP (deg) | shots | delta-mu (ADU) | d' per-site | contrast | scan_id |
|---|---|---|---|---|---|
| 26 (start/baseline) | 109 | 6.0 | 3.90 | 1.15 | 20260628142831 |
| 40 | 102 | 5.2 | 3.90 | 1.13 | 20260628144215 |
| 15 | 102 | 5.3 | 3.96 | 1.18 | 20260628145031 |
| 20 | 102 | 6.3 | 4.08 | 1.25 | 20260628145600 |
| 30 | 102 | 5.3 | 4.06 | 1.18 | 20260628150118 |
| 23 | 102 | 5.0 | 3.91 | 1.12 | 20260628150937 |
| **23** | **408** | **5.4** | **3.88** | **1.08** | 20260628151735 |
| **20** | **408** | **5.8** | **4.12** | **1.18** | 20260628153035 |

### Reading
- The **100-shot sweep was non-monotonic** (15:5.3, 20:6.3, 23:5.0, 26:6.0, 30:5.3, 40:5.2). A smooth HWP
  collection curve cannot dip at 23 then rise at 26, so the ~0.5-1.0 ADU spread at 100 shots is **within
  per-angle shot scatter** -- the angle effect is weak and under-resolved at that statistics.
- The **high-rep (408-shot) head-to-head settled it**: **20 deg > 23 deg** (delta-mu 5.8 vs 5.4, d'
  per-site 4.12 vs 3.88) -- a **real but modest** collection gain (~6%). The 100-shot 20 deg (6.3) was a
  high fluctuation; the true value is ~5.8.
- **Contrast stayed ~flat** (1.08-1.25, mostly 1.13-1.18) across all angles -- consistent with the
  hypothesis: the HWP changes **collection**, not **scattering**. (The 20-deg 100-shot 1.25 was
  baseline-fit noise; the broad-wing baseline overshoots ~1.2-1.35, so contrast-as-fit-depth is itself
  noisy -- the on-resonance **floor** ~0.02-0.13 is the cleaner "fully pushed out" check.)
- **26 deg high-rep was NOT taken** (paused before it). 26 read 6.0 at low rep; whether it rivals 20 at
  high rep is open.

## PAUSED -- why, and how to resume
**Paused 2026-06-28:** imaging is **fluctuating shot-to-shot**; the per-angle delta-mu scatter at 100
shots (~0.5 ADU) swamps the real ~0.4 ADU (~6%) angle effect, and even 408-shot points only just resolve
20 vs 23. Optimizing further is not worth it until the **imaging intensity is hardware-stabilized**.

**Resume plan (later, after hardware stabilization):**
- Re-run the high-rep head-to-head at **20 / 23 / 26 deg** (>=400 shots each), now that the per-shot noise
  is reduced -- the angle dependence should resolve cleanly.
- Bracket past 20 toward smaller angles if 20 still wins (the 100-shot peak was ~20).
- Decide the operating HWP angle; today it was **left at 20 deg** (the best high-rep point so far, better
  than 23/26 at low rep).
- Tooling is ready: `pyctrl/tools/hwp_point.py` + the beam-2 `Spectrum399Scan` edit.

## Caveats / gotchas for next time
- **delta-mu, not d', is the sensitive metric** here (empty sigma ~0.6 dominates d'); report both but judge
  on delta-mu / atom_mu.
- **Contrast from the Lorentzian fit depth is noisy** (broad-wing baseline overshoots >1). For a cleaner
  "scattering unchanged" check use the on-resonance **floor** survival, or fit with a constrained baseline.
- **~100 shots is too few** to resolve a ~6% collection change against the current imaging fluctuation;
  use >=400 and ideally repeat.
- Beam-2-only means beam 1 (`AmpAbsImag`) at amp 0; both 399 shutters still open in `Pushout399Step`
  (beam 1 just dark). Remember to **restore `Spectrum399Scan.py`** (Amp1/Amp2 + freq range) when done.
