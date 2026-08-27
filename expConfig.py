"""expConfig.py -- executable, actively-maintained twin of ``matlab_new/expConfig.m``.

pyctrl's PRODUCTION config source: channel aliases, channel default values, physical constants,
and AWG/Orca defaults. This replaces the frozen ``tests/reference/config_reference.json`` SNAPSHOT
for the live runner (the snapshot is demoted to the MATLAB-ground-truth reference for the drift
oracle), curing the silent staleness of the captured config -- the config is now CODE.

expConfig.py is pyctrl's live, executable config source -- the runner reloads it each job, and
calibration edits (e.g. the daily ``Resonance556mj0Freq`` update) land here directly. Its values
feed the serialize path, so ``tests/test_exp_config.py`` (the drift oracle) asserts this module
still resolves to the SAME config as the frozen MATLAB capture
(``tools/capture_config_reference.m``) -- a regression guard, not a live-MATLAB tracking
requirement. ``matlab_new/expConfig.m`` is the reference the snapshot came from; if a scan still
runs under MATLAB keep the change in sync there too, and re-capture the snapshot when you
deliberately change a value.

:func:`build_config` returns the raw dict ``SeqConfig`` consumes (same shape
``capture_config_reference.m`` emits). The long-lived runner reloads this module once per job
(``SeqConfig.load_real(reload=True)``), so edits go live without a restart -- only ``lib/`` (the
framework) and the expConfig snapshot capture still need a restart / re-capture.

Naming/units follow ``expConfig.m``: alias prefixes are load-bearing (``TTL*`` FPGA, ``V*`` NI DAQ,
``Freq*``/``Amp*`` DDS); defaults are Hz / s. Values mirror the ``.m`` literal forms for readability
(they resolve to the same IEEE-754 doubles); ints are float-coerced downstream by ``SeqConfig``.

The per-pattern overlay + const cross-ref logic lives in ``lib/expConfig_helper.py`` (this
module stays the pure data/config source); ``_consts`` calls it for the cross-refs.
"""

import expConfig_helper


def build_config():
    """Return the raw config dict (aliases / consts / defaults / NI wiring) for ``SeqConfig``."""
    channel_alias = _channel_alias()
    consts = _consts()
    default_vals = _default_vals(consts)
    return {
        "channel_alias_keys": list(channel_alias.keys()),
        "channel_alias_vals": list(channel_alias.values()),
        "consts": consts,
        "default_vals_keys": list(default_vals.keys()),
        "default_vals_vals": list(default_vals.values()),
        # NI DAQ external clock + start-trigger wiring (FPGA clock -> Dev1/PFI0;
        # FPGA TTL0 -> Dev1/PFI1). Run-loop NI arm only; omitted from the byte path.
        "ni_clocks_keys": ["Dev1"], "ni_clocks_vals": ["PFI0"],
        "ni_start_keys": ["Dev1"], "ni_start_vals": ["PFI1"],
    }


def _channel_alias():
    """Channel name -> backend path (``expConfig.m`` ``channelAlias``)."""
    a = {}
    a["Dev1"] = "NiDAQ/Dev1"                    # NIDAQ backend convention

    # ---- TTL channels (FPGA1) ----
    a["TTL556RydAWG"] = "FPGA1/TTL1"
    a["TTL399IMG1PIDMode"] = "FPGA1/TTL2"  # 1 for lock, 0 for hold
    a["TTLScopeTrig"] = "FPGA1/TTL3"
    a["TTL556RydbergShutter"] = "FPGA1/TTL4"
    a["TTL556MOTaShutter"] = "FPGA1/TTL5"
    a["TTL556MOTbShutter"] = "FPGA1/TTL6"
    a["TTL556MOTcShutter"] = "FPGA1/TTL7"
    a["TTL399AbsImagShutter"] = "FPGA1/TTL8"
    a["TTL399MOTShutter"] = "FPGA1/TTL9"
    a["TTL3992DMOTShutter"] = "FPGA1/TTL10"
    a["TTL369Shutter"] = "FPGA1/TTL11"
    a["TTL308RydAWG"] = "FPGA1/TTL12"
    a["TTLThorCamTrig"] = "FPGA1/TTL13"
    a["TTLQickTrig"] = "FPGA1/TTL14"
    a["TTL556RydAWGChSwitch"] = "FPGA1/TTL16"
    a["TTL556RydAWGSwitch"] = "FPGA1/TTL17"
    a["TTL399Imag2Shutter"] = "FPGA1/TTL18"
    a["TTL399IMG2PIDMode"] = "FPGA1/TTL19"  # 1 for lock, 0 for hold
    a["TTL308RydAWGSwitch"] = "FPGA1/TTL55"
    a["TTLOrcaTrig"] = "FPGA1/TTL54"
    a["TTLSampleAndHold"] = "FPGA1/TTL15"
    a["TTLTrig"] = "FPGA1/TTL3"
    a["TTL369Switch"] = "FPGA1/TTL44" # Unconnected be careful!!!
    a["TTL308RydAWGChSwitch"] = "FPGA1/TTL53"
    a["TTLIonizationSwitch5to8"] = "FPGA1/TTL48"
    
    # ---- DDS channels (FPGA1) -- Name/FREQ and Name/AMP, no prefix ----
    a["Freq556RydbergMOTh"] = "FPGA1/DDS0/FREQ"
    a["Amp556RydbergMOTh"] = "FPGA1/DDS0/AMP"
    a["Freq556MOTX"] = "FPGA1/DDS1/FREQ"
    a["Amp556MOTX"] = "FPGA1/DDS1/AMP"
    a["FreqSLM"] = "FPGA1/DDS2/FREQ"
    a["AmpSLM"] = "FPGA1/DDS2/AMP"
    a["FreqAODs"] = "FPGA1/DDS3/FREQ"
    a["AmpAODs"] = "FPGA1/DDS3/AMP"
    a["FreqAOM308"] = "FPGA1/DDS4/FREQ"
    a["AmpAOM308"] = "FPGA1/DDS4/AMP"
    a["FreqAODv"] = "FPGA1/DDS5/FREQ"
    a["AmpAODv"] = "FPGA1/DDS5/AMP"
    a["FreqSLMmodulation"] = "FPGA1/DDS6/FREQ"
    a["AmpSLMmodulation"] = "FPGA1/DDS6/AMP"
    a["FreqEOM616"] = "FPGA1/DDS7/FREQ"
    a["AmpEOM616"] = "FPGA1/DDS7/AMP"
    a["FreqAOM616"] = "FPGA1/DDS8/FREQ"
    a["AmpAOM616"] = "FPGA1/DDS8/AMP"
    a["Freq556RydbergHF"] = "FPGA1/DDS9/FREQ"
    a["Amp556RydbergHF"] = "FPGA1/DDS9/AMP"
    a["Freq369"] = "FPGA1/DDS12/FREQ"
    a["Amp369"] = "FPGA1/DDS12/AMP"
    a["Freq399Imag2"] = "FPGA1/DDS17/FREQ"
    a["Amp399Imag2"] = "FPGA1/DDS17/AMP"
    a["FreqAbsImag"] = "FPGA1/DDS18/FREQ"
    a["AmpAbsImag"] = "FPGA1/DDS18/AMP"
    a["FreqBlueMOT"] = "FPGA1/DDS19/FREQ"
    a["AmpBlueMOT"] = "FPGA1/DDS19/AMP"
    a["FreqZeeman"] = "FPGA1/DDS20/FREQ"
    a["AmpZeeman"] = "FPGA1/DDS20/AMP"
    a["Freq2DMOT"] = "FPGA1/DDS21/FREQ"
    a["Amp2DMOT"] = "FPGA1/DDS21/AMP"

    # ---- NI DAQ voltages (Dev1) ----
    a["VMOTCoil"] = "Dev1/0"
    a["VBiasCoilY"] = "Dev1/1"
    a["VIonizationSet5to8"] = "Dev1/2"
    a["VBiasCoilX"] = "Dev1/3"
    a["VBiasCoilZ"] = "Dev1/4"
    a["VRydCoil"] = "Dev1/6"
    a["VSLMservo"] = "Dev1/8"
    a["VPicoMotor308h"] = "Dev1/10"
    a["VPicoMotor308v"] = "Dev1/11"
    a["VElectrode1"] = "Dev1/12"
    a["VElectrode2"] = "Dev1/13"
    a["VElectrode3"] = "Dev1/14"
    a["VElectrode4"] = "Dev1/15"
    a["VElectrode5"] = "Dev1/16"
    a["VElectrode6"] = "Dev1/17"
    a["VElectrode7"] = "Dev1/18"
    a["VElectrode8"] = "Dev1/19"
    a["VImg1PIDSet"] = "Dev1/21"
    a["VPicoMotor369h"] = "Dev1/22"
    a["VPicoMotor369v"] = "Dev1/23"
    a["VImg2PIDSet"] = "Dev1/24"
    return a


def _consts():
    """Physical constants / calibrated values (``expConfig.m`` ``consts``)."""
    c = {}
    # Data saving
    c["MatlabURL"] = "tcp://127.0.0.1:1408"
    c["PathPrefix"] = r"D:\OneDrive - Harvard University\Documents - Yb"

    # Orca Quest camera
    c["Orca"] = {"ROI": [1000, 100, 2100, 2100], "ExposureTime": 0.050004}

    # 556nm resonance (calibrate daily by spectroscopy; 3P1 mj=0 near-magic)
    # fit 2026-08-18 (Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.952, FWHM 67.2 kHz, 205 shots, scan 20260818080342, 33x33_feedback11, loading 0.59); +17.8 kHz vs the standing 108.0600e6 (the 08-12 fit -- config was NOT updated on 08-13..08-17, so this is a ~6-day drift, ~3 kHz/day, not a one-day jump). mj0-mj1 splitting 2.6062 MHz (mj1 105.4716 MHz, FWHM 808.6 kHz, R^2=0.986, 105 shots, scan 20260818080808, stock 104.5:0.1:106.5 window, interior); +86.8 kHz vs 08-12's 2.5194, and the ~809 kHz mj=1 width is the same inhomogeneous trap-depth spread 08-11/08-12 documented (692.9 / 863.0 kHz), so the spread has still NOT come down. USER-DIRECTED SCOPE: mj=0 + mj=1 only, plus a 70 G Rydberg push-out spectrum; no 399, no 30 G set, no revival, no AT. Pre-flight: backend idle; 399 wavemeter PID engaged=true, det +1.17 MHz, DAC online, V 6.1167; 556 det +2.17 MHz; dashboard SLM-camera endpoint still 503 "no data for camera_png yet" (cache empty, not a blank SLM) so the pattern was validated by the warm-up loading rate 0.55-0.63 (627/1068 sites logical-1). was 108.0600e6  (08-12 fit; full history in the entry below)
    # fit 2026-08-12 (Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.956, FWHM 55.2 kHz, 205 shots, scan 20260812100054, 33x33_feedback11, loading 0.55); +8.9 kHz vs 08-11, within the usual daily drift. mj0-mj1 splitting 2.5194 MHz (mj1 105.5406 MHz, FWHM 863.0 kHz, R^2=0.985, 165 shots, scan 20260812100530), -144.8 kHz vs 08-11's 2.6642; mj=1 run on the WIDENED 103.8:0.1:107.0 window (33 pts) that 08-11 established, not the stock 104.5-106.5, and the fit is interior with R^2 0.985 -- the ~863 kHz width is the same inhomogeneous trap-depth spread 08-11 documented (692.9 kHz then), so the spread has NOT come back down. USER-DIRECTED SCOPE: mj=0 + mj=1 + the 30 G reference set + a NEW 20 G Rydberg set; no 399 scan. *** PUSH-INDEPENDENT ~45% SURVIVAL FLOOR IN THE RYDBERG PUSH-OUT STEP AT FIELD (new today, unexplained) ***: every high-field scan tops out at survival ~0.52-0.55 instead of the ~0.95 baseline every run through 08-07 showed, while the SAME shots' 0-field mj=0/mj=1 scans sit at 0.97 -- so it is specific to RydbergPushoutStep (TTL556RydbergShutter + Amp556RydbergMOTh), not imaging/loading (loading 0.55-0.59 all day, normal). It is NOT the 556 push power: halving the push at 30 G (amp 0.10 -> 0.05) left the off-resonant baseline flat at 0.54 -> 0.52 while the dip depth changed as expected. Ryd308.Amp defaults to 0 in expConfig, so it is not stray 308 either. Prime suspect = a 556-Rydberg-beam leak (shutter open, AOM off is not dark) or something in the bias-coil ramp; NOT diagnosed further today. Consequence: every high-field contrast below is quoted against a ~0.53 ceiling, not 0.95. THE 30 G PUSH AMP IS ALSO NOW TOO STRONG at the 08-05..08-07 operating value 0.10: it gave FWHM 307.7 kHz (vs 161.4 kHz on 08-07 at the same 0.10) and drove the dip to 0.01, i.e. saturated; 0.05 restored an unbroadened FWHM 95.6 kHz. Both amps agree on the CENTER to 5.0 kHz (143.6187 @ 0.10 / 143.6237 @ 0.05), so the center stands: 30 G dip 143.6237 MHz (FWHM 95.6 kHz, R^2=0.909 -- the modest R^2 is dip DEPTH against the 0.52 ceiling, not center uncertainty; scans 20260812100935 @ 0.10 and 20260812101355 @ 0.05), +84.6..+89.6 kHz vs 08-07's 143.5341. NEW: 20 G RYDBERG SET (user-directed; the 20 G window had to be derived, not looked up -- RydbergSpectrum556Scan hard-codes center_mhz=143.5 for 30 G and 556AutlerTownesScan hard-codes 143.4, so BOTH gained a --center/--half/--step CLI today, defaults unchanged). Window derived from TODAY's two measured points by linear Zeeman scaling: center20 = RES0 + (2/3)*(dip30 - RES0) = 108.0600 + (2/3)*35.5637 = 131.77 MHz, swept 131.22-132.22. **20 G 556 resonance = 131.8467 MHz** (FWHM 174.6 kHz, R^2=0.977, 205 shots, scan 20260812102327, push amp 0.05, window 130.85-132.85 @ 50 kHz). A first 20 G pass at amp 0.10 (scan 20260812101909) was WASHED OUT -- survival 0.00-0.34 across the whole +-0.5 MHz window, FWHM 885.8 kHz -- but still put the center at 131.8502, i.e. the two agree to 3.5 kHz. Zeeman scaling is self-consistent: 1.1893 MHz/G at 20 G vs 1.1855 MHz/G at 30 G (0.3% apart) against the 1.178 constant in the scan. **20 G 616 revival peak = 236.0263 MHz** (FWHM 14.8 MHz, R^2=0.986, 255 shots, scan 20260812105321, 556 parked on 131.8467 at amp 0.05, 308 amp 0.4; user confirmed the 616 ULE was locked before the run). That is +2.0 MHz vs the 30 G revival history (233.4-234.0), i.e. the 308 line moves only ~0.2 MHz per Gauss in EOM616 terms -- far less than the ~0.8 MHz/G a naive g=2 Rydberg Zeeman estimate gives, worth a look. Revival recovers to 0.53 = the full available ceiling, so the revival contrast is intact despite the floor. **20 G 556 AUTLER-TOWNES: dips 130.5990 MHz (FWHM 310.9 kHz) + 132.5223 MHz (FWHM 385.2 kHz), SPLITTING 1.9233 MHz**, 2-peak R^2=0.894 vs 0.205 single (decisively two-component), 429 shots, scan 20260812105931, 308 amp 0.4 parked on 236.0263 MHz, 556 probe amp 0.05, window CENTERED on the measured bare line 131.8467 +-3 MHz. Splitting 1.923 MHz at 20 G vs 1.952 (08-03) / 1.912 (08-07) at 30 G -- essentially unchanged, as expected for an AT splitting set by the 308 Rabi frequency rather than by the bias field. *** THIS RESOLVES THE OPEN QUESTION IN gotcha-556-pushout-amp-highfield-scans ***: the AT doublet MIDPOINT sits 131.5606 MHz = **-286 kHz below the bare line**, the same -335/-354 kHz offset seen on 08-01/08-03 -- but today the AT window was centered ON the measured bare line (131.8467, via the new --center flag) instead of the hard-coded 143.4, so the offset is NOT a window-centering / fit artifact. It survives a correctly centered window at a different field, which points at a real 308-induced light shift of the dressed doublet. Next test remains a --ryd308-amp sweep. NOTE also that fit_spectrum.py --peaks 2 FAILED on this doublet: its seed collapsed onto one shoulder (reported 132.2516/132.5516, splitting 0.300 MHz, R^2 0.596) even though the raw curve shows two obvious 1.9 MHz-separated dips; the numbers above come from a hand-seeded double-Lorentzian fit (figure fit_AT_20G_20260812105931.png in the scan dir). Do not trust --peaks 2 on a wide AT window without eyeballing the curve. Pre-flight: backend idle; the previous job was a 17x17_20um rearrangement scan so the SLM held a rearrangement phase -- user chose 33x33_feedback11, written with z4=-5 (base sha 8f8e2345, same as 08-07/08-11) and validated by the warm-up loading rate 0.55-0.61 since the dashboard SLM-camera endpoint still returns 503 "no data for camera_png yet". 399 wavemeter PID engaged=true, det -1.10 MHz, DAC online, V 6.043; 556 det +2.24 MHz; 616 has no lock block in yb_monitor (per 08-07, gate 616 on the ULE scope, not the wavemeter) and the user confirmed it locked. was 108.0511e6 (08-11: Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.947, FWHM 40.4 kHz, 205 shots, scan 20260811141215, 33x33_feedback11, loading 0.58); +1.0 kHz vs 08-10, within linewidth. REPRODUCED: a first 205-shot run (scan 20260811140626) gave 108.0509 MHz (R^2=0.918, FWHM 59.9 kHz) -- the two agree to 0.2 kHz, so the marginal R^2 is dip DEPTH (only ~14% at this push), not center uncertainty. TODAY'S OPERATING POINT IS NOT THE ByPattern DEFAULT: user directive VServo 1.9 (vs ByPattern 3.3) and loading plane z4 -4.0 (vs ByPattern SLM.Loading.Defocus -2.5), both applied as per-scan overrides -- the config keys were NOT edited. The plane was MEASURED today at VServo 1.9, not inherited: 17-plane sweep (20260811_135519, -7..-3 step 0.25, 15 shots/plane, pooled 2-Gaussian EM) puts the quadratic vertex at -4.5 on a flat -5.25..-3.5 plateau, and a 40-shot/plane head-to-head (20260811_140133) gives dist/d' 4.37/3.07 at -4.0 vs 4.16/2.92 at -5.5 vs 3.26/2.45 at the ByPattern -2.5. (A first coarse 21-plane stack 20260811_133531 peaked at -5.5 but is CONTAMINATED -- 399 unlocked partway through it, loading 0.26-0.35 vs 0.57-0.61 in every clean run -- discard it.) *** VSLMServo IS NOT ACTUALLY MOVING THE TRAP DEPTH ***: the descriptor confirms SLM.VServo=1.9 + Init.VSLMServo=1.9 reached the sequence, yet the depth-differential |mj|=1 line sits at 105.5464 MHz (scan 20260811141715) vs 105.5606 at VServo 3.3 on 08-10 -- a 15 kHz move on a 2.5 MHz splitting -- and loading is unchanged (0.58 vs 0.61). That matches job 773's VServo sweep (loading FLAT 0.54-0.57 across 2.7-3.9), so the 532 power servo appears saturated or its setpoint is not reaching the power loop over 1.9-3.9 V. mj0-mj1 splitting 2.6642 MHz (mj1 105.3869 MHz, FWHM 692.9 kHz, R^2=0.967, 165 shots, scan 20260811142749), +174.7 kHz vs 08-10's 2.4895. mj=1 NEEDED RE-WINDOWING AND THE WIDTH IS REAL: the stock 104.5-106.5 window and a 105.5-107.6 retry both TRUNCATE the line (the latter fit 363 kHz at 105.5464 by seeing only its upper half); widening to 103.8-107.0 (~3.7 FWHM) gave 692.9 kHz, and 104.8-106.4 gave 865 kHz at the same center, i.e. truncation inflates the WIDTH while the CENTER stays put (105.3961/105.3869, 9 kHz apart). Power broadening is NOT the cause (user hypothesis, tested): amp 0.10 -> 692.9 kHz, amp 0.075 -> 642.3 kHz (-7% while the dip depth halves, center 105.4244), amp 0.05 -> NO DIP AT ALL (survival flat 0.98-0.99, R^2=0.07) -- the push-out is sharply nonlinear and 0.10 is barely above threshold. So the ~2x width vs 08-10 (692.9 vs 316.0 kHz) is INHOMOGENEOUS: the array's trap-depth distribution has roughly doubled in spread. NOTE the mj=1 push reads Amp 0.10 / 20 ms from the BUILT ScanGroup -- the runbook's "Amp 0.18" is wrong (same docstring-lies gotcha as the 30 G amps). was 108.0501e6 (08-10: Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.968, FWHM 44.9 kHz, 203 shots, scan 20260810095925, 33x33_feedback11, loading 0.61); +14.7 kHz vs 08-07 -- above the usual few-kHz daily drift, flagged to user (3-day gap since 08-07, so ~5 kHz/day). mj0-mj1 splitting 2.4895 MHz (mj1 105.5606 MHz, FWHM 316.0 kHz, R^2=0.971, scan 20260810102020, 310 shots), -189.4 kHz vs 08-07's 2.6789 -- the SPLITTING NARROWED because mj=1 moved +204.1 kHz while mj=0 moved only +14.7 kHz. The mj=1 shift is REAL, not a fit artifact: a first 203-shot run (scan 20260810100930) gave 105.5718 MHz at a marginal R^2=0.931 and the re-run reproduced it to 11 kHz at R^2=0.971. mj=1 tracks trap depth (the |mj|=1 light shift is depth-differential), so a +204 kHz mj=1 move with a nearly-static mj=0 points at a TRAP-DEPTH change on 33x33_feedback11 since 08-07, not a laser/ULE drift -- worth checking VSLMServo / 532 power if it persists. ONLY the two 556 scans were run today (user asked for mj=0 + mj=1 only; no 399, no 30 G set, no revival, no AT). Pre-flight: loading 0.601, per-site d-prime median 5.32 (99% of sites >3), img1-img2 corr 0.955; 399 wavemeter PID read engaged=false with det +1.6 MHz and its lock VOLTAGE RAILED AT 8.0 V (range 2-8) -- left alone since 399 is transfer-cavity-locked (08-06), but a railed servo has no headroom and is worth a look; 556 det +1.6 MHz; 616 OFFLINE on the wavemeter (irrelevant today, no 616 scans); SLM dashboard camera endpoint still returns 503 "no data for camera_png yet" (cache empty, not a blank SLM) and the server tracks no loading-pattern name, so the pattern was reloaded blind (33x33_feedback11, z4=-5, base sha 8f8e2345 -- same as 08-07) and validated by the warm-up loading rate. TWO USER DIRECTIVES applied to the daily calibration this session: (1) the mj=1 window narrowed from 103.5-106.5 to **104.5:0.1:106.5** (21 pts, was 31) since the line has walked up and the low edge was dead range; (2) **never submit more than --reps 5** for these scans -- they converge fast and extra reps are wasted apparatus time (both changes written into Spectrum556Scan.py + daily-system-scan.md; the auto-reps memory was amended so its "add reps" advice no longer contradicts the cap -- remediate a poor fit by RE-RUNNING at reps 5, not by raising reps). was 108.0354e6 (08-07: Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.973, FWHM 47.9 kHz, 202 shots, scan 20260807150200, 33x33_feedback11, loading 0.61); +7.1 kHz vs 08-06, within the usual daily drift; mj0-mj1 splitting 2.6789 MHz (mj1 105.3565 MHz, FWHM 361.2 kHz, R^2=0.959, scan 20260807150614), +27.5 kHz vs 08-06's 2.6514. 399 run at Amp2 1.0 (carrying forward 08-06): first pass 87 shots gave R^2=0.957 with only a ~9%-deep dip, re-run at reps 5 -> 145 shots, center 310.1196 MHz, FWHM 15.5 MHz, R^2=0.970 (scan 20260807153530); center moved -7.7 MHz vs 08-06's 317.8174 -- LARGE, but the 08-06 fits sat at 317.8-318.0 while today's two independent passes agree at 310.59 / 310.12 MHz, so it is reproducible within the day. 2-Lorentzian DEGENERATE again on both passes (components merged, R^2 identical to single) -- no resolvable doublet, second day running. 30 G set run at push amp 0.10 (carrying forward the 08-05 user-approved value under the raised Rydberg-h power; NOTE the scan's built-in field-scaled default is 0.15 and its --help text claims 0.4 -- both stale, see the gotcha-556-pushout-amp-highfield-scans memory): 30 G dip 143.5341 MHz (FWHM 161.4 kHz, R^2=0.987, survival 0.22-0.96, 155 shots, scan 20260807153843), -5.4 kHz vs 08-06's 143.5395 -- amp 0.10 reproduced the expected line shape, confirming the choice. 616 revival peak 233.3752 MHz (R^2=0.978, FWHM 14.9 MHz, survival 0.18-0.96, 155 shots, scan 20260807154228), -603 kHz vs 08-06's 233.9782 -- a REAL line move. IMPORTANT PROCEDURAL CORRECTION (user, 08-07): the SINGLE criterion for a 616 lock is the 616ULE SCOPE, NOT the wavemeter. The yb_monitor wavemeter read 616 at -27.5 to -36.8 MHz all session (n=20 during this scan, mean -31.7, every sample |det|>15 MHz) and that was initially -- WRONGLY -- called an unlock and used to reject this scan. 616 has NO lock block in yb_monitor (nothing servos it there, unlike 399), so its target_ghz is a stale reference number and a large, STABLE offset (sigma ~1.7 MHz) is a wavemeter reference error, not a laser excursion. Do NOT gate 616 scans on the wavemeter detuning; check the 616ULE scope. This also puts 08-06's "616 cavity cycled in and out of lock" note in doubt -- that call was made from the same wavemeter signal. The revival was RE-RUN as a reproducibility check (scan 20260807160300, 154 shots): center 233.3464 MHz, FWHM 14948.3 kHz, R^2=0.975 -- agreeing with the first run to -28.8 kHz on a 14.9 MHz feature (0.2%) with FWHM matching to 0.1 kHz. and ~155 shots is ample for this line (user: enough reps for the 616 scan). *** BOTH OF THOSE REVIVAL SCANS ARE SUSPECT / DO NOT USE: at 16:20 the user reported 616 really unlocked and the 616ULE SCOPE (192.168.0.40, ch2 Transmission / ch3 Error signal, read via the scope_control dashboard at http://<rearr-tailscale>:8600/api/scope/192.168.0.40/read) CONFIRMED IT -- CH2 flat at 1.62-1.70 V, Vpp 0.08 V, no cavity fringe; CH3 error signal dead flat at zero (Vpp 0.024 V, mean 0.0003 V) -- while CH4 showed the scan ramp live and the trigger status TD, so the scope was acquiring fine. A locked ULE shows a transmission PEAK on ch2 and a dispersive zero-crossing on ch3; both were absent. Drop time is unknown, so the -603 kHz offset vs 08-06 AND the 28.8 kHz run-to-run agreement are BOTH explained by an unlocked-but-parked cavity (a stable wrong condition reproduces just as well as a right one) -- agreement between two scans is NOT evidence of lock. The baseline tilt below is likewise suspect. The 556 AUTLER-TOWNES scan that had been submitted off the 233.3752 center was ABORTED mid-run at 152 shots (job 549) rather than keep taking data against a bad 308 park. *** The user then RE-LOCKED 616 (16:24) and directed: skip the revival re-run, go straight to AT, queued at the top. NOTE the 616ULE scope still read the SAME flat signature after the re-lock (ch2 Vpp 0.04 V at 1.68-1.72, ch3 +-0.01 V at zero) -- the user confirmed the lock regardless, so that flat ch2/ch3 view is NOT a reliable lock indicator from the dashboard read (wrong timebase/scale, or a locked cavity simply has no scan ramp); ASK for the expected signature before judging lock from this scope again. 556 AUTLER-TOWNES (scan 20260807163624, 254 shots taken / 216 fit, 308 amp 0.4 parked on the pre-lock revival 233.3752 MHz, 556 probe amp 0.10): dips 142.1268 MHz (FWHM 338.4 kHz) + 144.0391 MHz (FWHM 385.0 kHz), SPLITTING 1.912 MHz, 2-peak R^2=0.961 vs 0.236 single (decisively two-component). Splitting 1.912 vs 08-06's 1.807 and the 07-20 reference ~1.3 MHz -- still growing with the 308 Rabi frequency, consistent with the raised push power. Because the --eom616 park came from an UNLOCKED-616 revival, the 308 may sit slightly off resonance, which would only REDUCE the splitting -> treat 1.912 MHz as a LOWER BOUND. Doublet midpoint 143.083 MHz is NOT a line-center measurement (the AT scan hard-codes its 556 window center at 143.4 MHz, no CLI override, vs today's 30 G line at 143.5341). Per user direction the AT scan was aborted on FIT QUALITY rather than a fixed shot count (2-peak R^2 >= 0.93, decisively better than single, both dips interior, splitting stable across two consecutive checks): the splitting read 1.912 MHz identically at n=216 and n=252, so ~250 shots sufficed vs the runbook's 350-450. LINESHAPE TILT (user-spotted): the revival is visibly tilted, but the tilt is in the BASELINE, not the peak -- left wing (210-215 MHz) floors at 0.305 vs right wing (255-260) at 0.202, a -0.103 slope across the window, while the peak itself is only mildly asymmetric (left half-width 7.74 MHz vs right 6.70, asym -0.072). The 556 is NOT mis-set: provenance confirms Pushout.Green.Freq = 143.5341 MHz = today's fitted dip exactly, and both centers are model-robust (30 G dip Lorentzian 143.5341 vs model-free half-crossing midpoint 143.5307, only 3.4 kHz apart on a 164 kHz FWHM; revival Lorentzian 233.375 vs half-midpoint 233.480, 105 kHz on a 14.4 MHz FWHM). The sloping floor is off-resonant 616/308 loss growing toward the blue end = the same over-power that gives FWHM 14.9 MHz vs the 07-20 reference ~4.5 MHz (third day running: 08-05 18.5, 08-06 15.0, 08-07 14.9), with 308 amp pinned at its 0.4 max and the 556 push raised ~7x on 08-05. Separately the 30 G DIP is itself asymmetric (left half-width 91.2 kHz vs right 72.5, asym -0.114 -- leans red), a smaller effect that does not move the center meaningfully. UNTESTED HYPOTHESIS: dropping 308 amp (e.g. to 0.2) should collapse both the tilt and the broadening if the over-power reading is right. 399 wavemeter PID lock read engaged=false at pre-flight with det only +2.5 MHz and the DAC online; it was re-engaged (det -> 1.2 MHz, V 6.39) -- but per 08-06 the user confirmed 399 is now locked via the TRANSFER CAVITY, so the wavemeter PID may no longer be the lock authority and the engage was likely unnecessary. Loading 0.60 throughout, per-site d-prime median 5.09 (100% of sites >3). was 108.0283e6 (08-06: Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.969, FWHM 51.0 kHz, 208 shots, scan 20260806110300, 33x33_feedback11, loading 0.55); UNCHANGED vs 08-05 -- delta -6.8 Hz, the flattest day-over-day yet (value kept at 108.0283e6); mj0-mj1 splitting 2.6514 MHz (mj1 105.3769 MHz, FWHM 381.3 kHz, R^2=0.948, scan 20260806110738), -71.8 kHz vs 08-05's 2.7232 MHz. 399 needed the push raised to Amp2 1.0: the standard 0.5 gave only a ~6%-deep dip at BOTH 87 and 216 shots (R^2 0.877 then 0.794 -- MORE shots made the fit WORSE, so power-limited not statistics-limited, same push-power regression the scan docstring records for 08-03); at Amp2 1.0 R^2 jumped to 0.975 (center 317.8174 MHz, FWHM 14.7 MHz, 145 shots, scan 20260806111950). 399 center reproducible across all three runs (317.9876 / 317.8557 / 317.8174 MHz) but the 2-Lorentzian fit went DEGENERATE (components merged) on every one -- no resolvable doublet today, unlike the usual ~15 MHz mj-split. 30 G set run at push amp 0.10 (user-confirmed, carrying forward 08-05's ~7x Rydberg-h power raise): 30 G dip 143.5395 MHz (FWHM 168.9 kHz, R^2=0.991, survival 0.17-0.95, 154 shots, scan 20260806112443), +11.5 kHz vs 08-05's 143.5280. 616 revival peak 233.9782 MHz (R^2=0.985, survival 0.16-0.93, 156 shots, scan 20260806112825), +15.8 kHz vs 08-05's 233.9624; FWHM 15.0 MHz -- still power-broadened vs the 07-20 reference ~4.5 MHz, second day running. 616 CAVITY CYCLED IN AND OUT OF LOCK mid-session (wavemeter det swung -23.2 -> -6.5 -> -28.2 -> -7.2 MHz over ~15 min; user confirmed unlock, then relock, then a second unlock). The revival was RE-RUN after the first relock (scan 20260806113438, 107 shots) and reproduced the center to 0.2 kHz (233.9782) with FWHM 15.5 MHz -- i.e. the 616 unlock did NOT cause the revival broadening; the ~15 MHz width is genuinely the raised-556-push power broadening, matching 08-05's 18.5 MHz. Two AT attempts died to 616 excursions (job 188 aborted at seq 31 with 616 at -28 MHz; an earlier run aborted at 86 shots); the KEPT AT run (scan 20260806113952) was polled with per-sample 616 tracking and 616 stayed in band the whole time (n=28 samples, min -14.1 max -3.7 mean -8.8 MHz, ZERO samples |det|>15 MHz). 556 AUTLER-TOWNES doublet: dips 142.4958 MHz (FWHM 330.4 kHz) + 144.3025 MHz (FWHM 407.2 kHz), SPLITTING 1.807 MHz, 2-peak R^2=0.949 vs 0.398 single (decisively two-component), 320 shots, 308 amp 0.4 parked on the measured revival 233.9782 MHz. Splitting 1.807 MHz vs the 07-20 reference ~1.3 MHz = larger 308 Rabi frequency, consistent with the raised push power. NOTE the AT scan hard-codes its 556 window center at 143.4 MHz (no CLI override) while today's 30 G line sat at 143.5395, so the doublet is ~140 kHz off-center in the window -- harmless at +-3 MHz half-width, but the doublet midpoint 143.399 MHz should NOT be read as a line-center measurement. 399 wavemeter PID lock read engaged=false at pre-flight but the USER CONFIRMED 399 IS NOW LOCKED VIA THE TRANSFER CAVITY, so the wavemeter PID is no longer the lock authority for 399 -- no DAC intervention needed today (contrast the 08-03/08-04/08-05 MCC-DAC outages). was 108.0283e6 (08-05: Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.960, FWHM 45.1 kHz, 214 shots, scan 20260805174955, 33x33_feedback11, loading 0.60); +11.5 kHz vs 08-04 -- above the usual few-kHz daily drift, flagged to user; mj0-mj1 splitting 2.7232 MHz (mj1 105.3051 MHz, FWHM 304 kHz, R^2=0.950, scan 20260805175514). 399 scan SKIPPED by user request. USER RAISED THE 556 RYDBERG PUSH-OUT BEAM POWER ~7x mid-session -> the whole 30 G set was re-amped from 0.15 to 0.10: amp 0.057 was far too weak (dip only 0.98->0.94, R^2 0.846, scan 20260805175959) and 0.10 reproduced the 08-03 line shape (dip 0.26-0.97, FWHM 150.2 kHz vs 154 kHz, center 143.5280 MHz vs 143.5267, scan 20260805180401) -- user approved 0.10. NOTE the ~7x is on the DEDICATED Rydberg-h path only (RydbergPushoutStep opens TTL556RydbergShutter + closes all 556 MOT shutters, driving Amp556RydbergMOTh alone), NOT the zero-field PushoutStep path (MOT shutters open, Amp556MOTX + Amp556RydbergMOTh together), which is why the mj=0/mj=1 scans at Amp 0.10 were unaffected (FWHM 45/304 kHz, both normal). 616 revival peak 233.9624 MHz (R^2 0.990, survival 0.28-0.97, scan 20260805180757) but FWHM 18.5 MHz vs the 07-20 reference ~4.5 MHz = power-broadened by the stronger 556 push. was 108.0168e6 (08-04: Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.953, FWHM 57.4 kHz, 219 shots, scan 20260804124801, 33x33_feedback11, loading 0.59); +5.8 kHz vs 08-03, within the usual daily drift; mj0-mj1 splitting 2.721 MHz (mj1 105.2959 MHz, FWHM 347 kHz, R^2=0.976, scan 20260804125248). 616 EXCLUDED by user request -> no 616-revival and no 556 Autler-Townes scan today (AT needs the revival peak as --eom616); 616 also read -242 MHz off / unlocked. SECOND CONSECUTIVE DAY of the same 399 outage: MCC USB DAC offline again at pre-flight (/health devices.dac=false, 399 lock engaged=true but dac_online=false, V frozen 6.1041, det +4678 MHz); DAC replugged, lock self-re-acquired to det 0.44 MHz / V 6.62, loading 0.54-0.59. Recurring 2 days running -> suspect a degrading DAC USB connection, not a one-off. was 108.0110e6 (08-03: Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.961, FWHM 49.1 kHz, 246 shots, scan 20260803101205, 33x33_feedback11, loading 0.57); +1.4 kHz vs 08-01, within linewidth; mj0-mj1 splitting 2.694 MHz (mj1 105.3172 MHz, FWHM 311 kHz, scan 20260803101739). Ran after a 399 outage: the MCC USB DAC driving the 399 wavemeter-PID piezo went offline overnight, its output collapsed and 399 parked +4.64 GHz off -> zero loading (0.005) + blank Orca frames (max 395 ADU); DAC replugged 10:10, lock re-acquired at V 6.24 / det +-3 MHz, loading back to 0.59. was 108.0096e6 (08-01: Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.950, FWHM 52.1 kHz, 202 shots, scan 20260801092042, 33x33_feedback11, loading 0.59); +11.8 kHz vs 07-30 -- above the usual few-kHz daily drift, flagged to user; mj0-mj1 splitting 2.720 MHz (mj1 105.2893 MHz, scan 20260801092459). was 107.9978e6 (07-30: Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.968, FWHM 47.6 kHz, 208 shots, scan 20260730140307, 33x33_feedback11, loading 0.61); +16.1 kHz vs 07-29 -- above the usual few-kHz daily drift, flagged to user; mj0-mj1 splitting 2.679 MHz (mj1 105.3184 MHz, scan 20260730140731). was 107.9817e6 (07-29: Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.959, FWHM 42.5 kHz, 236 shots, scan 20260729113144, 33x33_feedback11, loading 0.42); +1.7 kHz vs 07-28, within linewidth; first 200-shot run scan 20260729112300 gave 107.9883 MHz R^2=0.933, redone per auto-reps rule. was 107.9800e6 (07-28: Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.955, FWHM 60.0 kHz, 246 shots, scan 20260728105509, 33x33_feedback11, loading 0.58); -2.0 kHz vs 07-27, within linewidth. was 107.9820e6 (07-27 REDO (Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.964, FWHM 45.9 kHz, 208 shots, scan 20260727132532, 33x33_feedback11); -3.9 kHz vs the 12:07 fit, within linewidth. was 107.9859e6 (07-27 12:07, R^2=0.968, 460 shots, scan 20260727120744); 107.9618e6 (07-23, 33x33_feedback11); 107.9574e6 (07-22, 33x33_feedback11); 107.9501e6 (07-21, 33x33_feedback11); 107.9611e6 (07-20, 33x33_feedback11); 107.9253e6 (07-16, 33x33_feedback11); 107.9284e6 (07-14, 33x33_feedback11); 107.9054e6 (07-13, 33x33_feedback11); 107.8861e6 (07-05, 33x33_feedback9); 107.8762e6 (07-03); 107.8560e6 (07-02); 107.8448e6 (06-30); 107.8499e6 (06-29); 107.8478e6 (06-28, NEW LUT); 107.8389e6 (06-26); 107.8199e6 (06-23, 33x33_feedback9); 107.7753e6 (06-12, 47x47_uniform); 107.7673e6 (06-11); 107.7677e6 (06-10); 107.7552e6 (06-09); 107.7531e6 (06-09); 107.7573e6 (06-09); 107.7503e6 (06-08); 107.735e6 (06-05); 107.717e6
    # fit 2026-08-27 (Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.964, FWHM 69.3 kHz, 205 shots, scan 20260827111744, 33x33_feedback11, loading 0.555); -0.2 kHz vs the standing 108.1250e6 (the 08-18 fit) -- the flattest day-over-day yet, well inside the 69 kHz linewidth. mj0-mj1 splitting 2.3728 MHz (mj1 105.7521 MHz, FWHM 698.6 kHz, R^2=0.985, 105 shots, scan 20260827112208, stock 104.5:0.1:106.5 window, interior); -233.4 kHz vs 08-18's 2.6062, and the ~699 kHz mj=1 width is the same inhomogeneous trap-depth spread 08-11..08-18 documented (692.9 / 863.0 / 808.6 kHz), so the spread has still NOT come down. USER-DIRECTED SCOPE: the high-field set was run at 60 G, NOT the runbook 30 G. 399: Amp2 0.5 was skipped outright and the scan run at Amp2 1.0 per the gotcha-399-pushout-power-regression memory -- center 309.8315 MHz, FWHM 17.8 MHz, R^2=0.944, 87 shots, scan 20260827112701, survival 0.56-0.96 (a deep dip, so 1.0 is still the right push). The 2-LORENTZIAN FIT IS RESOLVED AGAIN after being degenerate on 08-06/08-07: peaks 299.9589 (FWHM 2.05 MHz) + 310.4780 MHz (FWHM 14.4 MHz), splitting 10.519 MHz, R^2=0.974 vs 0.944 single. *** 60 G SET (all new; the 30 G set was NOT run) ***: the scan's built-in push amp 0.15 is FIELD-INDEPENDENT and far too strong at 60 G -- it saturated the floor at exactly 0.00 for 7 consecutive points with a sagging left wing (0.57->0.48 pre-dip) and FWHM 551.2 kHz, i.e. no flat baseline and an unusable center (scan 20260827113243, ABORTED/discarded, user-spotted). USER CHOSE amp 0.08, which gave a clean fully-bracketed line: **60 G 556 dip = 119.0016 MHz** in Pushout.Green.Freq units (= 179.0016 low-field convention; the 50-80 G band auto-subtracts the 60 MHz HF AOM offset), FWHM 145.6 kHz, R^2=0.992, 156 shots, scan 20260827113932 -- FWHM matches the 30 G history (154 kHz @ 08-03), confirming 0.08 is unbroadened at this field. THE ~0.6 SURVIVAL CEILING IS THE SAME PUSH-INDEPENDENT HIGH-FIELD FLOOR OPENED 08-12, STILL UNEXPLAINED: both 60 G wings sit FLAT at 0.60-0.63 fully off-resonance while the same day's 0-field mj=0/mj=1 scans ran at their normal survival -- so it is not over-pushing (the flat wings are unpushed) and not loading (0.54-0.56 all day). Quote every 60 G contrast against a ~0.62 ceiling, not 0.95. **60 G 616 revival peak = 230.5074 MHz** (FWHM 7.68 MHz, R^2=0.990, 154 shots, scan 20260827114422, 556 parked on the measured 119.0016 via --green-freq-mhz, 308 amp 0.4), +907 kHz vs the 08-19 60 G value 229.6 and interior in the user-directed 205-255 window. **60 G 556 AUTLER-TOWNES: dips 118.4708 MHz (FWHM 284.2 kHz) + 119.6076 MHz (FWHM 329.5 kHz), SPLITTING 1.1368 +- 0.0135 MHz**, hand-seeded 2-peak R^2=0.931 vs 0.444 single, 330 shots, scan 20260827115704, 308 amp 0.4 parked on the measured revival 230.5074 MHz, 556 probe amp 0.08. TWO PROCEDURAL NOTES ON THAT AT RUN. (1) fit_spectrum.py --peaks 2 FAILED THE SAME WAY 08-12 RECORDS: its seed collapsed both components onto the left dip (118.4242/118.5113, splitting 0.087 MHz, R^2 0.455) even though the raw curve shows two obvious 1.14 MHz-separated dips; the numbers above come from a hand-seeded double-Lorentzian (fit_AT_60G_20260827115704.png in the scan dir). The --peaks 2 warning now has a second independent confirmation at a different field. (2) THE STOCK +-3 MHz AT WINDOW IS MOSTLY DEAD RANGE AT 60 G (user-spotted from the live dashboard): 41 of 61 points were flat baseline 0.60-0.67 with each dip resolved by only ~3-4 points; re-running at --half 0.8 --step 0.05 (33 pts over 118.2-119.8) roughly doubled the per-dip density at half the point count and converged in 330 shots. THE AT DOUBLET IS CENTRED ON THE BARE LINE AT 60 G, UNLIKE 20/30 G: midpoint 119.0392 MHz is only +37.6 kHz from the measured bare line 119.0016, versus the -286 kHz (20 G, 08-12) / -335..-354 kHz (30 G, 08-01/08-03) RED offsets. Both today's and 08-12's windows were centred on a MEASURED bare line via --center, so this is a genuine field-band difference in the 308-induced light shift of the dressed doublet, not a window-centering artifact -- a new datum for that open question. A first AT attempt (scan 20260827114815, 139 shots) was DISCARDED: the user reported 616 unlocked mid-run, so the 308 park was unreliable. After the user confirmed a re-lock, the kept run was polled with per-sample 616 wavemeter logging and 616 stayed put (n=28, min -22.1 max -9.2 mean -15.1 sd 3.13 MHz, zero excursions) -- logged as a STABILITY check only, not a lock verdict, per the 08-07 correction that the 616ULE scope is the only lock authority. Pre-flight: backend idle; 399 wavemeter PID engaged=true, det -2.17 MHz, DAC online, V 6.629; 556 det -3.14 MHz; /health devices all true except rp_pid (unused here). The dashboard SLM-camera endpoint STILL returns 503 "no data for camera_png yet" (10th+ consecutive day; cache empty, not a blank SLM), so the pattern was validated by the warm-up loading rate 0.554 + a cleanly bimodal array-averaged histogram (empty ~200 / atom ~210 ADU, threshold in the valley). The previous job was an ABORTED RearrangeSTIRAP scan, so the SLM held a rearrangement phase -- user chose 33x33_feedback11, written with z4=-5 (base sha 8f8e2345, same as 08-07..08-18). was 108.1250e6  (08-18 fit; full history in the entry below)
    c["Resonance556mj0Freq"] = 108.124830e6
    # 2026-08-26: two independent Spectrum399Scan fits today agree within errors --
    # 308.9 +- 0.4 MHz (job 1350, scan 20260826155010, 230 shots, R^2=0.974, loading 0.35) and
    # 308.3 +- 0.5 MHz (job 1353, scan 20260826162212, 145 shots, R^2=0.965, loading 0.46, run
    # on the new bias X 0.0344/Y 0.2533). Both are ~1.2-1.8 MHz below 08-07's 310.12 and well
    # below 08-06's 317.82; this line moves by MHz between sessions (it tracks trap depth), so
    # it is NOT a few-kHz/day drift like the 556 mj=0 line. Set to the consensus 308.6e6.
    # NOTE the quoted errors are ESTIMATOR SPREAD (the dip is asymmetric, so the center moves
    # with weighting), not the formal fit error, which is ~3x smaller and overstates precision.
    # The --peaks 2 doublet fit "resolved" 7.66 MHz on job 1353 (R^2 0.976 vs 0.954) but the raw
    # curve shows ONE asymmetric dip, not two minima -- the components land on the red shoulder
    # and the main body. Same seeding failure the AT doublet hit on 08-12; do not trust it.
    # This constant is the base for the 2D-MOT, Zeeman slower, blue MOT and 399 imaging (each
    # adds its own detuning), so changing it shifts ALL of them together.
    # was 310e6 (a held round anchor; MATLAB expConfig.m:124 still reads 310e6 with a reverted
    # %308.5e6 beside it, so 310 had been kept deliberately rather than tracked per-fit).
    # *** 399 IS ON THE TRANSFER-CAVITY LOCK (user-confirmed 2026-08-06, restated 2026-08-27). ***
    # The yb_monitor WAVEMETER PID IS NO LONGER THE LOCK AUTHORITY FOR 399. Do NOT gate a daily
    # pre-flight on 399 `lock.engaged` / `lock.dac_online`, and do NOT "fix" a healthy lock by
    # POSTing /wavemeter/lock/399/engage. 2026-08-27 read: engaged=false with detuning only
    # +1.97 MHz and the DAC online -- that pairing is now CORRECT (the laser is held by the
    # cavity), whereas under the old wavemeter-PID regime engaged=false meant a dropped servo
    # and the laser parked GIGAHERTZ off (07-23 +4539 MHz, 08-03 +4636 MHz, 08-04 +4678 MHz, all
    # with loading collapsed to ~0.005 and blank Orca frames). So the DISCRIMINATOR is the
    # DETUNING (and the loading rate), not the engaged flag: small |det| = fine regardless of
    # engaged; multi-GHz |det| = a real problem whatever the flag says. The 08-07 session wasted
    # an engage POST on exactly this (det was only +2.5 MHz), and 08-10 saw engaged=false with the
    # lock voltage RAILED at 8.0 V -- harmless for the lock, but a railed servo has no headroom.
    # NOTE the daily-system-scan runbook still instructs the old engaged/dac_online check and
    # still treats engaged=false as a fault -- it is STALE for 399; 556 is ULE-locked (no PID
    # block either) and 616 must be judged on the 616ULE SCOPE, not the wavemeter (08-07).
    c["Resonance399Freq"] = 308.6e6            # not magic; changes with trap depth

    # Init: 2D MOT & Zeeman, electric fields, SLM servo
    c["Init"] = {
        "TwoDMOT": {"FreqDetuning": -20e6, "Amp": 1},
        "Zeeman": {"FreqDetuning": -36.5e6, "Amp": 0.6},
        "EOM616": {"Freq": 252.07e6, "FreqOld": 252.07e6},
        "Electrodes": {"Vx": 0.0162, "Vy": 0.0008, "Vz": 0.0110},  # 2026-08-27 FULL DC-Stark E-field re-null AT 60 G (user-directed; every prior null was done at 30 G). Sequential Vx->Vy->Vz, each fit with the prior axes already nulled; all three vertices INTERIOR, all R2 >= 0.9986, and the three min-Stark centers agree to 222 kHz on a ~15 MHz-wide revival (230.345 / 230.393 / 230.567 MHz), which also matches the day's independently measured 60 G revival 230.3631 -- so the 556 park was right and the three axes are mutually consistent. Vx -0.0175 -> 0.0162 (20260827135024, R2=0.9986, a=1.800 MHz/V^2); Vy 0.0006 -> 0.0008 (20260827141130, R2=0.9991, a=1225); Vz 0.0092 -> 0.0110 (20260827143208, R2=0.9991, a=871.7). Curvatures reproduce the 07-16 30 G values (1.816 / 1215 / 877) to within a few percent even though the field doubled, as expected for a DC-Stark polarizability that does not care about B. Vy moved only 0.2 mV and Vz 1.8 mV (both were already near-nulled); Vx moved 33.7 mV, the only substantive change. Scramble was 0 for all three (the Stark/Revival scans set it; REQUIRED -- random EOM jumps unlock 616). All three ran 861/861 clean. EOM616 ramp slope was 6.0 (2x slower, changed this session) for these runs; cost only ~0.72 shots/s since the EOM grid is 1 MHz. NOTE StarkVxRevival616Scan does NOT apply the 50-80 G -60 MHz high-field AOM offset its siblings do, so --green-freq-mhz was passed ALREADY SHIFTED (119.0016); the low-field number parks the 556 60 MHz off -> flat ~80% survival, no revival (see bug-starkvx-scan-missing-hf-aom-offset). [prior 2026-07-16 (Vy 0.0006 -> 0.0008, scan 20260827141130, 861 shots, R2=0.9991, a=1225 MHz/V^2 = 680x steeper than Vx so swept narrow +-0.1 V, vertex INTERIOR, min-Stark center 230.393 MHz -- agrees with the Vx axis 230.345 to 48 kHz, two independent estimates of the same 308 line). Vz still the 07-16 value. Vx re-null AT 60 G (first Stark null done at 60 G, not 30 G -- user-directed): Vx -0.0175 -> 0.0162 (scan 20260827135024, 861 shots, R2=0.9986, a=1.800 MHz/V^2, vertex INTERIOR, min-Stark center 230.345 MHz which agrees with the day's measured 60 G revival 230.3631 -> confirms the 556 park was right). Vy/Vz NOT yet re-nulled at 60 G (still the 07-16 30 G values). NOTE StarkVxRevival616Scan does NOT apply the 50-80 G -60 MHz high-field AOM offset that Revival616Scan/RydbergSpectrum556Scan do, so --green-freq-mhz must be passed ALREADY SHIFTED (119.0016, not 179.0016); passing the low-field number parks the 556 60 MHz off and gives a flat ~80% survival with no revival (see bug-starkvx-scan-missing-hf-aom-offset). Also: the EOM616 ramp slope in RydbergPushoutSurvivalSeq was made 2x slower this session (3.0 -> 6.0) per user, which costs little here (~0.72 shots/s) since the EOM steps are 1 MHz apart. [prior 2026-07-16 full DC-Stark E-field re-null (new 616/308 line ~234 MHz; revival relocated from the old 282). Sequential Vx->Vy->Vz, each fit after the prior nulled; all parabola vertices INTERIOR, min-Stark centers agree 234.26/234.23/234.31 MHz. Vx -0.042->-0.0175 (20260716162632, R2=0.9999, a=1.816 MHz/V^2); Vy 0.0066->0.0006 (20260716171645, R2=0.9997, a=1215); Vz 0.0059->0.0092 (20260716173232, R2=0.9992, a=877). Scramble MUST be OFF for the 616-EOM sweep (random EOM jumps unlock 616). [prior 06-28: Vx -0.042/Vy 0.0066/Vz 0.0059; old min-Stark ~282.08 MHz]
        "VSLMServo": 3.7,                        # 112 sites at 6A at 30dB
        "VIonizationSet5to8": 0 # DC Ionization voltage for electrodes 5-8 (V) must < 5V
    }

    # BlueMOT
    c["BlueMOT"] = {
        "BFieldGradient": 30,                  # G/cm maximum
        "BiasCoilCurrent": {"Ryd": 3, "X": 0.1, "Y": 0, "Z": 0},
        "FreqDetuning": -44e6,                 # fast-loading opt 2026-06-05; was -40e6
        "Amp": 0.6,
        "LoadingTime": 500e-3,                 # fast-loading opt 2026-06-05 (loading saturates ~0.21); was 500e-3
        "Img1PIDSet": 0.57,  
        "Img2PIDSet": 0.41
    }

    # GreenMOT
    c["GreenMOT"] = {
        "BFieldRampTime": 100e-6,              # blue->green B-field ramp
        "BFieldGradient": 3,
        # 2026-08-26 2-D bias map (LACScan job 1351, scan 20260826155511, 33x33_feedback11):
        # X 0.030-0.040 x Y 0.24-0.30, 10x10, PARTIAL run (289/2000 shots, 1-2 shots/cell -- the
        # job errored early). X is razor-sharp (27x swing across the row, marginal peak 0.0344,
        # parabola vertex 0.03408, both neighbours lower so the peak is interior); Y is broad
        # (2.6x swing, marginal peak 0.2533, vertex 0.2545). Rounded to the grid rather than the
        # sub-grid vertex given 1-2 shots/cell. Peak cell load 0.462 vs 0.399 at the previous
        # (0.033, 0.273) = 1.16x, or 1.12x on the robust 3x3 average -- the old X already sat on
        # the ridge, so most of the gain is the Y move. RE-RUN THE MAP AT FULL REPS to confirm.
        # was X 0.033 / Y 0.273; fast-loading opt 2026-06-05: X was 0.039, Y was 0.27
        # 2026-08-27 GreenMOT alignment re-null (33x33_feedback11, LoadingTime work-point 0.28 s).
        # X 0.0344 -> 0.0353, Y 0.2533 -> 0.262. Base (not ByPattern) because this is the physical
        # MOT position relative to the array, not a per-array quantity.
        # WHY: loading had fallen ~34% during the day (0.478 -> 0.315 at the same 0.5 s LoadingTime)
        # with detection provably unchanged (pedestal 200.1 both, atom signal even slightly higher
        # 12.75 vs 12.28 ADU), i.e. real atom number. Cause: the X resonance is only ~4 mA WIDE
        # (dead 0.001 outside 0.0340-0.0360; 0.0320 -> 0.010, 0.0380 -> 0.013) and the in-use 0.0344
        # sat on its LOW-SIDE FLANK -- the MOT had walked off it.
        # X: coarse 2 mA grid (20260827151415, 400 shots) pointed at 0.0360, but a 0.5 mA refine
        # (20260827152139, 80 shots, 10/pt) put the true INTERIOR peak at 0.0350-0.0355 (0.506+-0.023
        # / 0.494+-0.026, within 1 SEM) with a hard rolloff above (0.0370 -> 0.241, 0.0380 -> 0.024)
        # -- 0.0360 was still the upper flank, the coarse grid could not resolve it. The x-GRADIENT
        # nulls in the SAME place (corr_x crosses zero between 0.0350 +0.13 and 0.0355 -0.09) and CV
        # bottoms at 0.34, so peak rate and minimum gradient coincide -> 0.0353 = the zero crossing
        # AND the plateau centre (drift margin on a ~4 mA resonance).
        # Y: broad in RATE (plateau 0.250-0.265, 0.455-0.478 all within ~1 SEM) but corr_y sweeps
        # MONOTONICALLY -0.779 -> +0.778 and NULLS at ~0.2635 (20260827152326, 88 shots) = the
        # vertical gradient; 0.262 takes that null while staying inside the rate plateau.
        # VERIFY 40 shots (20260827152706) at X 0.0353 / Y 0.262 / LoadingTime 0.28 vs the old bias
        # at a comparable 0.295 s: load 0.480+-0.014 vs 0.321+-0.043, CV 0.25 vs 0.67, corr_x -0.205
        # vs +0.554, shot-to-shot std 0.087 vs 0.122, and DEAD SITES 0/1068 vs 136/1068 (per-site p5
        # 0.275 vs 0.000). Both runbook axes (rate AND uniformity) improved together.
        # WATCH: ~+-2 mA tolerance on X (0.0370 already halves loading) -- this axis drifts, and
        # today's 34% loss was exactly that. Re-check X before blaming anything else for low loading.
        # 2026-08-27 ROUND 4/5 -- JOINT fine X x Y grid CORRECTS the Y above (0.262 -> 0.244).
        # The 0.262 came from a 1-D Y sweep at fixed X; a joint grid (20260827160412, aborted at
        # 67/486 but already monotonic toward low Y, best cells pinned at the 0.246 grid EDGE) then
        # Y shifted DOWN (20260827160629, 240 shots, 99/99 cells, ~2/cell) put the optimum well below.
        # SEM-weighted rotated 2-D Gaussian: X0 = 0.03526 +- 0.00003, Y0 = 0.2441 +- 0.0010, R2 0.909,
        # ridge tilt only +1.0 deg -- so X and Y are effectively NOT coupled and the 1-D/2-D
        # disagreement was Y's flatness + low stats, not a tilted ridge. Plateau (9 cells within 0.02
        # of the 0.610 max) spans X 0.0350-0.0360 x Y 0.234-0.250; corr_x nulls at X ~ 0.0354
        # consistently across Y rows. Three estimates agree on X (fit 0.03526, gradient-null 0.0354,
        # plateau centre 0.0355) -> X stays 0.0353; Y -> 0.244 = the fitted centre, inside the
        # plateau with ~10 mA margin either side (Y is broad: sigma_Y 0.033 vs sigma_X 0.0018).
        # Peak loading on this grid reached 0.61 at the 0.28 s work-point (vs 0.480 at Y 0.262).
        "BiasCoilCurrent": {"Ryd": 0, "X": 0.0353, "Y": 0.244, "Z": 0.17},  # was X 0.0344 / Y 0.2533 (06-05); Y 0.262 (08-27 1-D) superseded
        # fast-loading opt 2026-06-05: HandoverTime was 30e-3
        "PowerBroaden": {"HandoverTime": 15e-3, "FreqDetuning": 0.7e6, "Amp": 0.8},
        # fast-loading opt 2026-06-05: HoldTime was 200e-3, Amp was 0.2
        "CoolDown": {"RampdownTime": 50e-3, "HoldTime": 200e-3,
                     "FreqDetuning": 0.35e6, "Amp": 0.25},
    }

    # Absorption imaging
    # There is only onebeam for AbsImag and it requires flipping a mirror mount
    c["AbsImag"] = {"TOF": 0, "ExposureTime": 50e-6, "BetweenImagsTime": 50e-3,
                    "Freq": 315e6, "Amp": 0.5}
    # SLM
    c["SLM"] = {
        "AOM": {"Freq": 120e6, "Amp": 0.55},
        "VServo": None,                        # cross-ref -> Init.VSLMServo (set below)
        "Modulation": {"Time": 10e-3, "Freq": 100e3, "Amp": 0},
        # Every-scan default loading pattern. RUNTIME-ONLY: consumed by the pyctrl run loop
        # (_loading_defaults in YbExptCtrl/slm_runtime.py), never read by a sequence -> no
        # serialize() byte effect, so it does NOT touch the byte oracles. expConfig.py is the
        # live source of truth and hot-reloads per job, so the default array changes WITHOUT a
        # backend restart. The slm_runtime.py module constants (DEFAULT_LOADING_PATTERN_PHASE /
        # DEFAULT_LOADING_DEFOCUS / ALL_SCANS_LOAD_PATTERN) are now only the fallback when this
        # "Loading" key is absent (e.g. a bare JSON snapshot). After editing this, regenerate
        # the config drift oracle: ``python pyctrl/tools/capture_config_reference.py``.
        #   DefaultPhase        - server-side WGS phase written when a scan declares no pattern
        #   Defocus             - ANSI z4 loading defocus (rad); fixed loading plane (camera-set)
        #   AllScansLoadPattern - when True, EVERY no-pattern scan writes DefaultPhase + holds
        #                         the SLM lock + detects with that pattern's per-pattern thresholds
        "Loading": {
            "DefaultPhase": "phase/47x47_feedbackwarm3.pt",
            "Defocus": -5.0,
            "AllScansLoadPattern": True,
        },
    }

    # LAC
    c["LAC"] = {
        "FreqDetuning": 0.11e6, "Amp": 0.16, "Time": 20e-3, "DeadTime": 10e-3,
        "BlueLAC": {
            "FreqDetuning": -3.8e6, "Amp": 0.17, "Time": 500e-3, "DeadTime": 30e-3,
            "BiasCoilCurrent": {"Ryd": 0},
            "Resonance556mj0Freq": None,       # cross-ref -> Resonance556mj0Freq (set below)
            "X": {"FreqDetuning": 0.22 * 1e6, "Amp": 0.04},
        },
    }

    # Imag399
    c["Imag399"] = {
        "FreqDetuning": -5e6, "Amp1": 1, "Amp2": 1, # Now we use the VIMG1/2Set to control the imaging power
        "ExposureTime": None,                  # cross-ref -> Orca.ExposureTime (set below)
        "Cool556": {
            "FreqDetuning": 0.18e6, "Amp": 0.2,
            # cooling opt 2026-06-05 (CoolingScan, jointly converged at imaging amp 0.18):
            #   was X {0.12e6, 0.18}, h {0.14e6, 0.14}
            "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
            "h": {"FreqDetuning": 0.16e6, "Amp": 0.13},
        },
        # StrobeImag399Step: per super-cycle = BEAM 1 pulse -> BEAM 2 pulse -> 556 recool, within ONE
        # camera exposure (arXiv:2507.01011 scheme, our regime). The two COUNTER-PROPAGATING 399 imaging
        # beams (AmpAbsImag=img1, Amp399Imag2=img2) ALTERNATE -- each on for BeamPulseTime, the other off,
        # 556 off -- so a pair cancels net photon recoil ("400 ns pulses ... mitigate momentum transfer
        # from a single beam", here 1 us). Then a recool window (399 off, 556 on). period = 2*BeamPulseTime
        # + RecoolTime; n_cycles = round(Orca.ExposureTime/period) -> the train auto-fills the camera frame
        # (choose the collect window via Orca.ExposureTime, base or a ByPattern overlay).
        #
        # BeamPulseTime/RecoolTime set the LOOP structure -> FIXED scalars, NEVER a .scan() axis; sweep
        # them across ROUNDS (archive/tools/strobe_imaging_round.py --pulse-time). BeamPulseTime 1 us ~ AOM rise
        # -> partial pulse; verify on scope (trigger on the first PD pulse -- no scope-sync TTL wired).
        # WARN: small RecoolTime -> many cycles -> big sequence; validate at short Orca.ExposureTime first.
        #
        # Cool556 here is a SEPARATE recool set from the during-imaging Imag399.Cool556: with the 399 OFF
        # during recool there is no light shift -> the optimum detuning/amp differs. Seeded from
        # Imag399.Cool556; retune via archive/tools/strobe_imaging_round.py cool mode.
        # PulsesPerBurst (N): per super-cycle, fire N alternating beam1/beam2 SHORT pulses (the image
        # burst -- short pulses keep the per-beam momentum kick + heating low) THEN one recool window.
        # period = 2*N*BeamPulseTime + RecoolTime; duty = 2*N*BeamPulseTime/period. Raise N to raise duty
        # (more photons) WITHOUT lengthening the single-beam pulse. WARN: events ~ cycles*N -> a high-N,
        # short-pulse train over a 100 ms exposure is a LARGE sequence (~1e5 pulse events); watch the
        # engine. N/BeamPulseTime/RecoolTime are loop counts -> FIXED scalars, never a .scan() axis.
        "Strobe": {
            "BeamPulseTime": 1e-6, "RecoolTime": 5e-6, "PulsesPerBurst": 5,
            "Cool556": {
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.13},
            },
        },
    }

    # Cool556
    c["Cool556"] = {
        "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
        # RNR cooling opt 2026-06-05 (CoolingScan_RNR, release-recapture 50us, interleaved X<->h
        # joint converged, survival 0.28->0.31): was X {0.11e6, 0.16}, h {0.11e6, 0.14}
        "X": {"FreqDetuning": 0.135e6, "Amp": 0.13},
        "h": {"FreqDetuning": 0.13e6, "Amp": 0.12},
    }

    # RearrangeCool556 -- the post-rearrangement recool that precedes STIRAPPushoutStep
    # (RearrangeCool556hXStep). Same channels/shape as Cool556hXStep but its OWN block, so the
    # STIRAP recool can be optimized independently of the RNR release-recapture Cool556 above.
    # 2026-08-05/06: the STIRAP science block used to recool via Cool556Step, which drives BOTH beams
    # from the top-level Cool556.FreqDetuning/Amp (0.14 MHz / 0.08) instead of the per-beam X/h
    # sub-blocks the 07-20 RNR campaign optimized. Switching to the per-beam values lifted mid->final
    # survival 0.341 -> 0.514 (20260805_231444 vs 20260806_000351; every gap point +6..+11 sigma),
    # identifying the recool as the dominant limiter. Seeded from the base Cool556 X/h so behaviour is
    # unchanged where no overlay exists; the STIRAP atoms arrive hotter than the RNR ones (SLM
    # transport with RearrCoolAmp = 0, plus two 399 exposures before the pushout), so expect this to
    # want a longer/stronger recool than Cool556 -- that is what the scan knobs are for.
    c["RearrangeCool556"] = {
        "Time": 5e-3,
        "X": {"FreqDetuning": 0.135e6, "Amp": 0.13},
        "h": {"FreqDetuning": 0.13e6, "Amp": 0.12},
    }

    # Pushout
    c["Pushout"] = {
        "Time": 10e-3,
        "Green": {"Freq": 118.1e6, "Amp": 0},
        "Blue": {"Freq": 320e6, "Amp1": 0, "Amp2": 0},  # Amp1 and Amp2 for pushout (matches Imag399.Amp1/Amp2)
        "Ryd308": {"Freq": 200e6, "Amp": 0},
        "Ionization": {"Amp": 0},              # 369 ionization-beam amp default (RydbergPushoutStep)
        # Ryd bias coil current (A) during push-out. RydbergPushoutSurvivalSeq reads this to pick
        # the low-field (< 31 A) vs high-field (60-130 A) push-out step, and resolves the fallback
        # EAGERLY -- so it must exist here even though the scan always overrides it. 0 = low field,
        # matching the literal `g.BiasCoilCurrent.Ryd(0)` in both push-out steps.
        "BiasCoilCurrent": {"Ryd": 0},
        "STIRAP": {"delay": 1e-6, "reverse_delay": 1e-6, "gap": 10e-6},
        "MRabi": {"Freq": 4000, "Gain": 0},
        "Ramsey": {"Phase": 0},
    }

    # 616 AOM diverted/idle amplitude (RydbergPushoutStep restores this after pushout)
    c["AOM616Divert"] = {"Amp": 0.15}

    # SLM trap modulation
    c["SLMTrapModulation"] = {
        "Time": 5e-3, "Freq": 100e3, "AmpFactor": 0.5,
        "lowerTrapDepth": {"Vservo": 0, "Time": 3e-3},
    }

    # Rearrangement
    c["SLMRearrange"] = {"Time": 100e-3}

    # AWG defaults (Siglent SDG6X) -- TWO-CHANNEL SWITCH SCHEME.
    # Each box carries box-level hardware (resource_address, num_points, sample_rate_MHz) plus TWO
    # independent per-channel waveform dicts, Ch1 (-> SDG C1, "forward") and Ch2 (-> C2, "reverse").
    # AWGManager arms each channel as a single-cycle EXTERNALLY-triggered burst (TIME,1); an external
    # RF switch (e.g. TTL556RydAWGChSwitch = FPGA1/TTL16) selects which channel reaches the AOM, and
    # the 50 us STIRAP gap lives OUTSIDE the arb (FPGA/switch-controlled) so each short DDS arb keeps
    # the carrier well above Nyquist. Define pulses per channel in a scan, e.g.:
    #     g().AWG.AWG556.Ch1.shape = "rise_gaussian"
    #     g().AWG.AWG556.Ch1.pulse_width_us.scan(1, np.linspace(...))
    #     g().AWG.AWG556.Ch2.shape = "fall_gaussian"
    # Per-channel waveform fields (fall back to these defaults): shape / carrier_freq_MHz /
    # pulse_width_us / smooth_width_us / steepness / amplitude_scale / max_amplitude_vpp, plus
    # stirap_gap / f_delay / r_delay for double_half_gaussian_* and optional trig_delay_us (per-
    # channel burst DLAY, s->us; 0 = fire on the edge, switch does the timing).
    # shape gallery: pyctrl/tmp/pulse_10_examples.png ; pulse math: devices/sigilent_awg/pulse_waveform.py
    _AWG_CH_DEFAULTS_556 = {
        "carrier_freq_MHz": 143.4, "pulse_width_us": 1.437, "steepness": 3.5,
        "amplitude_scale": 1.0, "smooth_width_us": 0.0, "max_amplitude_vpp": 15,
        "trig_delay_us": 1.5,  # per-channel burst DLAY (us); >= the box's ~1.435us floor so it is
                               # HONORED (not clamped) -> deterministic edge->output latency. The
                               # seq must add this to its post-edge waits (edge + DLAY + 3*pw).
    }
    c["AWG556"] = {
        "resource_address": "USB0::62700::4353::SDG6XFCC900309::0::INSTR",
        "num_points": 10000,
        "sample_rate_MHz": 2500,  # floor num_points to this MSa/s: keeps carrier Nyquist for short arbs
        "Ch1": dict(_AWG_CH_DEFAULTS_556, channel="C1", shape="rise_gaussian"),  # forward
        "Ch2": dict(_AWG_CH_DEFAULTS_556, channel="C2", shape="fall_gaussian"),  # reverse
    }
    _AWG_CH_DEFAULTS_308 = {
        "carrier_freq_MHz": 200, "pulse_width_us": 1.463, "steepness": 3.5,
        "amplitude_scale": 1.0, "smooth_width_us": 0.0, "max_amplitude_vpp": 5.5,
        "trig_delay_us": 1.5,  # honored burst DLAY (us); >= ~1.435us floor. See AWG556 note.
    }
    c["AWG308"] = {
        "resource_address": "USB0::62700::4353::SDG6XFCD801391::0::INSTR",
        "num_points": 10000,
        "sample_rate_MHz": 2500,
        "Ch1": dict(_AWG_CH_DEFAULTS_308, channel="C1", shape="rise_gaussian"),  # forward
        "Ch2": dict(_AWG_CH_DEFAULTS_308, channel="C2", shape="fall_gaussian"),  # reverse
    }

    # QICK FPGA_AWG (RFSoC4x2) microwave AWG defaults -- OUT-OF-BAND (not in the byte blob), TTL-
    # triggered on TTLQickTrig = FPGA1/TTL14. A scan opts in with g().runp().QICK = True and declares
    # the microwave sequence via g().QICK.* (mirrors the Siglent g().AWG.<name>.* convention). The run
    # loop (YbExptCtrl/awg_runtime.py -> devices/qick_awg) builds one program per unique swept point,
    # batch-uploads them ALL once at scan start, then ARMS the active one before every shot (the board
    # is one-shot: each TTL fires the armed program once, so it must be re-armed per shot). Params:
    #   template   -- which pulse sequence to build: "Sine" | "Rabi" | "Ramsey" | "Echo" (see
    #                 devices/qick_awg/templates.py). Fixed per scan; the swept scalar mints the programs.
    #   freq       -- carrier / DDS frequency, MHz (0..6000 usable; >3000 aliases past Nyquist).
    #   gain       -- DAC gain, -2**15..2**15-1. DEFAULT 0 = silent (safe; set nonzero to emit).
    #   rabi_freq  -- microwave Rabi frequency, Hz. Derives the pulse lengths: t_pi2 = 1/(4*rabi_freq),
    #                 t_pi = 2*t_pi2 (author sets the physics, lengths follow).
    #   phase      -- final-pulse (Pi2_Phase) phase, DEGREES (server-native; NO rad->deg conversion).
    #   wait_time  -- Ramsey/Echo free-evolution time, s (sweepable). Split into loop(N,[Wait]) chunks
    #                 so each Wait pulse fits the 16-bit HW length register (see templates._split_duration).
    #   drive_time -- Rabi drive time, s (sweepable). Chunked the same way when it exceeds the cap.
    # Per-scan override any field via g().QICK.<field>; a swept field (e.g. wait_time.scan(1)) becomes the
    # per-shot program-selection key. Example (Ramsey wait scan):
    #     g().QICK.template = "Ramsey"; g().QICK.gain = 3000; g().QICK.rabi_freq = 7.187e6
    #     g().QICK.wait_time.scan(1, np.linspace(...)); g().runp().QICK = True
    c["QICK"] = {
        "host": "192.168.0.72", "port": 1234,
        "template":   "Ramsey",
        "freq":       10863.04,
        "gain":       0,
        "rabi_freq":  7.187e6,
        "phase":      0.0,
        "wait_time":  1e-6,
        "drive_time": 500e-9,
        "duration":   1e-6,          # Sine template: single-tone length, s (sweepable)
    }

    # Per-TTL-channel hardware timing managers (ExpSeq.add_ttl_mgr -> serialized into the byte blob
    # -> libnacs/FPGA shifts every edge on that channel). Consumed by the run loop's compile_point
    # (YbExptCtrl/engine_run.py), like LineTrigger; a channel absent here (or all-zero) adds NOTHING to
    # the bytes (byte-identical), and a manager only serializes if its channel is actually USED in
    # the sequence. Per-channel fields (all times in SECONDS):
    #   on_delay / off_delay -- fire the rising / falling edge THIS MUCH EARLIER (advance). The
    #     engine does t = max(t - on_delay, 0) (see libnacs zynq/bc_gen.cpp). Use this to
    #     PRE-COMPENSATE a downstream hardware latency so the effect lands at the nominal seq time.
    #   skip_time -- drop off-intervals shorter than this.  min_time -- stretch on-times to >= this.
    #   off_val   -- the channel's idle level (bool).
    # AWG gate use (2026-07-15): the 556/308 Rydberg AWG gates are EDGE triggers (NCYC=1); the box's
    # own trig_delay_us (1.5 us) fires the waveform 1.5 us AFTER the trigger. Setting on_delay =
    # trig_delay_us fires the gate 1.5 us early so the light lands at the nominal sequence time; the
    # step then drops the matching +DLAY from its post-gate wait. Set to 0 to disable (leave the
    # step's hand-added DLAY doing the accounting instead). Per-scan override: runp().TTLManagers.
    c["TTLManagers"] = {
        "TTL556RydAWG": {"on_delay": 0e-6, "off_delay": 0.0, "skip_time": 0.0,
                         "min_time": 0.1e-6, "off_val": False},   # = AWG556 trig_delay_us
        "TTL308RydAWG": {"on_delay": 0e-6, "off_delay": 0.0, "skip_time": 0.0,
                         "min_time": 0.1e-6, "off_val": False},   # = AWG308 trig_delay_us
        'TTL556RydAWGSwitch': {"on_delay": 1.05e-6, "off_delay": 1.05e-6, "skip_time": 0.0,
                         "min_time": 0, "off_val": False},   # = AWG556 trig_delay_us
        'TTL308RydAWGSwitch': {"on_delay": 0.85e-6, "off_delay": 0.85e-6, "skip_time": 0.0,
                         "min_time": 0, "off_val": False},   # = AWG308 trig_delay_us
        'TTLQickTrig': {"on_delay": 0e-6, "off_delay": 0e-6, "skip_time": 0.0,
                    "min_time": 0, "off_val": False},   # = AWG308 trig_delay_us
    }

    # 60 Hz AC-line trigger. When enabled, the FPGA waits for an edge on a TTL INPUT line at the
    # start of each basic sequence (ExpSeq.enable_global_wait_trigger -> a version-2 ZYNQZYNQ
    # block -> libnacs emits a WaitTrigger bytecode op), so every shot begins at the same mains
    # phase (B-field stability). Consumed ONLY by the run loop's compile_point
    # (YbExptCtrl/engine_run.py) -- it is never read by a sequence step, so it does NOT enter the
    # serialized bytes (the MATLAB byte oracles are unaffected; only this config snapshot needs
    # re-capturing). Per-scan override: runp().LineTriggerEnable / LineTriggerChannel /
    # LineTriggerRaise / LineTriggerTimeout.
    #   Channel: RAW FPGA TTL line number of the line-sync input (same numbering as the TTL
    #            outputs, e.g. FPGA1/TTL14 -> 14) -- NOT a channel alias. Must be 0..max_ttl_chn
    #            (config.yml), must NOT equal start_ttl_chn, and must NOT be driven as an output.
    #            None = unset -> the runner SKIPS enabling (and logs once) rather than guess a
    #            line; SET it to your physical line-sync input to activate for every scan.
    #   Raise:   True = wait for a rising edge, False = falling edge. This is the PHYSICAL edge you
    #            get -- but only because pyctrl compensates for a firmware inversion: molecube2
    #            maps the flag backwards relative to the gateware, so the run loop sends its
    #            COMPLEMENT (engine_run._MOLECUBE2_TRIG_EDGE_INVERTED -- read that note before
    #            touching either side; "fixing" molecube2 without clearing that flag double-inverts
    #            and silently restores the bug). Verified 2026-08-16 end to end on scope
    #            192.168.0.27: with the compensation in place, Raise=True fires on the line's
    #            rising edge, Raise=False on the falling edge, each within one 20 us sample.
    #   Timeout: seconds. FPGA clock is 100 MHz and the bytecode timeout field is 24-bit, so the
    #            max is ~0.168 s; ~0.02 s = one 60 Hz period + margin (catches the next edge, then
    #            proceeds if the signal is absent -- it does not hang the shot).
    c["LineTrigger"] = {
        "Enable": True,
        "Device": "FPGA1",
        "Channel": 0,                          # trigger-input MUX index (WaitTrigger `chn:8`). 
        # 0 = TTL in 0 = TTL channel 24 (where the 60Hz is wired), 
        # 1 = TTL 52 / bd0-24
        # 2 = SMA12 (FMC2 clock1p)
        # 3 = SMA00 (FMC1 clock0p)
        # clock SMA10 (FMC2 clock0p)
        # TTLout24 SMA13 (FMC2 clock1n)
        # TTLout52 SMA04 (FMC 1 la32p)
        "Raise": True,                         # True = rising edge, False = falling (the run loop
                                               # inverts it for the firmware quirk; see above)
        "Timeout": 0.02,                       # seconds (~one 60 Hz period + margin)
    }

    # ---- per-pattern overrides (RUNTIME-ONLY; see the "Per-pattern config overlay" section) ----
    # Map an SLM loading-pattern NAME (the phase-file basename) to a SPARSE override of the leaves
    # above that differ for that array. Any leaf left unset falls back to the base value above,
    # then a swept/manual g() value wins over both. 47x47_uniform is SEEDED below with the CURRENT
    # base cooling/imaging defaults, so the overlay is byte-identical until you tune these from a
    # 47x47 scan -- then edit the numbers here. (Trap depth Init.VSLMServo is intentionally NOT
    # seeded -> 47x47 uses the base value; add "Init": {"VSLMServo": <v>} to tune it per array.)
    c["ByPattern"] = {
        "47x47_uniform": {
            # Initialize the SLM servo to 3.7
            "Init": {
                "VSLMServo": 3.5
            },
            # imaging (399) + cooling-during-imaging (556 X/h).
            # 556 X/h cooling-during-imaging re-optimized for 47x47_uniform 2026-06-11 (CoolingScan,
            # real imaging amp Blue 0.18 + 1 s hold, interleaved X<->h to the joint fixed point;
            # survival ~0.915). Only h.Amp moved (0.13 -> 0.14); X (det +0.16, amp 0.20) and h det
            # +0.16 confirmed. ExposureTime omitted (cross-ref to Orca.ExposureTime, re-resolved).
            # 2026-06-12: briefly tried Imag399.Amp 0.2 (larger histogram split, +14% SNR) but it
            # cost survival 0.905 -> ~0.80 even after re-tuning cooling (X->0.22, h->0.16) -- reverted
            # to 0.18 and these 0.18-optimum X/h values (full data in Notion 06/12).
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.18, "Amp2": 0.18,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                },
            },
            # RNR / release-recapture cooling (556) -- re-optimized for 47x47_uniform 2026-06-11
            # (CoolingScan_RNR, 50us release, interleaved X<->h coordinate ascent to the joint
            # fixed point; survival ~0.29 at 50us, loading ~0.58, broad flat plateau).
            # Was X {0.135e6, 0.13}, h {0.13e6, 0.12} (seeded from the prior array).
            # 2026-06-12: h FreqDetuning 0.14e6 -> 0.12e6. Re-run at the more sensitive 30us
            # release (50us was washed out, spread ~0.06; 30us spread ~0.22, ~4x contrast);
            # interleaved X<->h converged, survival ~0.54 at 30us. X {0.13e6, 0.14} and h.Amp
            # 0.12 confirmed; only h det moved one fine step (0.14->0.12, ~4.6 SEM, ~0.02 gain).
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.13e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.12e6, "Amp": 0.12},
            },
            "LAC": {
                "FreqDetuning": 0.11e6, "Amp": 0.16, "Time": 10e-3, "DeadTime": 10e-3,
                "BlueLAC": {
                    "FreqDetuning": -3.8e6, "Amp": 0.17, "Time": 500e-3, "DeadTime": 30e-3,
                    "BiasCoilCurrent": {"Ryd": 0},
                    "Resonance556mj0Freq": None,       # cross-ref -> Resonance556mj0Freq (set below)
                    "X": {"FreqDetuning": 0.22 * 1e6, "Amp": 0.04},
                }
            }
        },
        "tri_3013_camfb": {
            "Orca": {"ExposureTime": 0.1},  # 100 ms COMMITTED 2026-07-16 evening (R216-R223).
            # The railed 399 servo (PD gain too low) leaves the floor light UNREGULATED -- it had
            # drifted hot by tonight: the 50 ms committed config re-measured 0.798 survival
            # (R223 _231231; overnight it was 0.955). At 100 ms x DDS amps 0.14/0.08 the
            # INSTANTANEOUS power drops ~2.7x (shallow traps boil at high intensity; photon count
            # ~matched) -> survival 0.9585, d' 3.85 (R222 _230657). 50 ms full-floor = 0.80. Until
            # the PD-gain fix, the attenuated long exposure is also the drift-safer config.
            "Init": {"VSLMServo": 3.5},
            # 2026-07-15 green-MOT POSITION scan on tri_3013_camfb @ committed LAC (0.22/0.10) + 1s
            # load: GreenMOT.BiasCoilCurrent X 0.037-0.041 x Y 0.24-0.28 (TweezerLoadingSeq). PEAK
            # loading 0.600 at X 0.0383 / Y 0.255; broad flat top X 0.038-0.040 x Y 0.245-0.265
            # (0.56-0.60). camfb loads ~2x better than v2 (0.60 vs 0.27) -- the ARRAY quality, not
            # position. Base 0.0388/0.265 already on the plateau; committed X 0.039 (mid-plateau) /
            # Y 0.255 (peak) as a PER-PATTERN overlay (base left global). scan _141456.
            "GreenMOT": {"BiasCoilCurrent": {"X": 0.039, "Y": 0.255}},
            # 2026-07-15 LAC drive-plane opt: 2-D LAC.Amp x FreqDetuning (TweezerLoadingSeq,
            # loading-rate). Seed 0.2/0.11 sat on the detuning RISING EDGE (loading ~0.12).
            # Moving det up (~0.20 MHz) + amp down (0.10) MORE THAN DOUBLED loading -> broad flat
            # plateau ~0.25-0.29 across Amp 0.04-0.16 x det 0.14-0.30 (no knife-edge). Committed
            # central-plateau Amp 0.10 / det 0.22 MHz. BlueMOT.LoadingTime 1.0 s (per user; note:
            # 1s vs default gave NO real gain -- mean 0.253 vs 0.240, loading is NOT blue-delivery-
            # limited; ~0.28 ceiling is depth/geometry). scans _132218 / _132755 / _133549(@1s).
            # 2026-07-16 ENHANCED LOADING (blue-detuned LAC; BlueTweezerLoadingSeq): campaign found
            # BlueLAC det -3.0 MHz (base -3.8) at amp 0.17 / 0.5 s = the dose-ridge peak. Loading
            # 0.611 (normal LAC baseline) -> 0.71 self-thresholded (~2150/3013 atoms; target 2200).
            # Sharp det structure (-3.6 WORST 0.53, -3.0 best 0.70); amp x time = dose ridge, overdose
            # collapses (0.29/1.0s -> 0.15). Red clean-up 10-30 ms equivalent; 1 ms hurts. scans
            # _050131 (det) / _050549 (amp x time) / _051552 (red) / _051742 (60-shot confirm).
            "LAC": {"FreqDetuning": 0.22e6, "Amp": 0.10, "Time": 30e-3,   # red LAC unchanged
                    "BlueLAC": {"FreqDetuning": -3.0e6}},                  # enhanced-loading det
            # 2026-07-15 imaging-power opt (PID scheme; DDS Imag399.Amp1/Amp2=1). 2-D
            # Img1PIDSet x Img2PIDSet on ImagingPushoutSurvivalSeq @ 50 ms, 0-pushout, z4=-2.
            # Survival collapses as either setpoint rises; flat plateau at low Img2 (0.06-0.20),
            # optimum low corner. Picked Img1 0.40 / Img2 0.10 (mid-plateau, off the low edge):
            # survival ~0.34, d' ~3.0. d' ceiling is depth/loading-limited (~14% fill), NOT
            # imaging-power -- cooling / 532 depth are the next levers. (Was 1.3/0.25, copied
            # from feedback11 @ 25 ms.) scans 20260715_111735 (7x7) / _112905 (Img2-low ext).
            # 2026-07-18 POST-ND imaging-power re-opt (amps=1, 100 ms, 0-pushout): 1-D beam-1 (r1) +
            # beam-2 (r2) 0..3 -> 2-D (r3, edge-pinned low corner) -> low-corner refine (r4, 0.2..0.6)
            # -> 100-shot verify (r5, id 2768 _173159): Img1 0.3 / Img2 0.3 = survival 0.964, fidelity
            # 0.993 (pooled), d' 3.84, dist 6.4 -- clears the 99% fid / 95% surv gate. Servo regulates
            # at >=0.2 so 0.3 is a real setpoint (100 ms kept). Beam-2 carries d' (up to 5.06@2.4 solo);
            # more of either beam only heats (survival cliff above ~0.6/0.6). (Was 0.40/0.10 @ 50 ms.)
            # 2026-07-19 IN-SEQUENCE (2-round rearrange, defocus -4) 3013 img PID re-opt: 0.3 was badly
            # underpowered (d' 3.4 med, fid 0.980 -> corrupted loading counts). 0.5 = d' 4.67 med / 5.49
            # mean, fid 0.997 med (matches 33x33-level). 0.7/0.9 add brightness but median d'/fid plateau
            # + more img1 heating. Adopted 0.5/0.5. scans id 2804-2807 (_04xxxx).
            # 2026-07-27 SHARED-SETPOINT POLICY (2-round rearrange, RearrangeCommSeq2): the 399
            # imaging PID is engaged ONCE in the ROOT BlueMOTStep -- i.e. at THIS (the loading /
            # initial) pattern's setpoints -- and then HELD for the whole shot, so these two values
            # are the imaging power of ALL THREE images (3013 / 2198 / 2078). The mid/final
            # patterns' own Img1/Img2PIDSet are inert in that context and are kept EQUAL to these
            # on purpose (no misleading value, no mid-shot relock -- a relock drifts/rails, see
            # gotcha-imaging-pid-held-multiround-rearrange). Per-image brightness differences must
            # come from the DDS amps (Imag399.Amp1/Amp2) below, which do not drift within a shot.
            "BlueMOT": {"Img1PIDSet": 0.5, "Img2PIDSet": 0.5, "LoadingTime": 1.0},  # LoadingTime 1.0 s per user 2026-07-15
            "Imag399": {
                # 2026-07-18: ND filters ADDED to the imaging path -> PIDSet is the light knob again
                # at higher (regulating) setpoints, so amps reverted to 1 (PID-servo convention,
                # gotcha-stale-dds-amps-pid-imaging). Re-optimizing Img1/Img2PIDSet at 100 ms.
                # [prev 2026-07-16 workaround: amps 0.14/0.08 -- servo railed, DDS amps were the only
                #  light knob; superseded by the ND filters.]
                # 2026-07-27: in a MULTI-ROUND rearrange these amps are the ONLY per-image light knob
                # (the PID is held at the root/loading setpoint). AOM knee: 0.5-1.0 is optically FLAT,
                # only <= 0.5 actually attenuates. 1 = full held power for the img1 (loading) frame.
                "Amp1": 1, "Amp2": 1,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    # 2026-07-16 evening @ 100 ms/low dose: committed X/h RE-CONFIRMED drift-free
                    # (R220 _225554: X(0.16,0.20) 0.9619 beats (0.175,0.24); R221 _230133:
                    # h(0.16,0.18) 0.9561, more h amp HURTS -- unlike kagome). Unchanged below.
                    # 2026-07-15 imaging-cooling opt (Imag399.Cool556 X/h) interleaved X<->h
                    # coordinate ascent on ImagingPushoutSurvivalSeq, 0.1 s hold, z4=-2, at the
                    # committed PID imaging power (Img1 0.40 / Img2 0.10). det robustly ~0.16 MHz
                    # (0.22 always died); h amp 0.24->0.18 was the big win (survival ~0.16->0.30).
                    # Converged within cross-round shot-noise (peak bounced 0.16-0.30 at ~13% loading,
                    # optimum LOCATION stable). scans _114434 (X r2) / _114948 (h r3) / _115455 (X r4).
                    # 2026-07-15 RE-OPTIMIZED on tri_3013_camfb @ 0.60 loading (clean stats vs v2's
                    # ~13% noisy): interleaved X<->h, 0.1s hold, Img 0.40/0.10. det 0.16 robust both
                    # (0.10/0.22 die); X amp 0.23->0.20, h unchanged 0.18 (confirmed). survival
                    # 0.166(X r6) -> 0.239(h r7). scans _143721 (X) / _144235 (h).
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},  # camfb r6 (was 0.23 on v2)
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.18},  # camfb r7 confirmed (was 0.24 feedback11 copy)
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        # 2026-07-16 NEW kagome arrays (names = trap count). SEEDED from tri_3013_camfb (the PID-servo
        # imaging scheme: DDS Imag399.Amp1/Amp2 stay at base=1, imaging power via BlueMOT.Img1/Img2PIDSet;
        # see gotcha-stale-dds-amps-pid-imaging). Same VSLMServo 3.5 as 3013 but HIGHER trap depth (per
        # user -> better imaging headroom). Loading phase phase/<name>.pt, z4=-2 baked. Imaging PID +
        # Cool556 X/h below are 3013's optima as a STARTING POINT -- being re-optimized 2026-07-16 (the
        # PID 2-D + Cool556 X<->h campaign). GreenMOT/LAC copied from camfb as loading seeds.
        "kagome_2078_camfb": {
            "Orca": {"ExposureTime": 0.1},  # 100 ms 2026-07-18 (match 3013/2198 for the ND-filter PIDSet re-opt; was 0.05)
            # 2026-07-16 per-pattern detection mask (see kagome_res_2198 / open-imaging-psf-double-lobe)
            "boxSize": 13, "maskSigma": 3.5,
            # 2026-07-16 retro z4 sweep (-4..-1, scans r117-120): z4 = -2 CONFIRMED optimal
            # (d' 4.84 fid 0.9969 surv 0.9575; -3: 4.53, -1: 3.88) -- keep passing loading_defocus -2.
            # 2026-07-16 trap-depth feedback (amp scaling, campaigns/feedback/kagome/): CV 8.84 -> 5.50 -> 3.40
            # -> 2.98% in 3 rounds (f0 107.9428 MHz, ~314 uK, spread 1.79x -> 1.35x). KEEPER = fb3 =
            # live phase/kagome_2078_camfb.pt (+_r3.pt record; prev in phase_history/). 6 chronic
            # bad-survival sites (521,855,1083,1111,1192,1913; surv 0.56-0.86) x1.5-boosted -> 5/6
            # revived + fit ~280 uK; 521 still unfit (accepted). scans _041344/_042150/_0429xx/_044x.
            "Init": {"VSLMServo": 3.5},
            "GreenMOT": {"BiasCoilCurrent": {"X": 0.039, "Y": 0.255}},   # seed from tri_3013_camfb
            "LAC": {"FreqDetuning": 0.22e6, "Amp": 0.10, "Time": 30e-3},  # seed from tri_3013_camfb
            # 2026-07-18 POST-ND imaging-power re-opt (amps=1, 100 ms, 0-pushout, z4=-2): low-corner
            # 2-D (r10) -> 100-shot verify (r11, id 2774 _181631) at Img1 0.4 / Img2 0.4. Common-mode-
            # normalized truth: survival 0.977, fidelity mean 0.998 / median 0.999 / worst-5% 0.990.
            # Fidelity clears 99.5%; survival ~1.3% under the 99% gate = trap-depth/loading-limited (NOT
            # imaging power -- power plateaus). Servo regulates at >=0.2 so 0.4 is a real setpoint.
            # [prev 0.40/0.10 @ 50 ms (workaround era).]
            # 2026-07-16 PID 2-D (Img1 0.40-1.00 x Img2 0.10-0.40, 0-pushout 50 ms): survival falls
            # with power on BOTH axes (heating cliff), pooled fidelity flat -> low corner optimal;
            # kept 3013's 0.40/0.10. Loading 0.68 (CV 0.05) after ~5 min thermalization. scan _011122.
            # 2026-07-27 UNIFIED to the SHARED 2-round setpoint 0.5/0.5 (was 0.4/0.4): as the FINAL
            # pattern of RearrangeCommSeq2 this entry's PIDSet is INERT -- the 399 PID locks once in
            # the root BlueMOTStep at the LOADING pattern (tri_3013_camfb, 0.5/0.5) and HOLDS, so
            # img3 is taken at 0.5/0.5 no matter what is written here
            # (gotcha-imaging-pid-held-multiround-rearrange). Keeping 0.4/0.4 only pretended img3 ran
            # dimmer. It DOES still apply when 2078 is the ROOT/loading pattern of a standalone scan;
            # 0.4/0.4 was that scan's 07-18 optimum and sits on the same power plateau (see above), so
            # 0.5 is within the flat region. Per-image brightness now via Imag399.Amp1/Amp2 below.
            "BlueMOT": {"Img1PIDSet": 0.5, "Img2PIDSet": 0.5, "LoadingTime": 1.0},
            "Imag399": {
                # 2026-07-27 EXPLICIT (was inherited base 1/1): with the PID held at the loading
                # pattern's setpoint these DDS amps are the ONLY per-image light knob for the img3
                # (final) frame -- dial them, never the PIDSet above. AOM knee: 0.5-1.0 is optically
                # FLAT, only <= 0.5 actually attenuates (07-16 R212).
                "Amp1": 1, "Amp2": 1,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    # 2026-07-16 imaging-cooling re-opt on kagome_2078_camfb (deeper traps than 3013
                    # at same VSLMServo 3.5): interleaved X<->h @ 0-pushout 50 ms, PID 0.40/0.10.
                    # Cooling was THE lever (PID power was not): X amp 0.20->0.30 the big win, det
                    # 0.16->0.14; det >=0.22 collapses survival (0.4-0.6). X re-check at new h stable
                    # (fixed point, 2 rounds). h broad plateau, balanced (0.14, 0.22). HEAD-TO-HEAD
                    # 100-shot confirm vs 3013-seed: survival 0.869->0.955 (+8.6%, ~80 SEM), per-site
                    # fid med 0.981->0.990, d' med 3.31->3.84. scans _011814(X) _012234(Xexp)
                    # _012505(h) _012913(Xrecheck) _013119(W_new) _013425(seed).
                    "X": {"FreqDetuning": 0.14e6, "Amp": 0.30},
                    "h": {"FreqDetuning": 0.14e6, "Amp": 0.22},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        "kagome_res_2198": {
            "Orca": {"ExposureTime": 0.1},  # 100 ms COMMITTED 2026-07-16 evening (user): floor light +
            # double-lobe PSF leave d' ~4 at 50 ms; power lever dead (servo floor, R201 _102820) so
            # buy separation with TIME. Camera follows via runner sync_camera_exposure. 150-shot
            # confirm _220111 (with the light-down + cooling below): per-site surv 0.9815(3),
            # fid_med 0.9966, d' 4.51 -- vs 50 ms baseline _095945 0.9555/0.9877/3.94.
            # (Prev: 0.05; 75 ms test 07-16 REVERTED --
            # Img1PIDSet servo is FLOORED below ~0.6 V -- 0.25->0.60 setpoint gave only 1.23x light,
            # scan _034653 -- so the power-down chain couldn't be exercised; camera reverted too)
            # 2026-07-16 per-pattern detection mask (scan_prep hook): the imaging PSF is DOUBLE-LOBED
            # (2.2 px sep, see open-imaging-psf-double-lobe); (13, 3.5) captures both lobes, +7% d'
            # measured offline vs the default (9, 2.0). KEPT.
            "boxSize": 13, "maskSigma": 3.5,
            "Init": {"VSLMServo": 3.5},
            "GreenMOT": {"BiasCoilCurrent": {"X": 0.039, "Y": 0.255}},   # seed from tri_3013_camfb
            "LAC": {"FreqDetuning": 0.22e6, "Amp": 0.10, "Time": 30e-3},  # seed from tri_3013_camfb
            # 2026-07-16 SURVIVAL-BIASED opt (goal: >=0.99 surv, fidelity tradeable -- rearrangement
            # mid-array). LOADING DEFOCUS z4 = -3 (NOT the inherited -2): z4 sweep -4..0 + refine
            # +-0.5 -> d' 3.7 -> 5.18, fid 0.988 -> 0.9984, surv 0.9365 -> 0.9463 (scans _014826..
            # _015616). Pass loading_defocus -3 on EVERY 2198 scan. PID-down 2-D at z4=-3 (_015808):
            # survival FLAT 0.93-0.97 over Img1 0.15-0.45 x Img2 0.04-0.12 (power not the lever at
            # this focus; ceiling depth/cooling-limited) -> picked low-dose best cell 0.25/0.10
            # (fid 0.9962, surv 0.9675). Trap-depth feedback + cooling re-opt = the path to 0.99.
            # 2026-07-18 POST-ND re-opt (amps=1, 100 ms, z4=-3): low-corner 2-D (r6) -> 100-shot verify
            # (r7, id 2770 _174249) at Img1 0.5 / Img2 0.5. Survival + fidelity RISE with Img2 (opposite
            # of 3013) and plateau at Img2>=0.5. Common-mode-normalized truth: survival 0.987, fidelity
            # mean 0.9996 / median 0.99995 / worst-5% 0.998. Cooling X<->h re-confirmed converged
            # (X 0.15/0.26~prior 0.15/0.27, h 0.15/0.22 unchanged; r8/r9 id 2771/2772) -> cooling NOT
            # the lever. Fidelity clears 99.5%; survival ~0.987 vs 0.99 = trap-depth-limited, not imaging.
            # [prev 0.25/0.10 (servo-floor workaround era).]
            # 2026-07-27: as the MIDDLE pattern of RearrangeCommSeq2 this PIDSet is INERT (the PID
            # locks once in the root BlueMOTStep at the LOADING pattern and HOLDS -- img2 runs at
            # tri_3013_camfb's 0.5/0.5). It is DELIBERATELY EQUAL to that shared setpoint so nothing
            # here is misleading; it still applies when 2198 is the ROOT pattern of a standalone
            # scan. Keep it in sync with tri_3013_camfb.BlueMOT.Img1/Img2PIDSet; change per-image
            # brightness with Imag399.Amp1/Amp2 below, never with this
            # (gotcha-imaging-pid-held-multiround-rearrange).
            "BlueMOT": {"Img1PIDSet": 0.5, "Img2PIDSet": 0.5, "LoadingTime": 1.0},
            "Imag399": {
                # 2026-07-16 evening DELIBERATE post-servo light-down (WITH the 100 ms exposure
                # above): the Img1/Img2 PID servo is RAILED (PD gain too low, user-diagnosed --
                # open-img1pid-servo-floor) so setpoints cannot regulate down (R211 _213205: dist
                # flat over Img1 0.10-0.25 x Img2 0.04-0.10) and DDS amps only attenuate below the
                # AOM saturation knee ~0.5 (R212 _214049 flat 0.5-1.0). Amps 0.175/0.14 cut the
                # light ~2x (dist 10.4 -> 5.1) -> survival 0.972 -> 0.9815 at fid_med 0.9966
                # (R213/R214 maps _214853/_215457; 150-shot confirm _220111). This intentionally
                # violated the amps=1 PID convention (gotcha-stale-dds-amps-pid-imaging).
                # 2026-07-18: ND filters ADDED to the imaging path -> PIDSet regulates again, amps
                # reverted to 1; re-optimizing Img1/Img2PIDSet at 100 ms. [prev workaround 0.175/0.14.]
                # 2026-07-27: in the 2-round rearrange (held PID) these are the ONLY per-image light
                # knob for the img2 (middle) frame. AOM knee: 0.5-1.0 optically FLAT, only <= 0.5
                # attenuates (07-16 R212).
                "Amp1": 1, "Amp2": 1,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    # 2026-07-16 evening re-opt AT 100 ms, defocus -3 (X 2-D _210347, h 2-D _211211,
                    # drift-free h A/B _212630: amp 0.22 beats 0.18 by ~10 SEM; det 0.15~0.175).
                    # X optimum unchanged from the 50 ms maps (_105535); h amp moved up at the
                    # doubled dose. Knife-edge det ridge (~2-3%/25 kHz) -- cross-scan drift real,
                    # only in-scan comparisons decide.
                    # [50 ms history: overnight fb5 X/h (0.16, 0.30/0.22) confirm _033639
                    # 0.9483(6); daytime 50 ms maps _105535/_204925.]
                    "X": {"FreqDetuning": 0.15e6, "Amp": 0.27},
                    "h": {"FreqDetuning": 0.15e6, "Amp": 0.22},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        # 2026-07-19 kagome_res_2198 MOVED +50 knm-px (row) toward the zeroth order to deepen the traps:
        # mean depth 292 -> 322 uK (+10%, mj1 dip shift 1.895 -> 2.087 MHz, id 2780), all 2198 spots still
        # load (a +75 variant dropped ~8 near-DC spots to the physical DC hole -> rejected). CV rose
        # 4.4 -> 6.9% (mostly speckle from the grating re-projection; ~12% radial) -- trap-depth feedback
        # to re-flatten is deferred. Loading hologram phase/kagome_res_2198_closer.pt, z4=-3 (pass
        # loading_defocus -3). Imaging/cooling/PID/mask SEEDED from kagome_res_2198 (being re-optimized
        # 2026-07-19 now that the array is deeper). Registry+detection key = this basename.
        "kagome_res_2198_closer": {
            "Orca": {"ExposureTime": 0.1},          # 100 ms. (2026-07-19 tested 50 ms r19: fidelity dropped 0.999->0.989, d' 4.2->4.1, no survival win -- fewer photons cost separation faster than shorter dose helped. Reverted.)
            "boxSize": 13, "maskSigma": 3.5,        # double-lobe PSF mask (see open-imaging-psf-double-lobe)
            "Init": {"VSLMServo": 3.5},
            "GreenMOT": {"BiasCoilCurrent": {"X": 0.039, "Y": 0.255}},
            "LAC": {"FreqDetuning": 0.22e6, "Amp": 0.10, "Time": 30e-3},
            "BlueMOT": {"Img1PIDSet": 0.5, "Img2PIDSet": 0.5, "LoadingTime": 1.0},  # seed from 2198; re-opt 07-19
            "Imag399": {
                "Amp1": 1, "Amp2": 1,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.15e6, "Amp": 0.27},   # seed from 2198
                    "h": {"FreqDetuning": 0.15e6, "Amp": 0.22},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        "33x33_uniform": {
            # Initialize the SLM servo to 3.7
            "Init": {"VSLMServo": 1.9},
            "LAC": {"Time": 20e-3},   # per-user override 2026-06-21 (base 10 ms); deep-merges, rest of LAC inherits base
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.10,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.26},
                    "h": {"FreqDetuning": 0.20e6, "Amp": 0.22},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },

        # 2026-07-10: 33x33_feedback11 = PRODUCTION successor to feedback10 (one more amp-scale round r4,
        # measured with a DENSE mj=1 sweep 104.8-105.7/0.05). Measured CV 1.54% (1065/1068 sites -- dense
        # sweep recovered the marginal edge sites feedback10's coarse sweep dropped); split-half TRUE CV
        # ~2.2% (residual is REAL corner under-illumination [optical, needs a beam realign] + speckle floor
        # -- amp-scale actuator exhausted, Dphi flat 0.007 rad, transfer +0.27). IS the r4 keeper phase
        # (identical .pt, sha df7dde588a54). Overlay = EXACT COPY of 33x33_feedback9. See CAMPAIGN_STATE_fb9b.md.
        "33x33_feedback11": {
            "Orca": {"ExposureTime": 0.050},  # 2026-07-29 30 ms TRIAL RUN AND REVERTED -- 50 ms KEPT. Tried 50->30 ms to cut the per-shot 399 heating budget (at 50 ms both power maps were FLAT: PIDSet r60 and DDS-amp r64 = power-saturated). 30 ms LOST decisively: the amp map stopped being flat and rose monotonically toward full power (= photon-STARVED, not saturated), and at the matched 1.0/1.0 cell / same cooling / same det -5: fid 0.9949 vs 0.9992, surv 0.9899 vs 0.9937, d' 4.74 vs ~6.4 (30 ms r66 data_20260729_175910 vs 50 ms r64 data_20260729_173122). The BEST 30 ms cell (fid 0.9954, d' 4.31) still lost to a typical 50 ms cell. Lesson: the 50 ms flatness was the GOOD regime -- saturated = all the photons we need; shortening left saturation and cost Gaussian separation without buying survival. (was 0.025 @ 07-14; 35 ms opt W 1.0/0.35, 25 ms opt W 1.3/0.25.)
            # 2026-08-11 3.3 -> 1.9 (user directive). 1.9 was the measured operating point all
            # of 08-11: the loading plane z4 -4.0 below was measured AT 1.9 (17-plane sweep
            # 20260811_135519 + head-to-head 20260811_140133), and the day's 556 mj=0/mj=1
            # scans ran there. Also note VSLMServo appears NOT to move trap depth over
            # ~1.9-3.9 on this array (mj=1 depth-differential line moved only ~15 kHz on a
            # 2.5 MHz splitting between 1.9 and 3.3; loading flat 0.54-0.61 across a 2.7-3.9
            # sweep, job 773) -- the 532 power servo looks saturated or its setpoint is not
            # reaching the power loop, so treat 1.9 as "the point we characterize at", not as
            # a known-lower depth. (was 3.3 @ 08-10, 3.5 before that.)
            "Init": {"VSLMServo": 1.9},
            # 2026-08-10: this array's LOADING PLANE, measured not assumed. A 21-plane z4 sweep
            # (-10..0 step 0.5, scan 20260810_134709) peaks at -2.5 (dist 6.27 ADU, d' 4.80) vs the
            # global -5 (5.50, 4.50), plateau -4.0..-1.5; the 100-shot head-to-head confirms it
            # (survival 0.9911 +- 0.0004 vs 0.9841 +- 0.0006 = ~10 SEM, per-site fidelity median
            # 0.9981 vs 0.9938). Read by slm_runtime._pattern_defocus as the DEFAULT plane for any
            # scan that does not set runp().loading_defocus itself.
            # ⚠ REARRANGEMENT IS NOT COVERED BY THIS. The global SLM->camera affine is calibrated at
            # one plane (-5), so a rearranged run at -2.5 would map coordinates against a stale
            # affine. The rearrangement scans all set rp.loading_defocus explicitly (matched to
            # rearrange_kwargs.extras.z4), which OVERRIDES this key -- so they are unaffected and
            # stay at their own plane until someone re-bootstraps the affine at -2.5.
            # 2026-08-12: -2.5 -> -4.0. The -2.5 above was chosen against the GLOBAL -5; it was never
            # compared to -4, and -4 is the plane the 08-11 and 08-12 imaging campaigns actually operate
            # at (they pass rp.loading_defocus=-4 explicitly), so the default disagreed with practice and
            # every scan that did NOT override -- including today's whole daily calibration -- silently
            # imaged at the worse plane. THREE interleaved 120-shot single-point runs at the identical
            # config (X 0.18/0.20, h 0.20/0.12, det +5, PIDset 0.80/1.00, DDS 1/1), ordered -2.5 / -4 /
            # -2.5 so drift cannot fake it: dist 5.1 / 5.7 / 4.8 ADU, per-site d' 4.245 / 4.619 / 4.072,
            # pooled fidelity 0.9944 / 0.9960 / 0.9928 (scans 20260812_115809, _120509, _120845). -4 beats
            # BOTH -2.5 runs on every brightness metric; survival is tied (0.9865 / 0.9859 / 0.9827).
            # The SPATIAL gradient is the clincher: at -2.5 the array carries a real left-right tilt
            # (per-site fidelity -0.0073, survival -0.0120, d' -0.74 across x) that VANISHES at -4
            # (+0.0013 / -0.0053 / +0.12). Consistent with 08-11's independent 40-shot/plane head-to-head
            # (dist/d' 4.37/3.07 at -4.0 vs 3.26/2.45 at -2.5) and with 08-12 r904/r905 (-4 tied or better
            # vs -5). Per-site figures at -4: fidelity median 0.9974 / mean 0.9964 / 78% >= 0.995,
            # d' median 5.06 (99.8% > 3), survival mean 0.9859. The REARRANGEMENT caveat below still
            # applies unchanged -- those scans set rp.loading_defocus explicitly and are unaffected.
            "SLM": {"Loading": {"Defocus": -4.0}},  # was -2.5 (see above); global base -5
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            # 2026-07-14 imaging optimization (399 now PID-servoed via Img1/Img2PIDSet;
            # DDS Imag399.Amp1/Amp2 held at 1). Beam-isolation tests showed both 399
            # beams image (beam 2 cleaner, peak d' 4.24; beam 1 saturates ~3.5-4); the
            # d' cap was COOLING-limited, not 399-power. Re-optimized at 25 ms exposure
            # (was 35 ms): 2-D setpoint map -> W Img1 1.3 / Img2 0.25 (higher Img1 recovers
            # the separation lost to the shorter dose; Img2 heating cliff sits higher at
            # 25 ms). Retuned Imag399.Cool556 (0-pushout X-then-h) -> per-site fidelity
            # 0.9995, d' 5.56, survival 0.992, spatially flat (confirm data_20260714_194013,
            # 100 shots). 25 ms matches/beats 35 ms at a gentler dose. Notion 07/14.
            # (35 ms optimum was W 1.0/0.35, X 0.14/0.28, h 0.14/0.20.)
            # 2026-07-16 25->50 ms re-optimization (imaging fidelity was LOW at 25 ms).
            # Doubling the collect window fixed fidelity to ~0.998 and dropped the imaging
            # power needed (25 ms 1.3/0.25 -> 50 ms 0.8/0.15; fidelity now FLAT vs setpoint,
            # so d' is cooling/depth-limited, not 399-power). R1 pidset 2-D (data_20260716
            # _130147) -> 0.8/0.15; R2 Cool556.X (data_..130525) d' 4.4->4.8; R3 Cool556.h
            # (data_..130933) -> h det 0.16->0.14, amp 0.24->0.20. Single-point 100-shot verify
            # data_20260716_131428: fidelity 0.9981, d' 4.14, survival 0.9922, load 0.58.
            # 2026-07-18 ND-FILTER RECAL. Added ND at the END of each 399 imaging path,
            # DOWNSTREAM of the imaging-PID photodiode: Img1 ND=0.3 (~50.1% T), Img2 ND=0.7
            # (~20% T). The PD reads pre-ND power, so the setpoints must rise ~1/T to restore the
            # atom-plane intensity. Naive 1/T predicted 0.8/0.15 -> ~1.6/0.75; the measured optimum
            # was HIGHER (servo gain-compressed): R1 PID map (Img1 1.0-2.2 x Img2 0.45-1.05,
            # data_20260718_112826) rose monotonically to the corner; R2 (2.2-3.4 x 1.05-1.65,
            # _113416) went DEAD FLAT -> the servo SATURATES above ~2.2/1.05 (mirror of the <0.6 V
            # low floor, open-img1pid-servo-floor). Rise was done by Img1 ~1.3. Picked COMFORTABLE
            # mid-range 2.0/1.0 (well above the 0.6 floor, below the 2.2 rail, servo actively
            # regulating; fid ~0.997/surv ~0.99, tied with the corner). d' flat vs power = cooling-
            # limited, so re-tuned Cool556 at 2.0/1.0: R3 X (_113829) det 0.14->0.16 amp 0.25
            # (plateau det 0.16-0.20 x amp 0.22-0.28); R4 h (_114920) det 0.14->0.18 amp 0.20->0.18
            # (det-0.18 column best, low-amp safe). 100-shot VERIFY (data_20260718_120029): fidelity
            # 0.9980, d' 4.32, survival 0.9909, load 0.59 -- at parity with the pre-ND 50 ms baseline
            # (fid 0.9981/d' 4.14/surv 0.9922). DDS Imag399.Amp1/Amp2 stay at base=1 (PID scheme).
            # 2026-08-27: added LoadingTime 0.28 s as a per-array overlay (base stays 0.5 s).
            # The 0a curve (20260827151119) shows the cliff dead <0.09 s, a steep rise 0.13-0.26 s
            # and a plateau from ~0.295 s, so 0.5 s was deep in saturation; 0.28 s gives BETTER
            # loading (verify 0.480+-0.014) at ~half the blue-load cycle time once the bias is
            # centred. Kept as ByPattern, not base, since it is an array/depth-dependent work-point.
            # GreenMOT.PowerBroaden.HandoverTime (the blue->green overlap) was swept 0.005-0.050 s
            # (20260827152511) and is a BROAD PLATEAU 0.010-0.050 (0.502-0.550, all within ~1-2 SEM;
            # only 0.005 is worse at 0.451), so the base 0.015 s already sits on it -> left alone.
            "BlueMOT": {"Img1PIDSet": 0.8, "Img2PIDSet": 1.0, "LoadingTime": 0.6},  # PIDSet: ND recal 2026-07-18 (was 0.8/0.15 pre-ND @50ms)
            "Imag399": {
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    # 2026-07-29 imaging re-optimization at 50 ms / det -5 (PIDSet 2.0/1.0, DDS amps 1/1).
                    # BOTH power axes came back FLAT above their low-edge cliffs -- PIDSet 5x5 (r60,
                    # data_20260729_170938: fid .9991-.9996, surv .992-.996, d' 6.1-6.4) and DDS amp 6x6
                    # (r64, data_20260729_173122: flat above 0.32, only amp1=0.15 collapses) -- i.e. imaging
                    # is POWER-SATURATED and cooling/depth-limited, so only the cooling moved. X from r61
                    # (data_20260729_171415, fit 0.178/0.260 vs argmax 0.19/0.23); h from r62
                    # (data_20260729_172113, fit 0.192/0.201 vs argmax 0.16/0.19; amp raised since the whole
                    # amp<=0.13 region is clearly worse). 100-shot verify data_20260729_172735: per-site
                    # fidelity median 0.99997, d' median 6.82, survival 0.9941+-0.0003, load 0.588, 99.8% of
                    # sites at fidelity >=0.995. Detuning kept at -5 MHz: a matched det -3 pass (r73,
                    # data_20260729_182850) was ALSO power-saturated and tied within noise (fid 0.9993,
                    # d' 6.80, surv 0.9962) -- detuning is not the limiter. Residual low sites 624/625
                    # (surv 0.52/0.53 at d' 2.6) are SHALLOW TRAPS (SLM/depth), not a cooling problem.
                    # 2026-08-03 imaging re-optimization after the 399 DAC outage moved the laser
                    # frequency (see Imag399.FreqDetuning below). Rounds r543/r544 detuning ->
                    # r545 power -> r547 cool X -> r548/r549 cool h, all at 0 pushout.
                    # X: survival collapses toward HIGH X detuning at the new imaging frequency
                    # (r547 data_20260803_155659: 0.9947 at 0.16/0.23 vs 0.9882 at the 07-29 0.19/0.23,
                    # and only 0.8915 at 0.25/0.17) while fidelity stays flat -> the move is free.
                    # h: r548 (data_20260803_160055) put both best cells at det 0.22; the 100-shot
                    # head-to-head r549 (data_20260803_160508) had amp 0.20 vs 0.23 statistically tied
                    # on survival (0.9941 +-0.0004 vs 0.9940 +-0.0003), so amp 0.20 was taken on its
                    # slightly better per-site d' (5.69 vs 5.48) and fidelity (0.99969 vs 0.99954).
                    # 2026-08-11 imaging re-opt at the day's operating point (VServo 1.9 / loading
                    # plane z4 -4.0, both per-scan overrides). Coordinate ascent with the 399 hold
                    # amps pinned at 0.3/0.3: X round 1 (20260811_150710, h at BASE 0.16/0.13 --
                    # build_2d pins the non-swept beam from BASE consts, NOT the ByPattern overlay)
                    # -> det 0.18 amp 0.24; h (20260811_151132, X pinned 0.18/0.24) -> det 0.14 amp
                    # 0.20, best on BOTH survival and d'; X round 2 (20260811_151619, h pinned at
                    # 0.14/0.20, amp grid shifted to 0.16-0.32) -> det 0.14 amp 0.24, i.e. the amp
                    # reproduced and the detuning bounced one 0.04 step with no survival gain =
                    # converged on a flat det 0.14-0.18 plateau, so det stays at 0.16 (plateau
                    # centre = the incumbent value). 50-SHOT HEAD-TO-HEAD decided the h move rather
                    # than the 6-shot grid cells: incumbent X 0.16/0.23 + h 0.22/0.20
                    # (20260811_152029) gave survival 0.9500 +- 0.0012, d' 2.64, fidelity 0.9336;
                    # this pair (20260811_152136) gave 0.9662 +- 0.0010, d' 3.25, fidelity 0.9625 --
                    # +0.0162 survival (~10 SEM) at matched loading (0.653 vs 0.665).
                    # 2026-08-12 imaging re-optimization for the rearrangement-methods campaign,
                    # at the campaign's own operating point (z4 -4, VSLMServo 1.9, 50 ms, 0 pushout).
                    # Order per the runbook: warm-up verify -> detuning -> loading plane -> power ->
                    # cool X -> cool h -> 100-shot head-to-heads. Rounds r901-r910.
                    #  * THE DETUNING WAS THE WIN, not the cooling: the in-use -4 MHz read
                    #    fid 0.9927 / d' 3.99 / surv 0.9816 (r901, 60 shots), and a -10..+6 sweep
                    #    (r902) plus a +1..+12 sweep at 8 reps (r903) BOTH peak at +5 MHz on a
                    #    +3..+7 plateau, falling off hard above +8 (d' 3.2-3.4 by +11).
                    #  * POWER IS SATURATED again: the 5x5 PIDSet map at det +5 (r906, 150 shots)
                    #    is flat -- fidelity 0.9918-0.9967, survival 0.978-0.990 across the WHOLE
                    #    grid -- so PIDSet stays 0.80/1.00 (mid-grid) and the "best" cell is
                    #    max-of-25 noise at 6 shots/cell. DDS amps stay 1/1.
                    #  * COOL X: the 5x5 map (r907) has real structure -- survival collapses toward
                    #    high det / low amp (0.938 at 0.26/0.16) and det 0.18 is the best column at
                    #    every amp >= 0.20. Moved 0.16/0.24 -> 0.18/0.20.
                    #  * COOL h: the 4x5 map (r908) collapses at low det + high amp (0.27 survival
                    #    at 0.08/0.28); good region det 0.16-0.20, amp 0.12-0.20.
                    #  * The h choice was decided by 100-shot head-to-heads, NOT the 6-shot argmax:
                    #    h(0.20,0.12) gave fid 0.9950 / d' 4.604 / surv 0.9824 (r909) vs
                    #    h(0.16,0.20) fid 0.9948 / d' 4.436 / surv 0.9846 (r910) vs the old
                    #    h(0.14,0.20) fid 0.9941 / d' 4.311 / surv 0.9839 (r904). Survival is TIED
                    #    across all three (~1 sigma at 100 shots); h(0.20,0.12) wins on d' (+0.29)
                    #    and fidelity, so it is taken on separation, not on survival.
                    #  * LOADING PLANE settled empirically and kept at -4: the depth spread is
                    #    ~1.7x tighter at -5 (mj=1 FWHM 331 vs 649 kHz, CV 5.68 vs 9.51 %, scans
                    #    20260812_043657 vs _042758) but imaging is TIED there -- 100 shots each at
                    #    det +5 gave -4: fid 0.9941 / d' 4.311 / surv 0.9839 (r904) vs -5: 0.9934 /
                    #    4.303 / 0.9818 (r905). So the uniformity gain does not reach the image, and
                    #    -4 keeps continuity with all the Fig-4-era rearrangement data.
                    #  * STILL OPEN: d' is 4.3-4.6 against 6.2-6.8 on 08-05 with BOTH power axes
                    #    flat and depth normal (429 uK, within 3 % of the paper's 418 and 0.3 % of
                    #    08-11), so the residual is NOT imaging power, cooling, plane or trap depth.
                    #    Next suspects are the collection path / beam pointing (see the
                    #    open-imaging-psf-double-lobe memory), which is hardware, not a scan knob.
                    # 2026-08-19 imaging re-optimization (rounds r120-r129, all 0 pushout, z4 -4,
                    # VSLMServo 1.9, 50 ms). THE DETUNING MOVED AGAIN (see FreqDetuning below): +5 -> -1.
                    # At det -1 the X map (r124 data_20260819_075810 + edge-extension r125 _080200) has a
                    # clear interior optimum at MORE 556 power -- (0.18, 0.28-0.30), survival 0.992-0.993
                    # vs 0.9858 at the incumbent (0.18, 0.20); consistent with more resonant-photon
                    # heating at the new detuning. The h map (r126 _080433, X pinned 0.18/0.28) was FLAT
                    # (surv 0.987-0.995), so h stays. PIDSet map at -1 (r123 _075408) flat = power still
                    # saturated -> 0.80/1.00 kept, DDS 1/1. 100-shot head-to-head: new W 0.9932/d' 5.41
                    # vs old W 0.9870/d' 5.05 (~10 SEM). Combined 200-shot verify (r127 _080750 +
                    # r129 _081307): per-site fidelity median 0.99975 (99.8% >= 0.995), d' median 6.12,
                    # survival 0.9929, spatially FLAT. The 08-12 "STILL OPEN d' 4.3-4.6" residual is
                    # RESOLVED: it was the drifted 399 detuning, not the collection path. Low tail =
                    # chronic shallow traps 624/625 (d' 2.5/2.8, SLM depth).
                    # 2026-08-27 imaging re-optimization (Cool556 X+h, 1 s-hold locate -> 0-pushout
                    # confirm). CONTEXT: the PIDSet 2-D grid came back a flat plateau (pooled d'
                    # 5.69-6.40 unstructured over Img1 0.6-1.0 x Img2 0.8-1.2), and the correct
                    # per-site metric showed WHY power was the wrong knob: per-site fidelity ALREADY
                    # passed (median 0.99999, d' 6.95, only 0.3% of sites d'<3) while array survival
                    # FAILED the >=99% gate at 0.9125. So the limiter was 556 cooling, not 399 power --
                    # the same read as the 07-16/07-18 "d' flat vs setpoint -> cooling/depth-limited"
                    # notes above. R1 X 1 s-hold 7x8 grid (data_20260827_131026, 168 shots): SEM-weighted
                    # rotated-2-D-Gaussian peak det 0.1558+-0.0031 amp 0.2965+-0.0066, interior, ridge
                    # tilted -15.6 deg; argmax cell (0.16,0.26) S=0.711 vs the then-current (0.18,0.28)
                    # S=0.437, i.e. a large structured gain, not plateau noise. R2 h grid 0.06-0.20
                    # (data_20260827_131721) was EDGE-PINNED at amp 0.20 (monotonic rise, slope +1.02/unit,
                    # r=0.959, p=0.0002) -> expanded per the edge rule. R3 h amp 0.18-0.42
                    # (data_20260827_132513, 135 shots) revealed a hard rolloff above 0.30 (S -> 0.004 at
                    # amp 0.42/det 0.12), so R2's "still climbing" was the LEFT FLANK of a peak sitting on
                    # its old boundary; the R2+R3 union amp-marginal peaks at 0.18 (0.644 -> 0.810 -> 0.085)
                    # and the rounds agree in their overlap (0.791 vs 0.810 at amp 0.18). h det is FLAT
                    # (r=-0.39, p=0.34), weakly favoring 0.15. DECIDED BY A 0-PUSHOUT HEAD-TO-HEAD, 25
                    # shots each, minutes apart: new W X(0.156,0.296)+h(0.15,0.18) data_20260827_133112
                    # per-shot survival 0.9902+-0.0008, d' 5.92, fid 0.9983 vs old W X(0.18,0.28)+h(0.20,0.12)
                    # data_20260827_133207 0.9652+-0.0042, d' 5.11, fid 0.9979 -> +2.50%, Welch t=5.90,
                    # p=3.3e-06 (5.9 sigma on the honest per-shot statistic; atom-level z=14.3 is optimistic).
                    # CROSSES the >=99% survival gate. Per-site verify at new W: fidelity median 0.99998
                    # (p5 0.99860, p1 0.99492), survival mean 0.9899, d' median 6.88, 0.2% of sites d'<3,
                    # NO significant survival/fidelity gradient (x p=0.57, y p=0.90); the low tail is the
                    # chronic dim-site set (s591 d'=1.4, s624/s625 d'=3.0-3.4) = SLM/trap-depth, not cooling.
                    # CAVEAT: trap depth NOT verified on scope this session (user-approved to proceed).
                    "X": {"FreqDetuning": 0.156e6, "Amp": 0.296},  # 2026-08-27 (was 0.18e6/0.28 @ 08-19)
                    "h": {"FreqDetuning": 0.15e6, "Amp": 0.18},  # 2026-08-27 (was 0.20e6/0.12 @ 08-12; R2 edge-pinned -> R3 expand, union peak amp 0.18)
                },
                # 2026-08-03: +2.0 MHz, NOT the base -5e6. The 399 wavemeter-PID DAC dropped overnight,
                # the laser parked +4.64 GHz off and was re-acquired mid-morning -- it came back on a
                # slightly different frequency, so the AOM offset that puts the imaging on resonance
                # moved by ~7 MHz. Symptom: per-site d' 4.47 all morning vs 6.82 on 07-29, which floored
                # every STIRAP number of the day. r543 (data_20260803_154335, -9..-1) was monotone toward
                # less-negative detuning and railed at the edge; r544 (data_20260803_154659, -2..+4)
                # found the interior optimum -- d' 6.18 at +2.0 vs 4.57 at the in-use -5, with fidelity
                # and survival both turning over by +2.5..+4. Re-verify this after any further 399
                # frequency work; it tracks the laser, not the atom.
                # 2026-08-05: -5e6 -> +10e6. The 08-03 note above was never committed (the value stayed
                # at -5), and the imaging was still running there: this morning's warm-up read pooled
                # d' 4.04. A -8..+5 MHz sweep (r601, data_20260805_182144) rose monotonically to the +5
                # edge (d' 3.96 at -5 -> 5.94 at +5, fidelity 0.9911 -> 0.9994); extending to +4..+14
                # (r602, data_20260805_182344) found the plateau: d' flattens ~6.3-6.5 above +6 and
                # dist saturates ~10-11 ADU. A 30-rep/pt dense confirm of +8/+10/+12 (r603,
                # data_20260805_182534) gave d' 5.95/6.23/6.25 with survival 0.972/0.975/0.976, so +10
                # was taken as mid-plateau (safer against drift than the +12 edge). 100-shot verify at
                # the adopted W (r615, data_20260805_204416): fidelity 0.9995, d' 6.22, survival 0.9946,
                # load 0.571; per-site over 200 combined shots (r613/r614/r615) fidelity median 0.99995,
                # d' median 6.82, survival mean 0.9946, 99.7% of sites >=0.995, spatially FLAT (survival
                # gradient 0.0000/-0.0004 across the array). Nothing else moved: the PIDSet map inside
                # the VALID Img1PIDSet range 0.2-0.8 (r610, data_20260805_202513) was a plateau above
                # Img1 ~0.7 with the in-use 0.8/1.0 already in it; the Cool556.X map (r611,
                # data_20260805_202926) put its argmax at (0.16,0.24) but a 50-shot head-to-head vs the
                # in-use (0.16,0.23) tied exactly (survival 0.9944 vs 0.9943, d' 6.13 vs 6.33 -- r614 vs
                # r613), so X was KEPT; the Cool556.h map (r612, data_20260805_203529) was flat across
                # the whole grid (survival 0.988-0.996), so h was KEPT. Residual low sites are the
                # chronic shallow traps 624/625/591 (survival 0.36/0.37/0.65 at d' 2.5/1.8/3.7 = SLM
                # depth, not cooling). Confirms the 08-03 lesson: this tracks the 399 laser -- re-scan
                # the detuning after ANY 399 frequency excursion, and don't trust a stale value.
                # 2026-08-07: +10e6 -> +2e6. Morning warm-up at the in-use +10 read pooled d' 3.5-4.1
                # (r120 amp map data_20260807_131659, r121 PIDSet map data_20260807_131901) vs 6.2 on
                # 08-05 -- and BOTH maps were flat, so power was not the limiter. A -8..+16 MHz sweep
                # at 0 pushout (r122, data_20260807_132454, 147 shots) found a clean single peak with
                # fidelity AND survival maximal at the SAME point, +2 MHz: fid 0.9982 / d' 5.36 /
                # surv 0.9931, vs fid 0.9908 / d' 4.19 / surv 0.9784 at the in-use +10, falling off
                # hard above (+16: fid 0.9603, d' 3.12, surv 0.9168). Plateau is -4..+4. NOTE the
                # 08-05 sweep that picked +10 ran through the OLD imaging_round detuning path, which
                # hardcoded Pushout.Time=0.2 s -- its survival column was a 200 ms-pushout amplified
                # proxy, which flattened the +5..+14 region into a fake plateau; that hold is now
                # honored from --hold (default 0 pushout), so this sweep is the honest one.
                # Power re-measured AT +2 (saturation does not transfer across detuning): the DDS-amp
                # map 0.4:0.15:1.0 squared (r123, data_20260807_133044, 104 shots) was flat above amp
                # ~0.55 with only the 0.4 row/column clearly worse, and its best cell was 0.7/0.85
                # (surv 0.9942 vs 0.9876 at the in-use 1.0/1.0). The drift-free 2x2 head-to-head
                # (r124, data_20260807_133331) was aborted early at 25 shots, but 0.7/0.85 again came
                # out top on survival (0.9905 vs 0.9890 at 1.0/1.0) -- best cell in BOTH scans, so
                # adopted. CAVEAT: r124 is 25 shots and the four cells span only 0.9849-0.9905, well
                # inside its own error bars -- the adoption rests on r123 + the repeat direction, NOT
                # on a statistically decisive head-to-head. Re-run r124 at the full 50 reps/cell to
                # confirm. PIDSet stays 0.80/1.00; cooling deliberately NOT re-optimized this round.
                # Not yet done: the full head-to-head above, and a high-rep per-site/spatial verify.
                # 2026-08-10 imaging re-optimization at the user-reported VSLMServo 3.3 (NOT the 3.5
                # this block's Init sets -- actual on-scope depth is lower than the set value).
                # Warm-up r700 (data_20260810_132250) read d' 3.5 / dist 3.9 ADU / survival 0.907 at
                # the in-use config -- roughly HALF the 08-05 separation -- and its whole detuning
                # sweep topped out at d' 3.67 at EVERY detuning. Everything from r701 on read
                # d' 4.4-4.7. CAUSE = THE TRAP DEPTH, not the imaging power: r700 is the only scan of
                # the day that ran at VSLMServo 3.5 (the user lowered this block to 3.3 between
                # 13:22 and 13:26, mid-campaign -- confirmed from each scan's config snapshot), so
                # the deeper array was decisively worse at every 399 detuning. The 08-07 DDS amps
                # (0.7/0.85) were first blamed and that was WRONG: the drift-free 2x2 head-to-head at
                # 3.3 (r708, data_20260810_140232, 50 reps/cell, 0.7/1.0 x 0.85/1.0) ties all four
                # cells (fid 0.9956-0.9961, d' 4.69-4.72, surv 0.9853-0.9883), i.e. the amp axis is
                # saturated here and 0.7/0.85 was never the regression. Amps set back to 1.0/1.0
                # anyway (best-fidelity cell of a flat map, and it matches base); PIDSet 0.80/1.00
                # kept -- its 5x5 map (r701, data_20260810_132551, Img1 0.4-0.8 x Img2 0.6-1.4) is
                # likewise FLAT at fidelity 0.991-0.9965 with the in-use cell already the best.
                # NOTE r700 is also the one scan measured at the OLD depth, so it is not comparable
                # to the rest of the day; treat the 3.5-vs-3.3 difference as observed, not isolated
                # (nothing else was held fixed across that edit).
                # Detuning re-measured at the restored power (r702, data_20260810_132859, -8..+6):
                # broad plateau -7..-1 (fid 0.995-0.996, d' 4.4, dist 5.2-5.4, surv 0.985-0.987),
                # falling hard above +1 (d' 3.58 at +6). Took -4 = mid-plateau. This is the third
                # large 399 detuning move in a week (+10 -> +2 -> -4); it tracks the LASER, not the
                # atom -- re-scan it after any 399 excursion, and today's daily scan flagged the 399
                # wavemeter PID as engaged=false with its lock voltage RAILED at 8.0 V.
                # Cooling re-checked at det -4 and NOT moved: the X map (r703, data_20260810_133209,
                # 7x7) and the h map (r704, data_20260810_133638, 7x7) are both FLAT -- the
                # SEM-weighted rotated-2D-Gaussian fit fails to converge on either (R^2 ~ 0, fitted
                # center runs off-grid), so neither has a resolvable peak. The h map's argmax
                # (0.25, 0.17) sat ~0.6% above the in-use (0.22, 0.20), which per the runbook earns a
                # drift-free head-to-head rather than a shrug: the 2x2 at 50 reps/cell (r705,
                # data_20260810_134126) tied all four cells (surv 0.9864-0.9878, fid 0.9954-0.9957),
                # so the argmax was noise and X/h both stay put.
                # LOADING DEFOCUS is the real remaining lever and is NOT set here (each YbScans file
                # hardcodes rp.loading_defocus = -5): a 21-plane z4 sweep at this W (-10..0 step 0.5,
                # 168 shots, scan 20260810_134709, per-shot Otsu self-thresholding since dim planes
                # cannot trust the registry) peaks at z4 ~ -2.5 (dist 6.27, d' 4.80) vs -5 (dist 5.50,
                # d' 4.50), plateau -4.0..-1.5. The 100-shot head-to-head confirms it decisively:
                # z4 -2.5 (r706, data_20260810_135225) fidelity median 0.9981 / d' 5.19 / survival
                # 0.9911 +- 0.0004 / 86% of sites >= 0.995, vs z4 -5.0 (r707, data_20260810_135452)
                # 0.9938 / 4.56 / 0.9841 +- 0.0006 / 40% -- a 0.0070 survival gain at ~10 SEM,
                # spatially flat (survival delta +0.005 across x, +0.000 across y). Residual low
                # sites are the chronic shallow traps 624/625 (d' 2.6) = SLM depth, not imaging.
                # 2026-08-12: -4e6 -> +5e6. Fourth large 399 imaging-detuning move in a week
                # (+10 -> +2 -> -4 -> +5): it tracks the LASER, not the atom, so re-scan it after any
                # 399 excursion and never trust a stale value. Two independent sweeps agree -- r902
                # (-10..+6, 5 reps) and r903 (+1..+12, 8 reps) both peak at +5 on a +3..+7 plateau --
                # and the in-use -4 was costing ~0.5 in d' and ~1-2 % survival. See the Cool556 block
                # above for the full round-by-round record.
                # 2026-08-19: +5e6 -> -1e6. FIFTH large 399 detuning move (+10 -> +2 -> -4 -> +5 -> -1);
                # it tracks the LASER -- re-scan after any 399 excursion, never trust a value >2 days old.
                # r121 (-2..+12, data_20260819_074745) railed at its low edge (d' 5.67 at -2 vs 4.84 at
                # the in-use +5); r122 (-10..0, _075055) resolved the interior peak: d' 5.84/5.80 at
                # -1/0 on a -2..+1 plateau, falling both ways. Survival at -1 recovered by the Cool556.X
                # re-opt (see above). Verify at the adopted W: d' median 6.12, fid 0.99975, surv 0.9929.
                "FreqDetuning": -1e6,  # imaging re-opt 2026-08-19 (was +5e6 @ 08-12; d' peak -1..0)
                "Amp1": 1.0,   # 2026-08-10 REVERTED from the 08-07 0.7 -- that value was the regression
                "Amp2": 1.0,   # 2026-08-10 REVERTED from the 08-07 0.85 (see r700/r701 above)
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                # 2026-07-20 RNR re-optimization (30 us release, interleaved X<->h coordinate ascent,
                # 4 rounds, data_20260720_174455/174908/175316/175704). The RNR Cool556 block predated
                # the 07-16/18 50 ms + ND re-optimizations (was X 0.16/0.14, h 0.16/0.12 -- detuning too
                # high). Converged to a symmetric fixed point det 0.12 MHz / amp 0.14 on BOTH beams
                # (X and h coupled; total 556 power is the lever). Peak recapture survival ~0.59 @ 30 us
                # (loading flat ~0.58 across the grid = real cooling signal). Plateau det 0.10-0.14 x
                # amp 0.12-0.18 flat within SEM. Notion 07/20.
                # 2026-08-19 RNR re-optimization (30 us release, X<->h coordinate ascent r10-r13,
                # 7 reps/cell, z4 -4). Both beams moved UP together -- same direction as the day's
                # Imag399.Cool556 re-opt (more 556 power wanted): X map (r10, scan 20260819081731)
                # peak (0.15, 0.17) 0.536+-0.006 vs 0.498 at the old (0.12, 0.14); h map (r11,
                # 20260819082129) plateau det 0.12-0.18 x amp 0.14-0.20, its amp-edge argmax killed
                # by the extension r12 (20260819082527: survival falls monotonically above amp 0.17)
                # -> h (0.15, 0.17); X re-check at the new h pin (r13, 20260819082821) reproduced
                # (0.15, 0.17) 0.544 = converged (pins match returns both ways). 100-shot drift-free
                # head-to-head old-vs-new full config (r14 20260819083234 / r15 20260819083438):
                # 0.4912 vs 0.5400 (+0.049, ~10 SEM) at matched loading 0.54.
                "X": {"FreqDetuning": 0.15e6, "Amp": 0.17},  # RNR re-opt 2026-08-19 (was 0.12e6/0.14 @ 07-20)
                "h": {"FreqDetuning": 0.15e6, "Amp": 0.17},  # RNR re-opt 2026-08-19 (was 0.12e6/0.14 @ 07-20)
            },
            # Post-rearrangement recool before STIRAPPushoutStep (RearrangeCool556hXStep). Seeded to
            # the Cool556 X/h values ABOVE, which is exactly what the 2026-08-06 A/B ran
            # (20260806_000351: mid->final 0.514 vs 0.341 with the old top-level Cool556Step values),
            # so this overlay reproduces that measurement bit-for-bit. Now scannable on its own axis
            # via RearrangeSTIRAPScan.py -- the STIRAP atoms arrive hotter than the RNR ones (SLM
            # transport at RearrCoolAmp = 0 + two 399 exposures), so a longer/stronger recool than the
            # RNR-tuned numbers is the open hypothesis for the residual (0.783 at 13.5 us total
            # trap-off vs RNR's 0.943 at 13 us).
            "RearrangeCool556": {
                "Time": 5e-3,
                "X": {"FreqDetuning": 0.12e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.12e6, "Amp": 0.14},
            },
        },
        # 2026-07-10: 33x33_feedback10 = PRODUCTION successor to 33x33_feedback9 after the optical-path
        # move. It IS the 33x33_feedback9b_r3 keeper phase (identical .pt, sha e5995eaab4a7), renamed for
        # production. Amplitude-scaling depth re-flatten restored CV 7.05% (post-move) -> 2.22% (split-half
        # true ~1.5%) in 3 rounds. Overlay = EXACT COPY of 33x33_feedback9 (depth-only campaign; cooling/
        # imaging unchanged, already optimized). Phase phase/33x33_feedback10.pt. See CAMPAIGN_STATE_fb9b.md.

        # 2026-07-07: new array with 14.5um spacing for Rydberg
        "23x23_14p5um": {
            "Orca": {"ExposureTime": 0.035},
            "Init": {"VSLMServo": 3.2},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.22,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.26},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        
        # 2026-07-07: new array with 20um spacing for Rydberg
        "17x17_20um": {
            "Orca": {"ExposureTime": 0.050},
            # 2026-08-10: kept at the NORMAL loading/imaging depth. An imaging re-optimization was
            # done at VSLMServo 3.5 and then REVERTED (user directive): the plan is to load and
            # image at this depth and RAMP the trap up only just before the ping-pong transport
            # step, so the imaging W must stay the one that matches THIS depth. What the 3.5 pass
            # found, should it ever be needed: det +14e6 (vs -5e6 here), DDS amps 1.0/1.0,
            # PIDSet 0.80/1.00, Cool556 X (0.50e6, 0.24) / h (0.62e6, 0.08), loading plane z4 -3.5,
            # giving load 0.596 / fidelity 0.9917 / d' 3.50 / survival 0.9355 (r800-r809,
            # data_20260810_161608 .. _164212). It never met the >=99.5% per-site fidelity gate
            # (median 0.987, 13% of sites >= 0.995) because of a +0.032 survival gradient across y
            # = a 399/556 beam-alignment issue on this array, which is worth fixing regardless of
            # which depth it runs at. Note the mj=0 line sits at 107.8705 MHz here (scan
            # 20260810_170103), NOT the 108.05 the base config carries from the 33x33.
            "Init": {"VSLMServo": 0.6},
            # 2026-08-10 imaging optimized AT THE LOADING DEPTH (VSLMServo 0.6, ~400 uK) and
            # COMMITTED here -- every scan that is not imaging_round (Spectrum556Scan, the
            # transport scans, ...) reads this block, so leaving the W as g()-overrides meant those
            # ran with the starved pre-ND values. Baseline with the old numbers: load 0.057,
            # d' 2.66, survival 0.140 (r820, data_20260810_181142) -- the Imag399 block dated from
            # 07-07, BEFORE the 07-18 ND-filter recal, so the atom-plane power was ~5x too low.
            # det -5e6 -> -2e6 at restored power: a -12..+12 sweep (r821, data_20260810_181244)
            # peaked at -2 (d' 2.71, survival 0.831, load 0.444), a much smaller move than the
            # +14e6 the same array wanted at VSLMServo 3.5 -- the light shift scales with depth.
            # COOLING was the big lever here. X (r822, data_20260810_181447, 0.06-0.34 x 0.10-0.40):
            # survival collapses to ~0.5 at det 0.06 and peaks 0.9856 at (0.18, 0.25) -- the
            # inherited 0.16 sat right on the cliff edge. h (r823, data_20260810_181858, with X
            # pinned): best (0.26, 0.08) -> survival 0.9923 / fidelity 0.9963 / d' 4.30.
            # LOADING PLANE also moves with depth: a 17-plane z4 sweep (scan 20260810_182313) peaks
            # at -7 (dist 6.32) / -6 (d' 4.98, parabolic -6.28) vs the inherited -5 (dist 5.20),
            # i.e. +22% separation -> -6.5. (At VSLMServo 3.5 the same array wanted -3.5.)
            # 100-shot verify (r824, data_20260810_182553): load 0.569, per-site fidelity median
            # 0.99829 with 85.6% of sites >= 0.995, d' median 5.28, survival 0.9865 +- 0.0015,
            # SPATIALLY FLAT (dx -0.004, dy +0.004). NOTE the +0.032 y-gradient seen at VSLMServo
            # 3.5 is ABSENT here, so it was depth-dependent, not the fixed beam misalignment it was
            # first called. Survival sits just under the 99% goal only because sites 126 and 157 are
            # dead (0.52/0.53 at d' 2.8-2.9, chronic shallow traps also worst at 3.5 V); two such
            # sites out of 284 cost 0.34%, so the array excluding them is ~0.990.
            "BlueMOT": {"LoadingTime": 300e-3, "Img1PIDSet": 0.8, "Img2PIDSet": 1.0},
            "GreenMOT": {"CoolDown": {"HoldTime": 150e-3}},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            "SLM": {"Loading": {"Defocus": -6.5}},
            "Imag399": {
                "FreqDetuning": -2e6,       # 2026-08-10 (was -5e6)
                "Amp1": 1.0, "Amp2": 1.0,   # 2026-08-10 (were 0.11/0.22 = pre-ND-recal power)
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.18e6, "Amp": 0.25},  # 2026-08-10 (was 0.16e6/0.26)
                    "h": {"FreqDetuning": 0.26e6, "Amp": 0.08},  # 2026-08-10 (was 0.16e6/0.14)
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        
        # 2026-07-04: NEW two-layer bifocal array 2x11x11_5um (phase/2x11x11_5um.pt): 11x11 grid
        # duplicated at TWO axial planes z4 = +-2.7778 rad about the stack center, SAME xy for both
        # layers (pure bifocal stack -- camera boxes catch both layers; per-layer readout needs the
        # loading defocus moved to -5 +- 2.7778). Entry = EXACT COPY of 33x33_feedback9's current
        # params (user 07-04) except Init.VSLMServo 0.39 (242 traps vs 1068 -- proportionally less
        # total power). Per-pattern thresholds start fresh (seeded flat 202.5).
        "2x11x11_5um": {
            "Orca": {"ExposureTime": 0.035},
            "Init": {"VSLMServo": 0.39},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            # 2026-07-14: migrated to the PID-servo imaging scheme (399 power via
            # BlueMOT.Img1/Img2PIDSet, DDS Imag399.Amp1/Amp2 held at 1 -- the stale legacy
            # 0.23/0.22 DDS amps starved the imaging light ~4x under the new scheme and were
            # the likely cause of the back2um d'~2.8). Setpoint seed = feedback11's 35 ms
            # optimum (1.0/0.35); to be re-optimized per-array.
            "BlueMOT": {"Img1PIDSet": 1.0, "Img2PIDSet": 0.35},
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 1, "Amp2": 1,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.26},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },

        
        
        "3270_tri": {
            "Orca": {"ExposureTime": 0.100},  # 2026-06-29: 35->50->75->100ms. Each step lifts survival ceiling (0.83->0.92->0.94) + fidelity/d' (limit was dim imaging, not cooling). 75ms W: amps 0.22/0.22, cool X(0.18,0.24)/h(0.18,0.18), surv ~0.94 fid 0.987 d'3.7. Pushing to 100ms (survival still < 99% gate).
            "Init": {"VSLMServo": 3.5},
            # 2026-07-01: per-pattern MOT loading raised 0.7 (global) -> 0.9 for this large 3270-site
            # array only, to load the wide array more fully (user request).
            "BlueMOT": {"LoadingTime": 0.9},
            "GreenMOT": {"CoolDown": {"HoldTime": 0.3}},  # 2026-07-01: per-pattern MOT loading raised 0.7 (global) -> 0.9 for this large 3270-site array only, to load the wide array more fully (user request).
            "LAC": {"Time": 25e-3},
            "Imag399": {
                # 2026-06-29 imaging optimization for 3270_tri (the prior values were copied from
                # 33x33_feedback9 -- never tuned for this array). Full campaign at 100 ms exposure
                # (raised 35->50->75->100 ms; each step lifted survival ceiling + d'/fidelity, the
                # limit being dim imaging not cooling) and at the re-found focus z4 = -1.2 (the array
                # was DEFOCUSED at -5; a z4 scan peaked at -1, then drifted to -1.2 -- thermal lensing,
                # separation dMu 7.2->4.9 over the session, so the loading defocus needs periodic re-scan).
                # Amp1 0.24->0.19, Amp2 0.26->0.14 (r237 amp scan at z4=-1; plateau, survival turns over
                # ~amp1 0.25+). X cooling 0.158/0.267 -> 0.18/0.21, h 0.14/0.24 -> 0.16/0.17 (0-pushout
                # confirms at 100 ms). FINAL per-site (r255, z4=-1.2, 100 shots): per-site survival 0.961,
                # per-site fidelity (analytic Gaussian-overlap) median 0.985 / d' 3.84. BELOW the 99%/99.5%
                # gate -- separation-limited by (a) thermal-lensing focus drift, (b) a chronic shallow
                # BOTTOM-LEFT trap region (BL survival 0.921 vs rest 0.964, X/Y MOT-position independent
                # -> trap-depth/SLM, needs trap-depth feedback). MOT left at default (X 0.0385/Y 0.265).
                # 2026-06-30 RE-OPTIMIZED on the now depth-BALANCED array (after the trap-depth feedback
                # campaign flattened CV 11%->1.54%). Focus re-scanned z4 -1.2 -> -1.5 (drifted, thermal
                # lensing). Amp1 0.19->0.28 (r3/r4 amp scan: fid plateaus + survival turns over >0.28; low
                # amp = detection-limited). Cooling re-opt (the win): X(0.18,0.21)->(0.16,0.17), h det
                # 0.16->0.20 amp 0.17 -- drove d' 3.60->4.59. The old shallow-BL d' deficit is GONE (depth
                # flat); fidelity is now globally separation-limited (cooling, not photons). FINAL per-site
                # (job 1140, z4=-1.5, 150 shots): fidelity median 0.9965 (72.7% sites >=0.995, GATE MET),
                # d' 4.59, survival mean 0.967 / median 0.988. Residual survival = scattered hi-d' loss +
                # mild BL 0.955 (hardware: 556/imaging-beam align or trap depth) + the 100 ms heating budget
                # (exposure-shorten deferred: it's a GLOBAL live-camera change, shared with rearrange).
                "FreqDetuning": -5e6, "Amp1": 0.28, "Amp2": 0.14,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.17},
                    "h": {"FreqDetuning": 0.20e6, "Amp": 0.17},
                },
            },
            "Cool556": {
                # 2026-06-29 RNR (release-recapture) cooling re-opt at 30 us release (focus z4=-1.2):
                # X r260 det 0.16->0.14 (amp 0.14 kept), h r261 det 0.16->0.14 (amp 0.12 kept) -- both
                # detunings to 0.14 MHz peaked recapture survival (X 0.476 / h 0.502 @30us, +~0.03 ~8 SEM
                # = colder atoms). Loading flat ~0.58 (cooling-independent). Amps unchanged.
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.14e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.14e6, "Amp": 0.12},
            },
        },
    }

    # 2026-07-05: "2x11x11_5um_3d" = the 242-site (per-plane, no-dedup) DETECTION alias of the
    # bifocal array, used by the layer-isolation rearrangement runs (extras.initial_pattern keys
    # BOTH the rearrange detector calibration AND the per-bseq config overlay -> alias must exist
    # here or those runs fall back to base config). Same physical array -> same dict object.
    c["ByPattern"]["2x11x11_5um_3d"] = c["ByPattern"]["2x11x11_5um"]

    # 2026-07-14: "2x11x11_5um_back2um" = the NEW bifocal array (phase/2x11x11_5um_back2um.pt) whose
    # two 11x11 layers focus at z4 = -5 +- 2.7778 (same axial offset as 2x11x11_5um) but are OFFSET
    # in xy by ~4.6 knm px (front/back layers laterally separated, NOT xy-coincident). Because the
    # layers are xy-separated they are BOTH read from ONE image at ONE loading defocus via a 242-site
    # NO-DEDUP detection grid (dedup would merge the 4.6-px pairs). Layer-focus crossover measured
    # 2026-07-14: FRONT peaks z4 ~-3.5, BACK ~<-9, equal-focus point -6.5 (NOT the geometric -5).
    # Params seeded from 2x11x11_5um then re-tuned FOR THIS PATTERN ONLY (own deep copy, NOT the
    # shared-dict alias the _3d name uses -- edits here must not leak to 2x11x11_5um):
    #   - Orca.ExposureTime 0.035 -> 0.050 (2026-07-14: at -6.5 crossover fid ~0.92/d'~2.8/surv
    #     ~0.64 photon-limited; more exposure to buy separation. Remember the LIVE camera is a
    #     separate global setting -- camera_apply_settings.)
    #   - 07-14 optimization campaign at 50 ms / defocus -6.5 (setpoint map 20260714_232556,
    #     X-cool 233319, h-cool 234018, 150-shot confirm 234903): W = Img1 0.625 / Img2 0.35
    #     (plateau Img1 0.6-1.1 x Img2 0.15-0.35; heating cliff at max dose), X (0.14 MHz, 0.26)
    #     (inherited was near-optimal), h (0.13 MHz, 0.20) (inherited amp 0.14 was WEAK -- the
    #     big cooling gain). Confirm: fid 0.940/0.948, d' 3.08/3.13, surv 0.759/0.789 F/B
    #     (from 0.929/2.92/0.694 at the old scheme). Still photon-limited.
    #   - Init.VSLMServo: step-2 bump 0.39 -> 0.5 tested 2026-07-15 (baseline 20260715_001135,
    #     X re-opt 001425 + edge 002404 (det 0.02 catastrophic 0.41; optimum X (0.10,0.26)),
    #     h re-opt 002630 (flat), final confirm 20260715_003338: F 0.720/B 0.779) -- NO GAIN vs
    #     0.39 (F 0.759/B 0.789, confirm 20260714_234903), front slightly worse -> REVERTED to
    #     0.39 with the 0.39-optimal cooling below. Depth is not the remaining-loss lever.
    import copy as _copy
    c["ByPattern"]["2x11x11_5um_back2um"] = _copy.deepcopy(c["ByPattern"]["2x11x11_5um"])
    _b2 = c["ByPattern"]["2x11x11_5um_back2um"]
    _b2["Orca"]["ExposureTime"] = 0.050
    _b2["Init"]["VSLMServo"] = 0.39
    _b2["BlueMOT"]["Img1PIDSet"] = 0.625
    _b2["BlueMOT"]["Img2PIDSet"] = 0.35
    _b2["Imag399"]["Cool556"]["X"] = {"FreqDetuning": 0.14e6, "Amp": 0.26}
    _b2["Imag399"]["Cool556"]["h"] = {"FreqDetuning": 0.13e6, "Amp": 0.20}

    # ---- cross-references (mirror expConfig.m's const-to-const assignments) ----
    return expConfig_helper.apply_cross_refs(c)


def _default_vals(consts):
    """Per-channel default values (``expConfig.m`` ``defaultVals``; Hz / s).

    ``consts`` (the ``_consts()`` dict) is passed in so a default can cross-reference a
    constant (e.g. ``VSLMservo`` <- ``Init.VSLMServo``), mirroring expConfig.m's
    ``defaultVals(...) = consts....`` lines. Values are float-coerced downstream by
    ``SeqConfig`` (so an int constant like ``Init.VSLMServo`` becomes a float here too)."""
    d = {}
    # TTLs
    d["TTLThorCamTrig"] = 0
    d["TTLOrcaTrig"] = 0
    d["TTL556ImagingShutter"] = 0
    d["TTL556MOTaShutter"] = 1
    d["TTL556MOTbShutter"] = 1
    d["TTL556MOTcShutter"] = 1
    d["TTL399AbsImagShutter"] = 0
    d["TTL399Imag2Shutter"] = 0
    d["TTL399MOTShutter"] = 1
    d["TTL3992DMOTShutter"] = 1
    d["TTL556RydAWG"] = 0
    d["TTL308RydAWG"] = 0
    d["TTLScopeTrig"] = 0
    d["TTLQickTrig"] = 0
    d["TTLSampleAndHold"] = 1
    d["TTL399IMG1PIDMode"] = 0
    d["TTL399IMG2PIDMode"] = 0
    # DDS
    d["Freq556MOTX"] = 118e6
    d["Amp556MOTX"] = 0
    d["Freq556RydbergMOTh"] = 110e6
    d["Amp556RydbergMOTh"] = 0
    d["Freq556RydbergHF"] = 90e6
    d["Amp556RydbergHF"] = 0
    d["FreqBlueMOT"] = 270e6
    d["AmpBlueMOT"] = 0.85
    d["Freq369"] = 250e6
    d["Amp369"] = 0
    d["FreqAbsImag"] = 320e6
    d["AmpAbsImag"] = 0
    d["Freq399Imag2"] = 320e6
    d["Amp399Imag2"] = 0
    d["Freq2DMOT"] = 290e6
    d["Amp2DMOT"] = 1
    d["FreqZeeman"] = 273.5e6
    d["AmpZeeman"] = 0.6
    d["FreqSLM"] = 120e6
    d["AmpSLM"] = 0.4
    d["FreqAOM308"] = 200e6
    d["AmpAOM308"] = 0
    
    # NI DAQ
    d["VSLMservo"] = consts["Init"]["VSLMServo"]  # default SLM servo to its Init value
    d["VElectrode1"] = 0
    d["VElectrode2"] = 0
    d["VElectrode3"] = 0
    d["VElectrode4"] = 0
    d["VElectrode5"] = 0
    d["VElectrode6"] = 0
    d["VElectrode7"] = 0
    d["VElectrode8"] = 0
    d["VImg1PIDSet"] = consts["BlueMOT"]["Img1PIDSet"]
    d["VImg2PIDSet"] = consts["BlueMOT"]["Img2PIDSet"]
    # EOM616 (FreqEOM616Old is a MemoryMap runtime override in MATLAB; default here)
    d["FreqEOM616"] = 200e6
    d["AmpEOM616"] = 0.7
    d["FreqAOM616"] = 120e6
    d["AmpAOM616"] = consts["AOM616Divert"]["Amp"]  # default AOM616 to the divert amplitude
    d["FreqAODs"] = 80e6
    d["AmpAODs"] = 0
    d["FreqAODv"] = 89.3e6
    d["AmpAODv"] = 0.14
    return d
