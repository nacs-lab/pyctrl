# NI PCIe-6738 — analog-output card reference (Yb experiment)

Reference digest for the analog-output device used by the Yb tweezer experiment. The MATLAB
stack drives it today via `NiDAQRunner.m` (DAQ Toolbox); the pyctrl Phase-5 `nidaq_runner.py`
port will drive it via the `nidaqmx` Python package. See sibling PDFs in this folder for the
authoritative full specs.

## Local files (this folder, gitignored)
- `PCIe-6738-Specifications.pdf` — official NI device specifications (the numbers below).
- `NI-6738-6739-User-Manual.pdf` — NI 6738/6739 User Manual (375140b): terminals, PFI routing,
  AO timing engine, sample-clock/start-trigger configuration, DAC/bank architecture.

## Key specifications
| Spec | Value |
|---|---|
| AO channels | **32**, 16-bit resolution |
| DAC architecture | banks of **4 channels per DAC** → 8 DACs; all channels on a DAC update simultaneously on each sample-clock edge |
| Max update rate | **1 MS/s** simultaneous (hardware-timed) |
| Output range | ±10 V (selectable ranges incl. ±10 V) |
| PFI lines | **PFI0–PFI15** (16), bidirectional — route external sample clock / start trigger in, or clock out |
| Onboard timebases | 100 MHz, 20 MHz, 100 kHz |
| External base clock | 0–25 MHz |
| Triggering | digital start trigger (PFI) + external sample clocking supported |
| Bus | PCIe (x1) |

## How the Yb experiment uses it (verified against config + libnacs)
- **Externally clocked, externally triggered.** The card does NOT use its onboard timebase for the
  sequence. The libnacs **zynq/FPGA backend** generates both signals into the FPGA bytecode:
  - **Sample clock** — dedicated CLOCK line, period = `step_size/2`; full cycle = `step_size` =
    2.5 µs → **400 kHz**. (config.yml `NiDAQ.step_size: 2500000` ps, `clock_device: FPGA1`.)
  - **Start trigger** — FPGA **TTL0** (`config.yml FPGA1.start_ttl_chn: 0`), rising edge ~100
    cycles before t=0. TTL0 is reserved → never appears in expConfig.m channel aliases.
- **Physical wiring** (expConfig.m): FPGA clock → NI **PFI0** (`niClocks('Dev1')='PFI0'`, ScanClock);
  FPGA TTL0 → NI **PFI1** (`niStart('Dev1')='PFI1'`, StartTrigger).
- Device name in NI-DAQmx: **`Dev1`** (channelAlias `Dev1` → `NiDAQ/Dev1`). AO channels `ao0..ao19`
  in use (card has 32).
- `NiDAQRunner.m` sets `session.Rate = 500e3` — a deliberate OVER-estimate of the real 400 kHz
  external clock (under-estimating makes the card stop updating before the sequence ends).

## nidaqmx mapping (for nidaq_runner.py — see also reference_pcie6738_nidaq memory)
Driver: **`nidaqmx`** Python package (`pip install nidaqmx`) over the NI-DAQmx runtime (already
installed; same runtime the MATLAB DAQ Toolbox uses). Device-agnostic — no 6738-specific package.

| MATLAB (NiDAQRunner.m) | nidaqmx |
|---|---|
| `daq.createSession('ni')` | `nidaqmx.Task()` |
| `session.Rate = 500e3` | `rate=500e3` arg to `cfg_samp_clk_timing` (keep the over-estimate) |
| `addAnalogOutputChannel(s,dev,chn,'Voltage')` | `task.ao_channels.add_ao_voltage_chan(f"{dev}/ao{chn}")` (same order) |
| `addClockConnection(...'External',dev/clk,'ScanClock')` | `source="/Dev1/PFI0"` arg of `cfg_samp_clk_timing` (+ `Edge.RISING`, `AcquisitionType.FINITE`, `samps_per_chan`) |
| `addTriggerConnection(...'External',dev/trig,'StartTrigger')` | `task.triggers.start_trigger.cfg_dig_edge_start_trig("/Dev1/PFI1", Edge.RISING)` |
| `queueOutputData(s, data)` data `[nsamps,nchns]` | `task.write(data, auto_start=False)` data **`[nchns,nsamps]`** — TRANSPOSE (channel-major) ⚠ |
| `startBackground(s)` | `task.start()` (non-blocking; trigger gates output) |
| `wait(s)` | `task.wait_until_done()` |

⚠ **Highest-risk porting bug:** MATLAB queues sample-major `[nsamples, nchannels]`; nidaqmx writes
channel-major `[n_channels, n_samples]`. Must transpose — wrong order scrambles channels/time
silently (no error). Covered by the NO-HARDWARE shape test.

## Docs
- Specs: https://www.ni.com/docs/en-US/bundle/pcie-6738-specs/page/specs.html
- nidaqmx Python: https://nidaqmx-python.readthedocs.io/ · https://github.com/ni/nidaqmx-python
- In-house nidaqmx examples (internal clock only): matlab_new/lib/NIDAQReadWriteLib.py
