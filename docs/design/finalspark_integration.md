# FinalSpark Neuroplatform Integration — Design / Scope

Status: **proposed** · Target: `m9h/bl1` · Owner: m9h

BL-1 already contains everything needed to drive a closed-loop biological-intelligence
experiment *in silico*. This document scopes connecting that same machinery to **live
human-neuron organoids** on the [FinalSpark Neuroplatform](https://finalspark.com/neuroplatform/),
so the identical encode → stimulate → record → decode → feedback loop runs against either a
simulated culture (the digital twin) or real wetware.

## 1. Goal

Run BL-1's closed-loop game experiments (starting with Pong) against a FinalSpark organoid,
reusing BL-1's sensory encoding, motor decoding, game environments, feedback modes, logging,
and analysis unchanged — with the simulator serving as a **dry-run digital twin** to tune and
de-risk every experiment before spending scarce, shared wetware time.

## 2. What is reused unchanged

| BL-1 component | Role in the integration |
| --- | --- |
| `loop/encoding.py` (`encode_sensory`) | game state → stimulation pattern (quantized, see §5.1) |
| `loop/decoding.py` (`decode_motor`)   | neural activity → action (count-based variant, see §5.3) |
| `games/pong.py`, `games/doom.py`      | environments + FEP / open-loop / silent feedback |
| `compat/cl_sdk.py`                    | reference for the "backend behind a stim/spike interface" pattern |
| `monitor/`, `analysis/`               | dashboards, raster/heatmap, burst & criticality metrics |
| `validation/`                         | sim-vs-real comparison framework (now sim-vs-organoid) |

This is a **new backend + slow-loop driver**, not a rewrite.

## 3. The FinalSpark API (as scoped)

From the [NeuroPlatform docs](https://finalspark-np.github.io/np-docs/):

- **Connect:** `Experiment(token)` → `exp.start()` / `exp.stop()`; available electrodes via `exp.electrodes`.
- **Stimulate:** `StimParam` (biphasic: `phase_duration1/amplitude1`, `phase_duration2/amplitude2`,
  `polarity`, `nb_pulse`, `pulse_train_period`, charge-recovery/settle fields), bound to a
  `trigger_key` (0–15). `IntanSofware.send_stimparam([...])` uploads the bank — **a ~10-second
  blocking call**. Firing is `Trigger.send(np.uint8[16])`.
- **Closed-loop read:** `IntanSofware.set_count()` selects counting triggers; `Trigger.send()`
  starts a **200 ms** counting window; `IntanSofware.read_count()` returns a **128-element
  per-electrode spike-count array**. Polling, **~1 Hz** loop, *not* event-driven.
- **Historical / analysis:** `Database().get_spike_event(start, stop, fs_name)` (Time, Amplitude,
  channel), `get_spike_count(...)` (per-minute counts/electrode), `get_raw_spike(...)` (3 ms @ 30 kHz).
- **Hardware:** 4 MEAs × 32 electrodes; **8 electrodes per organoid**; electrode indices 0–127.
  Spike = voltage crossing 6×SD of noise.

### Deployment constraint
The core `neuroplatform` package is **not public** — it is provided inside FinalSpark's supplied
environment. Therefore `FinalSparkBackend` and the driver **run inside FinalSpark's environment**
(python ≥ 3.11; BL-1 supports 3.10+). The `SimBackend` digital twin runs anywhere. Public helper
`np-utils` (`StimParamLoader`, etc.) installs via
`pip install "git+https://github.com/FinalSpark-np/np-utils.git#egg=np_utils[all]"` and is useful
for managing the trigger bank.

## 4. Architecture

```
src/bl1/hardware/                # new subpackage
  base.py          # NeuralBackend Protocol
  finalspark.py    # FinalSparkBackend  -> neuroplatform (env-provided)
  sim_backend.py   # SimBackend (wraps cl_sdk/ClosedLoop; organoid geometry; count-coded)
  electrode_map.py # organoid sensory/motor electrode assignment + StimParam trigger bank
  fake.py          # FakeNeuroplatform — in-memory mock so tests/CI run without credentials
experiments/
  finalspark_smoke.py   # M0 (shipped first)
  finalspark_pong.py    # M3
```

**Unifying abstraction — a rate-coded tick** (not spike-times), matching FinalSpark's natural
granularity and sidestepping the latency/timescale mismatch:

```python
class NeuralBackend(Protocol):
    @property
    def sensory_electrodes(self) -> list[int]: ...
    @property
    def motor_electrodes(self) -> list[int]: ...
    def configure_stim_bank(self, bank: StimBank) -> None:  # once per episode (FS: the 10 s upload)
        ...
    def tick(self, active_triggers: np.ndarray) -> np.ndarray:  # -> per-electrode spike counts
        ...
    def close(self) -> None: ...
```

`FinalSparkBackend.tick()` = `Trigger.send(active_triggers)` → wait 200 ms → `read_count()`.
`SimBackend.tick()` = advance the JAX sim by the window and bin spikes to per-electrode counts.
The driver, encoder, count-decoder, game, and feedback are backend-agnostic.

## 5. The three hard constraints (these shape the design)

### 5.1 Stimulation cannot be reparametrized per tick (10 s upload)
`send_stimparam()` costs ~10 s, so sensory encoding must be **pre-discretized into ≤16
trigger→StimParam bindings** uploaded once at episode start. Per tick the driver only chooses
*which* triggers to fire. → `encode_sensory` is wrapped by a quantizer that maps the continuous
sensory variable onto the trigger bank. **Hard caps: 16 stim patterns; 8 electrodes/organoid.**

### 5.2 ~1 Hz polling loop at biological tempo
200 ms count window + Switzerland round-trip → episodes run slowly; Pong is playable but slow.
Doom's I/O will not fit one organoid's 8 electrodes — **sim-only / multi-organoid, later**.
Logging and analysis must tolerate long, low-rate runs (BL-1 monitor already streams fine).

### 5.3 Count-based decoding + stimulation safety
`read_count()` yields counts, while `decode_motor` assumes spike-time activity → add
`decode_motor_counts()` and re-validate. Stimulation must enforce **charge balancing**
(`D1·A1 = D2·A2`), must **not reuse an electrode index across StimParams**, and must respect a
**maximum safe amplitude** (not stated in docs — *resolve with FinalSpark before any stim*).

## 6. Digital-twin strategy (BL-1's leverage)

Configure `SimBackend` with the **same 8-electrode organoid geometry, 1 Hz count-coded tick**,
and calibrate its spontaneous firing rate to the organoid's measured baseline
(`Database.get_spike_count`, mirroring the existing Wagenaar calibration). Then dry-run every
experiment — encoder quantization, electrode map, decoder thresholds, episode length — in silico
before booking wetware. Identical driver + maps run against sim or hardware.

## 7. Decisions

- **Start single-organoid (8 electrodes).** `electrode_map.py` is parameterized so a full MEA
  (32 electrodes / 4 organoids) is a later config change, not a redesign.
- **Rate-coded tick** is the cross-backend contract.
- **Token via `FINALSPARK_TOKEN` env var**, never committed; `fs_name` via arg/env.
- **CI stays credential-free** via `FakeNeuroplatform`; experiment scripts live in `experiments/`
  (outside `tests/` and `src/bl1/`, so they do not affect the lint/test/quality jobs).

## 8. Milestones

| M | Deliverable | Effort | Creds |
| --- | --- | --- | --- |
| **M0** | Connectivity smoke: connect, list `exp.electrodes`, pull spontaneous spike counts/events, plot | hours | yes |
| **M1** | Stim bank + open-loop evoked response (pre-loaded triggers, fire, confirm evoked counts) | days | yes |
| **M2** | `NeuralBackend` Protocol + `SimBackend`/`FinalSparkBackend` parity; `FakeNeuroplatform` mock + tests | days | mock |
| **M3** | Closed-loop reduced-Pong (8 electrodes, FEP feedback, slow loop, logged to monitor) | 1–2 wks | yes |
| **M4** | Calibrated twin + learning comparison (sim vs organoid) | ongoing | both |

## 9. Risks & open questions

- 10 s reconfig caps sensory resolution at 16 patterns; confirm amplitude cannot vary per
  trigger-fire without re-upload.
- 8-electrode I/O ceiling per organoid; multi-organoid neurons may not be functionally connected.
- Network/Python timing jitter (docs warn `time.sleep` is imprecise — prefer pulse-train timing).
- Count-vs-raster decoder needs revalidation against the twin.
- Shared 16-organoid resource → scheduling/contention; mandatory cleanup (disable all StimParams
  on exit).
- **Max safe stimulation amplitude/current is unspecified in the docs — must be obtained from
  FinalSpark before M1.**
