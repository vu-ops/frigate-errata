# Errata

Active-learning pipeline for [Frigate](https://frigate.video). Errata harvests
Frigate events, flags likely detector mistakes (false positives,
label/description mismatches, low-confidence noise bursts), lets you correct
them in a web dashboard, and turns those corrections into a YOLO model
(`yolo11n`) that runs inside Frigate as a local ONNX detector.

It is a self-hosted alternative to the Frigate+ subscription: your imagery and
corrections stay on your own hardware, and the trained model runs locally.

## How it works

```
Frigate API ──▶ harvester ──▶ SQLite ──▶ analyzer ──▶ review queue (/errata)
 (events,         every N min            (mismatch /        │
  snapshots)                             low confidence /   ▼
                                         noise bursts)   user corrects labels
                                              │                │
                                              └── corrections ─┘
                                                     │
                                              trainer (UI / CLI / Mac kit)
                                                     │
                                          YOLO dataset → train → ONNX
                                                     │
                                      models/published → Frigate detector
```

1. **Harvest** — every `harvest.interval_minutes` (default 30), pull completed
   events from the Frigate API since the last processed timestamp (resumable),
   download clean snapshots (no detection overlay), and store them in SQLite.
2. **Analyze** — flag events for review with four rules:
   - `mismatch`: vocabulary coherence — the detector label doesn't appear in the
     GenAI event description (e.g. label `cat`, description "gray raccoon").
     High-confidence mismatches sort first.
   - `low_confidence`: detector confidence below
     `analysis.low_confidence_threshold`.
   - `noise`: the same camera + label firing more than `analysis.noise.count`
     times within `analysis.noise.window_hours`.
   - `oversized`: the box covers more than `analysis.oversized_box.max_area` of
     the frame (merged detections, e.g. two cars under one box in IR night
     frames). Oversized events are never auto-confirmed and never become
     training pseudo-labels.
   Events below `harvest.min_confidence` are auto-ignored; `analysis.safe_labels`
   (person, car, ...) skip the description check.

   Auto-confirmation depends on Frigate GenAI descriptions: with
   `analysis.confirm_on_coherent_description`, a detection that would otherwise
   be flagged for low confidence or noise is auto-confirmed only when its GenAI
   description mentions the label (or a synonym), and training pseudo-labels
   are drawn from those same coherent descriptions. Without GenAI descriptions,
   no event is auto-confirmed by description — flagged detections stay in the
   review queue.
3. **Review** — open the dashboard, confirm the detection, pick the correct
   label, or mark it a false positive. **Edit box** opens any event's snapshot so
   you can move, resize, or draw the box the correction trains on. Camera/Label/
   Reason filters narrow the queue, and **Apply to matching** bulk-applies a
   status. Already-reviewed events can be reopened, re-labelled, or sent back to
   the queue.
4. **Train** — once enough corrections accumulate (`training.trigger_threshold`,
   default 50), build a YOLO dataset and train; see below.

## Features

- Automated review queue with priority ordering
- Synonym-aware label/description mismatch detection (configurable per class)
- Noise-burst, low-confidence, and oversized/merged-box detection
- In-browser box editor: move, resize, or draw the training box on any event
- Three training paths: in-app button, CLI, or a Mac/Colab kit
- Optional automatic deployment of the trained model into Frigate's config
  (with backup + rollback)
- Snapshot retention/pruning with per-label budgets
- Prefix-aware web UI, designed to sit behind your existing reverse proxy

## Quick start

Prerequisites: Docker + Docker Compose on the Frigate host, and a running
Frigate (0.14+; developed and tested against 0.18) reachable from the Errata
container by container name on a shared compose network.

Frigate 0.16+ GenAI descriptions are **required for description-based
auto-confirm** (`analysis.confirm_on_coherent_description`), for mismatch
detection, and for training pseudo-labels. Without them Errata still works via
the low-confidence and noise rules, but flagged detections are never
auto-confirmed by description and wait in the review queue for a human.

1. **Place the project** anywhere on the Frigate host, e.g. `/opt/errata-app`.
2. **Credentials** — if Frigate has authentication enabled (0.16+), create
   `.env` from `.env.example` and set a dedicated Frigate admin user
   (`ERRATA_FRIGATE_USER` / `ERRATA_FRIGATE_PASSWORD`). If Frigate has no auth,
   leave both empty (a `.env` file must still exist if the compose snippet
   references it).
3. **Configure** `config.yaml`:
   - `frigate.api_url` must resolve from the Errata container. The default
     `http://frigate:5000` assumes the Frigate compose service is named
     `frigate` on the same network.
   - `server.base_path` must match your nginx location prefix (default `/errata`).
   - `labels.track` lists the object classes your cameras detect — it drives the
     review UI, the exported dataset, and `labels.txt`.
4. **Compose service** — copy the `errata:` service from
   `deploy/docker-compose.snippet.yml` into the compose project that runs
   Frigate and replace the `/path/to/errata-app` placeholders. If Frigate runs
   in a separate compose project, attach Errata to that network instead
   (`docker network ls`; `external: true`).
5. **Build and start**:

   ```bash
   docker compose build errata && docker compose up -d errata
   ```

   The first build downloads pip wheels and the pinned Intel GPU-stack debs, so
   network access is required at build time.
6. **nginx** — paste `nginx/errata-location.conf` inside the HTTPS server block
   that fronts Frigate and reload nginx. The app is unauthenticated by default;
   add basic auth in the location block if it is internet-exposed.
7. **Open** `https://<host>/errata/` — the queue fills on the next harvest cycle
   or immediately via the "Fetch latest" button.

## Training hardware

Errata's image ships an Intel XPU PyTorch build with automatic CPU fallback.
Set `training.device` in `config.yaml` accordingly.

| Device | Status | Notes |
|---|---|---|
| CPU | Supported | Default fallback on every host. Fine for small datasets; slow for many epochs. |
| Intel Arc (XPU) | Supported | Tested on an Arc Pro B50. Compose snippet wires up `/dev/dri/renderD128` + the render group. |
| NVIDIA (CUDA) | **Untested** | Build with `GPU_TYPE=cuda` (requires the NVIDIA Container Toolkit on the host). Set `training.device: cuda`. |
| AMD (ROCm) | **Untested** | Build with `GPU_TYPE=rocm` (requires a ROCm-capable host). Set `training.device: rocm`. |

> **Untested** configurations are provided on a best-effort basis. They may need
> extra host setup beyond what Errata ships; issues and PRs from anyone who gets
> them working are welcome.

Build the image for a specific accelerator by passing the `GPU_TYPE` build arg:

```bash
docker build --build-arg GPU_TYPE=cuda -t errata:local .   # NVIDIA
docker build --build-arg GPU_TYPE=rocm -t errata:local .   # AMD
docker build --build-arg GPU_TYPE=cpu  -t errata:local .   # CPU only
```

The default is `GPU_TYPE=xpu` (Intel). On a host without the matching hardware,
PyTorch falls back to CPU at runtime.

## Training and deploying a model

1. Confirm some events in the queue (pick the right label, mark false positives,
   or skip). Corrections accumulate in `data/errata.db`.
2. Train — pick one:
   - **Train button** (UI header): runs inside the Errata container on the
     configured device.
   - **Mac kit** (UI header): a zip with the exported dataset plus `train-mac.sh`
     and a README. Trains on a Mac (Apple-Silicon MPS by default, CPU fallback),
     or in the bundled Google Colab notebook. Copy the resulting
     `published/errata_<stamp>.onnx` + `labels.txt` back into
     `models/published/` on the server.
   - `./deploy/train.sh` (CLI, CPU container; needs a `NETWORK` env var or the
     default `frigate_default` network).
3. A successful train writes the ONNX, `labels.txt`, and `metrics.json` to
   `models/published/` and lists the version on the Summary page.
4. **Point Frigate at it** (only after a successful train). In Frigate's
   `config.yml`, replace the model block (keep the old path commented for
   rollback):

   ```yaml
   model:
     # path: plus://...            # previous model, kept for rollback
     path: /config/models/published/errata_<stamp>.onnx
     width: 640
     height: 640
     input_dtype: float
     labelmap_path: /config/models/published/labels.txt
     model_type: yolo-generic
   ```

   Apply via the Frigate UI (Config → Save & Restart) or
   `POST /api/config/save?save_option=restart`.

   Frigate-version gotchas (verified on 0.18):
   - `input_dtype` must be `float` — `float32` fails schema validation.
   - `model_type: yolo-generic` is required for Ultralytics YOLO exports (the
     default SSD parser rejects the `[1, classes+4, 8400]` head).
   - The exported ONNX must accept NHWC input `(1, H, W, 3)`. Errata's trainer
     and the Mac kit convert the export automatically (a Transpose is inserted
     at the graph input). A hand-exported `model.export(format="onnx")` will
     crash the Frigate detector — run it through the provided tooling.
5. Validate: Frigate restarts, the log shows `Loading OpenVINO model ...` with
   no traceback, and `/api/stats` reports a finite `inference_speed`. Rollback =
   restore the commented model line and reload.

## Operational notes

- **Never train on annotated snapshots.** Frigate burns the detection box and a
  confidence label into the default event snapshot; a model trained on those
  learns the overlay and scores poorly on clean frames. Errata always fetches
  `/api/events/<id>/snapshot.jpg?bbox=0` — keep it that way. If snapshots were
  ever fetched annotated, refresh them with
  `python -m errata.trainer --refresh-snapshots` then `--rebuild`.
- The review UI draws the detection box on the fly when serving snapshots; the
  on-disk image stays clean, so training never sees the overlay.
- Retention (`nightly prune`, once per 24h) caps three buckets at
  `review.auto_keep_per_label` most-recent per label; false positives are kept
  forever. Pending and skipped events are never pruned.
- `config.yaml` changes require `docker compose restart errata`.
- Errata's mismatch and auto-confirm rules read Frigate GenAI event
  descriptions. GenAI is required for description-based auto-confirm
  (`analysis.confirm_on_coherent_description`) and for training pseudo-labels;
  without it Errata still flags low-confidence and noise events, but nothing
  self-confirms on description and every flagged detection waits for review.

## Configuration

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for the full reference:
every config key, environment variables, DB schema, API reference, and
troubleshooting.

## License

MIT — see [`LICENSE`](LICENSE).
