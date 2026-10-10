# Errata

> **v0.2 is a breaking release — start over.** The detector class taxonomy
> changed: brands (usps, ups, fedex, amazon, dhl, gls) and `license_plate` are no
> longer detector classes (they are attributes/metadata), and the packaged
> defaults now live in `errata/settings.yaml`. Existing databases, datasets, and
> models are incompatible — use the **Reset** button (or `python -m errata.trainer
> --reset`) to wipe learned data and begin again from a base model.

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

- Automated review queue with priority ordering, split into independent
  **object** and **brand** items
- **Brands as attributes** (usps, fedex, …) — reviewed separately, never trained
  as detector classes (a USPS van is `car` + brand `usps`)
- **Candidate search** — flag a new label (e.g. coyote) and Errata scans GenAI
  descriptions for its synonyms, surfacing matches for review so you can grow the
  label before enabling training
- **Per-label controls** — collect mode, GenAI Description Search, include in
  training, auto-approve, include-auto-approved-in-training, and Snapshot Limit,
  with a column-header bulk apply
- **GenAI Help** — a second opinion on any review item via OpenRouter
  (Gemini Flash / Qwen Flash): suggested label, description, and an optional
  bounding box that pre-fills the box editor
- Synonym-aware label/description mismatch detection (configurable per class)
- Noise-burst, low-confidence, and oversized/merged-box detection
- In-browser box editor + **training-crop preview** (the exact crop the model
  learns from; Edit opens the full frame to re-crop)
- **Base models**: packaged YOLO 9/11/12 variants, uploads, and one-click
  Frigate+ model import (deploy-only)
- **Reset / start over** — wipe all learned data and controls with one action
- Three training paths: in-app button, CLI, or a Mac/Colab kit
- Optional automatic deployment of the trained model into Frigate's config
  (with backup + rollback)
- Snapshot retention/pruning with per-label budgets
- Prefix-aware web UI, designed to sit behind your existing reverse proxy

## Labels: objects, brands, and candidates

- **Object classes** (`labels.track` in `errata/settings.yaml`) are the only
  YOLO classes. Order defines class indices and `labels.txt`.
- **Brand attributes** (`labels.attributes`) are metadata: a delivery van is
  detected as `car` and separately carries a brand item (`usps`). Brands are
  never detector classes, so a class can't mix person-shaped and van-shaped
  boxes.
- **Candidates** are object classes with `include_training` off and `search` on.
  While search is on, the analyzer scans GenAI descriptions of new events for the
  label's synonyms (across all detector labels) and surfaces matches as normal
  review items — the way to build up a new label like `coyote` before training
  it. Turn `include_training` on when you have enough samples; class indices
  change, so retrain.

Per-label controls live on the **Controls** page (collect mode, search,
train/collect-only, auto-confirm, pseudo-labels). A candidate hit always
surfaces, overriding the detected label's auto-confirm.

## Reset / start over

The Summary page's **Reset** action deletes all events, snapshots, corrections,
brand items, label controls, the dataset, published models, and Frigate config
backups, then disables every per-label control so you can focus one label at a
time. Base models are kept. Optionally point Frigate at a base or kept trained
model first; one final Frigate config backup is saved afterward. CLI equivalent:
`python -m errata.trainer --reset`.

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
3. **Configure** `config.yaml` (a thin override file — the container ships the
   full baseline at `errata/settings.yaml`, so list only what you change):
   - `frigate.api_url` must resolve from the Errata container. The default
     `http://frigate:5000` assumes the Frigate compose service is named
     `frigate` on the same network.
   - `server.base_path` must match your nginx location prefix (default `/errata`).
   - `harvest.lookback_hours` defaults to `0` — Errata trains *going forward*
     and never backfills history unless you opt in with a positive value.
   - `labels.track` / `labels.attributes` in `settings.yaml` list object classes
     and brand attributes; `labels.track` order drives the exported dataset and
     `labels.txt`. Synonym lists are **appended** (your entries plus the
     packaged baseline).
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

The image is built in two parts so per-commit builds stay fast:

1. **Base image** (`Dockerfile.base`) — stock Python + system libs + the GPU
   user-space and training stack (torch/ultralytics). Heavy (~8.7 GB) and rarely
   changed; CI publishes it to GHCR and publishes a new tag only when
   `BASE_VERSION` is bumped.
2. **App image** (`Dockerfile`) — `FROM` the base and adds only the app deps and
   code (a few tens of MB), so it builds in seconds.

Build the base for a specific accelerator, then the app image on top:

```bash
# Base (pick one backend)
docker build -f Dockerfile.base --build-arg GPU_TYPE=cuda -t errata-base:local .   # NVIDIA
docker build -f Dockerfile.base --build-arg GPU_TYPE=rocm -t errata-base:local .   # AMD
docker build -f Dockerfile.base --build-arg GPU_TYPE=cpu  -t errata-base:local .   # CPU only

# App image on top of the base you just built
docker build --build-arg BASE_IMAGE=errata-base:local -t errata:local .
```

The default backend is `xpu` (Intel). On a host without the matching hardware,
PyTorch falls back to CPU at runtime.

## Training and deploying a model

1. Confirm some events in the queue (pick the right label, mark false positives,
   or ignore). The grid supports multi-select (checkbox, shift-click, drag) with
   a bulk toolbar — confirm, mark false positives, ignore, or change the label.
   Corrections accumulate in `data/errata.db`.
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
     width: 320                    # must match the model's training imgsz
     height: 320                   # must match the model's training imgsz
     input_tensor: nhwc            # Errata exports channels-last (NHWC)
     input_dtype: float
     labelmap_path: /config/models/published/labels.txt
     model_type: yolo-generic
   ```

   Apply via the Frigate UI (Config → Save & Restart) or
   `POST /api/config/save?save_option=restart`.

   Frigate-version gotchas (verified on 0.18):
   - `width`/`height` must match the image size the model was exported at
     (`training.imgsz`, 320 or 640). Errata writes these automatically when it
     activates a model; set them by hand only if you edit the config yourself.
   - `input_dtype` must be `float` — `float32` fails schema validation.
   - `model_type: yolo-generic` is required for Ultralytics YOLO exports (the
     default SSD parser rejects the `[1, classes+4, 8400]` head).
   - The exported ONNX must accept NHWC input `(1, H, W, 3)` — set
     `input_tensor: nhwc`. Errata's trainer and the Mac kit convert the export
     automatically (a Transpose is inserted at the graph input). A hand-exported
     `model.export(format="onnx")` will crash the Frigate detector — run it
     through the provided tooling.
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
- Retention caps auto-confirmed snapshots at `review.auto_keep_per_label`
  most-recent per label (nightly). Human corrections and false positives are
  ground truth and kept forever; pending events are never pruned. **Ignored**
  events (including events of labels set to `collect_mode: off`) are deleted
  with their snapshots once older than `review.keep_ignored_hours` (default 24).
  Events whose snapshot file is missing are deleted from the database.
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
