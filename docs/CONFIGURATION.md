# Errata Configuration Reference

> **v0.2 breaking changes.** Config is now layered: the container ships the full
> baseline at `errata/settings.yaml`, and your `config.yaml` only overrides the
> keys you change. Object classes (`labels.track`) exclude brands
> (`labels.attributes`: usps, ups, fedex, amazon, dhl, gls) and `license_plate`
> — those are attributes/metadata, never detector classes. Synonym lists are
> append-only (baseline + your additions). `harvest.lookback_hours` defaults to
> `0` (train going forward). Use **Reset** (Summary page) or
> `python -m errata.trainer --reset` to start over; existing data/models are
> incompatible.

Everything needed to configure, operate, and re-deploy Errata on a fresh
host. The tool is fully self-contained in the project folder — copy it, point
it at your Frigate, build, and go.

---

## 1. Architecture recap

- **One container** (`errata`) runs the harvester loop, the analyzer, the
  review web UI, and the in-app Train button. The image ships torch XPU +
  ultralytics plus a matched Intel Level-Zero/compute-runtime stack, so the
  Train button uses the GPU on Intel Arc hosts (config `training.device: xpu`)
  and falls back to CPU anywhere else.
- **Training** can also run as a CLI step (`python -m errata.trainer`) or on a
  Mac via the downloadable kit. GPU training uses the Intel Arc GPU (XPU) when
  available — an earlier build paired a mismatched 2025 userspace with a 2026
  kernel and crashed the GPU; the current image pins a matched stack and is
  verified stable.
- **State** lives in `data/`: SQLite DB (`errata.db`), event snapshots,
  exported YOLO dataset. Delete `data/` to reset the tool entirely.
- **Models** are published to `models/` (mounted read-only into Frigate at
  `/config/models`).

---

## 2. Config file (`config.yaml`)

Loaded from (first match wins): `--config` CLI arg, `$ERRATA_CONFIG`,
`/config/errata.yaml` (default in container), `./config.yaml`.
Values support `${VAR}` and `${VAR:-default}` env expansion.
Every key below is optional; defaults shown. Restart the container after edits.

```yaml
frigate:
  api_url: "http://frigate:5000"   # from inside the compose network
  username: ""                     # Frigate auth username (empty = no auth)
  password: ""                     # Frigate auth password
  request_timeout: 30              # seconds per HTTP request
  snapshot_dir: "/data/snapshots"  # where event snapshots are stored
  config_path: ""                  # optional local path to frigate config.yml
                                   # (unused by default; API is used instead)
  backup_dir: "/data/frigate-config-backups"  # Frigate config backups
  backup_keep: 5                   # keep the N most recent config backups
                                   # (files + DB rows); older ones are pruned
                                   # whenever a new backup is written (0 = keep all)

harvest:
  interval_minutes: 30             # scheduler loop interval
  lookback_hours: 24               # window on first run / resumable window
  min_confidence: 0.50             # below this, events are auto-ignored
  max_confidence: 0.95             # above this, only mismatch flagging applies
  max_events_per_run: 500          # page size per Frigate API request
  max_pages: 20                    # pages drained per run; the harvester
                                   # pages backwards (time_to = oldest seen)
                                   # until the whole window is consumed, so
                                   # a deep lookback fully backfills in one
                                   # run (max_events_per_run x max_pages cap)

analysis:
  vocabulary_coherence: true       # flag label != description mismatches
  low_confidence_threshold: 0.75   # flag events below this confidence
  confirm_on_coherent_description: true  # when the GenAI description mentions
                                   # the label, treat it as independent
                                   # confirmation: auto-confirm even if the
                                   # detector confidence is low (skips review)
  auto_skip_implausible_boxes: true  # detector boxes that fail the plausibility
                                   # check (full-width slivers anchored on the
                                   # frame edge) are artifacts: auto-skip them
                                   # instead of queuing them for review. Set
                                   # false to fall back to flagging them as
                                   # noise in the review queue.
   oversized_box:                  # boxes covering an unusually large fraction
     enabled: true                 # of the frame are usually merged detections
     max_area: 0.4                 # (e.g. two adjacent cars under one box in IR
                                   # night frames) or whole-frame hallucinations.
                                   # When the box area exceeds max_area (frame
                                   # fraction) the event is flagged as `oversized`
                                   # for review instead of being auto-confirmed,
                                   # and is excluded from training pseudo-labels.
                                   # A single large object close to the camera can
                                   # trip this too — confirming it in the queue
                                   # trains the model on the big box as-is.
  synonyms:                        # words that count as a label match in
    person:                        # descriptions. Purely data-driven: edit
      - man                        # this list to teach the vocabulary check,
      - woman                      # no code changes needed. Plurals and
      - child                      # possessives are handled automatically
      - male                       # ("cats", "foxes", "cat's"), and built-in
      - female                     # irregulars exist for person/mouse/goose/
      - adult                      # foot/child. Brand labels (usps, ups,
      - kid                        # fedex, amazon...) are intentionally left
      - boy                        # empty: a generic "delivery van" must not
      - girl                       # validate a brand. The UI's Synonyms page
      - guy                        # edits these live: saves are stored as a
      - lady                       # database override (source of truth until
      - gentleman                  # "Reset to config.yaml baseline"), and
      - driver                     # config.yaml remains the shareable
      - pedestrian                 # baseline for fresh installs.
      - courier
      - worker
      - delivery
      - individual
      - homeowner
      - jogger
      - shopper
      - mailman
      - mail carrier
      - postman
      - delivery driver
    car: [sedan, suv, coupe, hatchback, minivan, convertible, "golf cart"]
    cat: [feline, kitten, kitty, tabby]
    dog: [canine, puppy, pup, hound]
    bird: [crow, raven, owl, sparrow, hawk, songbird]
    deer: [buck, doe, fawn, elk]
    rabbit: [bunny, hare]
    horse: [pony, foal, mare, stallion]
    package: [box, parcel, envelope]
    garbage_truck: ["trash truck", "rubbish truck"]
    license_plate: [plate, plates]
  suggestion_groups:               # used to rank suggested labels on
    animal:                        # mismatched events: when the description
      - cat                        # names other tracked labels, the analyzer
      - dog                        # suggests them (pre-selected in the UI).
      - deer                       # Candidates from the same group as the
      - fox                        # detector's label sort first, since
      - rabbit                     # confusions happen within categories
      - raccoon                    # (dog<->cat, cat<->raccoon). Scene context
      - bird                       # ("Ford SUV") sorts later.
      - possum
      - skunk
      - squirrel
      - rodent
      - horse
    vehicle: [car, garbage_truck, boat, usps, ups, fedex, amazon, dhl, gls]
    person: [person]
    object: [package, license_plate]
  safe_labels:                     # common labels; a description that does not
    - person                       # confirm them is still flagged as mismatch
    - car                          # (so persistent false positives surface),
    - truck                        # just at lower priority than other
    - bicycle                      # mismatches
  noise:
    count: 10                      # >N events of same camera+label...
    window_hours: 1                # ...within this window flags "noise"

review:
  auto_keep_per_label: 500         # default snapshot cap per label (see the
                                   # per-label "Snapshot Limit" on the Controls
                                   # page). Human corrections and false positives
                                   # are ground truth and kept forever. 0
                                   # disables pruning entirely (keep everything).
                                   # Pending queue items are never pruned; ignored
                                   # events expire after keep_ignored_hours below.
  queue_limit: 200                 # max rows shown / exported per request
  preview_scale: 1.33              # review preview context vs the object's larger
                                   # side (1.35 = the training crop)
  preview_imgsz: 640               # review preview render size (px)
  keep_ignored_hours: 24           # delete ignored events (+snapshots) this many
                                   # hours after they were ignored (0 = keep)
  page_size: 100                   # review grid items per page
  genai_help:                      # "GenAI Help" button in the review queue
    enabled: true                  # auto-disabled when api_key is empty
    api_key: "${OPENROUTER_API_KEY:-}"   # or a literal key
    base_url: "https://openrouter.ai/api/v1"
    timeout: 60
    max_image_px: 1024
    default_model: "~google/gemini-flash-latest"
    models:
      - { name: "Gemini Flash (latest)", id: "~google/gemini-flash-latest" }
      - { name: "Qwen Flash (latest)",   id: "qwen/qwen3.8-flash" }

labels:
  track:                           # candidate labels in the review UI AND the
    - car                          # class order of exported datasets and
    - cat                          # labels.txt. Order matters — never reorder
    - dog                          # once you have trained a model!
    - person
    - license_plate
    - usps
    - ups
    - fedex
    - amazon
    - dhl
    - gls
    - deer
    - horse
    - fox
    - rabbit
    - raccoon
    - bird
    - possum
    - skunk
    - squirrel
    - rodent
    - package
    - garbage_truck
    - boat

training:
  trigger_threshold: 50            # corrections goal shown in the UI
  dataset_dir: "/data/dataset"     # exported YOLO dataset
  model_output_dir: "/data/models" # intermediate training artifacts
  publish_dir: "/publish"          # host models/ dir (mounted into Frigate)
  keep_versions: 20                # how many published models to retain
  model_type: "yolo11n"            # default ultralytics base model. The header
                                   # Train button also has a variant picker
                                   # (yolo11n/s/m/l/x tested; yolo12n/s/m/l/x
                                   # offered but flagged untested in the UI);
                                   # a picked variant overrides this for that run
  epochs: 100
  imgsz: 640
  device: "xpu"                    # ultralytics device: xpu (Intel Arc, default),
                                   # cuda (NVIDIA), rocm (AMD), or cpu.
                                   # Training falls back to CPU automatically
                                   # on failure. Set "cpu" to force CPU.
  val_split: 0.1                   # fraction of corrections used for val
  region_crops: true               # export square region crops around objects
                                   # instead of full frames. Frigate 0.15+
                                   # feeds the detector zoomed square regions
                                   # (side = 2x the object, min 320px, scaled
                                   # to imgsz) - training on the same crops is
                                   # required or the model scores ~0 at
                                   # inference. Set false only to reproduce the
                                   # old full-frame exports.
  auto_update_frigate_config: false # trainer patches frigate model.path via API

notifications:
  enabled: false
  discord_webhook: ""              # pinged when corrections >= trigger_threshold

server:
  listen_host: "0.0.0.0"
  listen_port: 8501
  base_path: "/errata"             # must match the nginx location prefix

database:
  path: "/data/errata.db"

logging:
  level: "INFO"
```

### Analyzer flow details

For every unprocessed event (newest first):

1. `confidence < min_confidence` → status `ignored` (out of scope).
2. Degenerate detector box (fails `is_plausible_box`: a full-width sliver
   anchored on the frame edge) → status `ignored` when
   `auto_skip_implausible_boxes` is on (the default), so it never reaches the
   queue or the dataset. With the option off it is flagged as `noise` instead.
   `python -m errata.trainer --check-boxes` reports these, and
   `--skip-degenerate-boxes` retroactively marks any already queued as ignored.
2. No snapshot on disk (never downloaded, or removed by the nightly retention
   prune once it aged past `review.auto_keep_per_label`) → the event and any of
   its corrections are **deleted from the database**. An image-less event can't
   be reviewed or trained on, so it is removed rather than queued. This runs at
   the start of every analysis cycle and on each review-queue page load;
   `python -m errata.trainer --purge-missing-snapshots` forces it on demand.
3. Vocabulary check (needs a GenAI description). A match is the label
   itself, any configured `synonyms` entry, their plural/possessive forms, or
   a built-in irregular plural:
   - no match → `mismatch` (priority 3, or 4 if confidence >= 0.9 —
     confidently wrong). Labels in `safe_labels` are flagged too, but at a
     lower priority (2), so a persistent false positive like a patio umbrella
     scored as `person` still surfaces for review. The analyzer also scans the
     description for **other** tracked labels (incl. synonyms, ranked by
     `suggestion_groups`) and stores them as `suggested_label` — the review UI
     pre-selects the top suggestion so a correction is one click.
   - match → the VLM independently confirms the detection; if
     `confirm_on_coherent_description` is on, low-confidence
     flagging is skipped (no review needed for an agreed label).
4. `oversized` — box area exceeds `oversized_box.max_area` (frame fraction).
   Geometrically legal but usually a merged detection (two adjacent cars under
   one box, common in IR night frames). Priority 2, unless suppressed by the
   confirmation above. Use `python -m errata.trainer --reanalyze-confirmed` to
   apply this rule retroactively to already auto-confirmed events.
5. `low_confidence` — `confidence < low_confidence_threshold`, unless
   suppressed by the confirmation above. Priority 1.
6. `noise` — same camera + label appears `count` times inside
   `window_hours`. Priority 2.
7. Otherwise → status `confirmed` (nothing to review).

The review queue is ordered by priority (high first), then newest.

---

## 3. Environment variables (`.env`)

| Variable | Purpose |
|---|---|
| `ERRATA_FRIGATE_USER` | Frigate UI username, if you enabled Frigate 0.16+ auth |
| `ERRATA_FRIGATE_PASSWORD` | Frigate UI password |
| `ERRATA_CONFIG` | Override config file path inside the container |

`.env` is referenced by `env_file:` in compose, passed into the container,
and interpolated into `config.yaml`. It is excluded from the Docker build
context and from git. Start from `.env.example`.

Frigate auth note: Frigate 0.16+ enables authentication once an admin user
exists (created via the setup wizard). Direct API calls then require a
Bearer token; Errata performs `POST /api/login` transparently and re-auths
on 401. If your Frigate has no auth configured, leave both variables empty.

---

## 4. Deploy on a fresh host (packaging guide)

The whole tool is the `object-model` folder. To share it, copy everything
except `data/`, `models/`, and `.env` (or `git archive` — those are
gitignored).

### 4.1 Prerequisites

- Docker + docker compose plugin
- Frigate 0.16+ running (0.18+ recommended for GenAI descriptions in events)
- A reverse proxy (nginx) in the same compose network with TLS termination

### 4.2 Steps

1. **Copy the project** to e.g. `/opt/errata` (the folder containing
   `Dockerfile`, `errata/`, `config.yaml`, ...).

2. **Create `.env`** from `.env.example`; fill in Frigate credentials if
   needed.

3. **Edit `config.yaml`**:
   - `frigate.api_url` — reachable from the compose network
     (`http://frigate:5000` if the Frigate service is named `frigate`).
   - `labels.track` — the object classes you care about (order = dataset
     class order = `labels.txt` order).
   - `server.base_path` — URL prefix; keep `/errata` or choose your own
     (must match the nginx location).

4. **Add the compose service** — copy the `errata:` service from
   [`deploy/docker-compose.snippet.yml`](../deploy/docker-compose.snippet.yml)
   into your Frigate compose file and fix the `/path/to/object-model` paths.
   Paths are absolute to be robust against symlinked compose files.
   While you're at it, also add this volume to the **frigate** service (for
   later custom-model deployment — harmless until you switch models):

   ```yaml
   frigate:
     volumes:
       - /path/to/object-model/models:/config/models:ro
   ```

5. **Add the nginx location** — paste
   [`nginx/errata-location.conf`](../nginx/errata-location.conf) inside your
   HTTPS `server` block, next to the Frigate `location /` block. It requires
   the `map $http_upgrade $connection_upgrade` stanza (Errata's own
   nginx.conf has it; see that file for reference).

6. **Build and start**:

   ```bash
   docker compose up -d --build errata
   docker compose exec frigate-nginx nginx -t
   docker compose exec frigate-nginx nginx -s reload
   ```

   If you edited the frigate service (models mount), recreate it too:
   `docker compose up -d frigate` (brief recording gap).

7. **Verify**: open `https://your-domain/errata/` — you should see an empty
   or filling review queue. Check `GET /errata/api/health` for the last
   scheduler run stats.

### 4.3 Optional systemd integration

If your host manages compose projects via a template unit (like
`docker-compose@<name>.service` pointing at a project directory), Errata is
picked up automatically because it lives in the same compose file. Nothing
else to do. Otherwise create a wrapper unit or just use
`docker compose up -d` in your own tooling.

---

## 5. Review UI

| Route | Purpose |
|---|---|
| `GET /errata/` | Review queue. Grid of wider-context crops with multi-select (checkbox, shift-click range, drag marquee). The `View` dropdown selects the status (`pending`, `corrected`, `false_positive`, `confirmed`, `ignored`, `all`); camera/label/reason filters narrow the list; 100 items per page with prev/next. `Set status` + Apply acts on **all matching** events |
| `POST /errata/review/{id}` | Per-item actions: `confirm` (+ `label`), `false_positive`, `ignore` (status `ignored`, never trained), `reset` (undo → back to `pending`). Optional `box` (JSON `[x,y,w,h]`) overrides the stored box. `return_to` preserves filters |
| `POST /errata/bulk/selected` | Apply an action to **selected** events: repeated `event_ids` + `action` in `confirm` / `relabel` (+ `label`) / `false_positive` / `ignore` / `reset` |
| `POST /errata/brand/{id}`, `POST /errata/brands/bulk/selected` | Confirm/reject/undo a brand item, individually or for selected `brand_ids` |
| `POST /errata/bulk` | Bulk-apply a target status (`confirmed`, `false_positive`, `ignored`) to every pending event matching the `camera` / `label` / `reason` filters |
| `GET /errata/summary` | Stats: statuses, corrections by camera/label, retrain progress, model versions, and the on-disk dataset the trainer consumes (human vs pseudo labels, backgrounds, per-class counts) |
| `GET /errata/synonyms` | View/edit label synonyms (saved as a DB override, applied on next analyzer run) |
| `POST /errata/synonyms` | Save the synonym form (comma-separated per label) |
| `POST /errata/synonyms/reset` | Discard overrides, fall back to the config.yaml baseline |
| `GET /errata/api/queue.json` | Export queue as JSON |
| `GET /errata/api/snapshots/{id}.jpg` | Stored snapshot with the detection box burned in (click a queue image to enlarge) |
| `GET /errata/api/snapshots/{id}/clean.jpg` | The stored snapshot without the burned-in box, used by the **Edit box** editor |
| `GET /errata/api/health` | Liveness + last scheduler stats |
| `POST /errata/api/run-now` | Trigger an immediate harvest+analyze cycle (also a "Fetch latest" button in the header) |
| `POST /errata/api/train` | Start a training run in the container background (also a "Train" button in the header). Re-exports the dataset first, then trains with the device from `training.device` (xpu with CPU fallback). 409 if already running, 503 if the image lacks ultralytics |
| `GET /errata/api/models` | List published models, the model Frigate is currently serving, and config backups |
| `POST /errata/api/models/activate` | Body `{"model": "errata_<stamp>.onnx"}`. Backs up the Frigate config, patches the model path via the Frigate API, saves with `save_option=restart`, waits for Frigate to return |
| `POST /errata/api/models/backups/<id>/revert` | Restore a previous Frigate config backup (a fresh backup of the current config is taken first), then restart Frigate |
| `GET /errata/api/dataset/stats` | Summarize the on-disk YOLO dataset (human/pseudo images and objects, backgrounds, per-class train/val counts); the summary page renders this and warns when unexported corrections are not yet in the dataset |
| `GET /errata/controls` / `POST /errata/controls` | Per-label controls. UI names: Collection (`all`=Human + Machine Review, `review_only`=Human Review Only, `off`=Disabled (Ignore)), GenAI Description Search (`search`), Include in Training (`include_training`), Auto-approve (`auto_confirm`), Include Auto Approved in Training (`pseudo_labels`), Snapshot Limit (`keep`, blank = global `review.auto_keep_per_label`, 0 = keep forever). Each column header fills every row client-side before saving |
| `POST /errata/brand/{id}` / `POST /errata/brands/bulk` | Confirm/reject a brand review item (metadata only) |
| `GET /errata/api/snapshots/{id}/crop.jpg` | The actual training crop for an event (matches the trainer's region crop) |
| `GET /errata/api/snapshots/{id}/preview.jpg` | Wider-context crop used by the review grid (scale `review.preview_scale`, rendered at `review.preview_imgsz`) |
| `POST /errata/api/genai_help/{id}` | Body `{"model": "<id>", "kind": "object"\|"brand", "expected": "<brand>"}`. Sends the clean snapshot to OpenRouter for a second opinion; returns `{label, description, confidence, box}`. 503 when disabled (no API key). Enabled automatically when `review.genai_help.api_key` is set |
| `GET /errata/base-models` / `POST /errata/base-models/import-plus` / `POST /errata/base-models/upload` / `POST /errata/base-models/{name}/delete` | Base-model registry, Frigate+ import (one-time key), uploads |
| `POST /errata/api/reset` | Body `{"confirm": "RESET", "return_model": "<optional published model>"}`. Wipes all learned/review data + files, disables all label controls (base models kept), optionally re-activates a kept model, then saves a final Frigate config backup |
| `GET /errata/api/train/status` | Training job state (running / finished / error + detail); also shown on the Summary page |
| `GET /errata/api/train/mac-kit.zip` | Download the Mac training kit: dataset, source files, `train-mac.sh`, and a README with Homebrew setup; refreshes the dataset export first. Also linked as "Mac kit" in the header |
| `GET /healthz` | Unprefixed health endpoint (used by the Docker healthcheck) |

### Editing the detection box

Every card has an **Edit box** button. It opens the clean snapshot with the
detection box overlaid; drag the box to move it, drag a corner/edge handle to
resize, or drag on empty image space to draw a new box. **Save correction**
stores the chosen label paired with the edited box, and **Save as false
positive** stores the drawn region as a background crop. The saved box is what
the exported YOLO label uses, so a correction can fix a merged/oversized box or
add a tightly-drawn object the detector missed. Boxes are clamped to the frame;
a box smaller than 1% of a side is rejected.

### Reviewing past decisions

The `View` filter also exposes already-reviewed events, so decisions can be
revised after the fact. Corrected cards show the label you chose
(`→ label`); selecting a different label and pressing **Change** replaces the
stored correction (the old `corrections` row for that event is removed and a
new unexported one is inserted, so the next training run uses your latest
decision). **Undo review** drops the correction and returns the event to the
queue (`flagged` if it still has a flag reason, otherwise `new`). Re-confirming
an auto-`confirmed` event with the same label promotes it to a human
correction, which is stronger training signal than a pseudo label.

The UI has no authentication. It is only reachable through your nginx, which
may already sit behind your SSO/VPN. To add HTTP basic auth, extend the
location block:

```nginx
location /errata/ {
    auth_basic "Errata";
    auth_basic_user_file /etc/nginx/errata.htpasswd;
    ...
}
```

---

## 6. Database schema (`data/errata.db`)

SQLite, WAL mode.

- **events** — `id` (Frigate event id, PK), `camera`, `label`, `confidence`,
  `description` (GenAI), `attributes` (JSON), `start_time`, `end_time`,
  `box` (JSON `[x,y,w,h]` at detect resolution), `snapshot_path`,
  `status` (`new → flagged → corrected|confirmed|false_positive|ignored`),
  `flag_reason` (`mismatch|low_confidence|noise|oversized`), `priority`,
  `created_at`, `reviewed_at`.
- **corrections** — one row per user correction: `event_id`, `image_path`,
  `correct_label` (target class or `false_positive`), `original_label`,
  `confidence`, `camera`, `box`, `collected_at`, `exported`.
- **model_versions** — `model_path`, `created_at`, `training_count`, `metrics`.
- **event_brands** — brand review items: `event_id`, `brand`, `source`, `model`,
  `score`, `status` (`pending|confirmed|rejected`), `reviewed_at`, `created_at`.
- **label_settings** — per-label controls: `label`, `collect_mode`, `search`,
  `include_training`, `auto_confirm`, `pseudo_labels`, `updated_at`.
- **base_models** — `name`, `source` (`yolo|frigate_plus|upload`), `format`
  (`pt|onnx`), `path`, `imgsz`, `labels`, `trainable`, `meta`, `created_at`.
- **config** — key/value (progress markers, `synonyms_override`,
  `controls_default_disabled`).

Indexes: `events(status)`, `events(camera,label)`, `events(start_time)`,
`corrections(exported)`, `corrections(correct_label)`.

---

## 7. Training & model deployment

### 7.1 Export the dataset

```bash
docker exec errata python -m errata.trainer --export-only

Re-export the dataset from scratch (all corrections, current export format):

docker exec errata python -m errata.trainer --rebuild
```

- Writes YOLO format to `data/dataset/`:
  `images/{train,val}/`, `labels/{train,val}/`, `data.yaml`, `labels.txt`.
- One image per correction; the box comes from the detector output stored on
  the event (approximate annotation — good enough for fine-tuning).
- `false_positive` corrections are exported as **background images**: the
  image is copied and its label file is left empty, which is the standard YOLO
  mechanism for suppressing false positives ("nothing here"). Only mark FPs
  when the frame truly contains no object of interest — a background image
  that secretly contains, say, a raccoon teaches the model to ignore
  raccoons. FPs without a stored image are skipped.
- Records without image/box are marked processed and skipped.
- With fewer than 4 samples, train and val contain the same images.

### 7.2 Train

Requires `ultralytics` (`requirements-training.txt`), which is not installed
in the slim runtime image. Either build a training image on top of it:

```bash
docker build -t errata-train:local -f - . <<'EOF'
FROM errata:local
USER root
RUN pip install --no-cache-dir -r requirements-training.txt
USER errata
EOF
docker run --rm -v ./data:/data -v ./models:/publish \
  -v ./config.yaml:/config/errata.yaml:ro errata-train:local \
  python -m errata.trainer --train
```

or run directly on any machine with Python 3.10+ (needs the same `data/`
mounts):

```bash
pip install -r requirements.txt -r requirements-training.txt
python -m errata.trainer --train            # trains and publishes ONNX
```

`--train` exports the dataset first, then runs `yolo` training
(`model_type`, `epochs`, `imgsz` from config), exports ONNX, and publishes:

- `models/published/errata_<timestamp>.onnx`
- `models/published/labels.txt` (class order — must match `data.yaml`)
- `models/published/metrics.json`

Artifacts live in the `published/` subfolder so the top-level `models/`
directory stays free for other artifacts you may want to deploy into
Frigate (`/config/models`).
- a row in `model_versions`; older versions pruned to `keep_versions`.

### 7.3 Point Frigate at the model

In your Frigate `config.yml`:

```yaml
detectors:
  gpu0:
    type: openvino
    device: GPU

model:
  path: /config/models/published/errata_20261003-120000.onnx
  width: 640
  height: 640
  input_dtype: float
  labelmap_path: /config/models/published/labels.txt
  model_type: yolo-generic
```

> `input_dtype` must be `float` (`float32` fails schema validation), and
> `model_type: yolo-generic` is required so Frigate uses the Ultralytics YOLO
> parser rather than the default SSD one.

With `training.auto_update_frigate_config: true`, the trainer rewrites
`model.path` itself (reads config via `GET /api/config/raw`, patches only the
`path:` line under the top-level `model:` section, writes via
`POST /api/config/save`). Frigate reloads the changed config without a
container restart.

Verify in Frigate logs that the detector loads the model; watch the first
detections for sanity. Roll back: restore the previous `model.path`
(kept `keep_versions` deep in `models/`) and reload again.

### 7.4 Retrain loop

The summary page shows corrections since the last published model against
`training.trigger_threshold`. When the threshold is reached (and
notifications are enabled) a Discord ping goes out. Then run `--train`
manually (or cron it).

---

## 8. Operations

| Task | Command |
|---|---|
| Immediate harvest+analyze | **"Fetch latest" button** in the UI header, or `curl -X POST https://.../errata/api/run-now` |
| Snapshot cleanup | Automatic: the scheduler cycle (every `harvest.interval_minutes`) runs the prune at most once per 24h. Only auto-confirmed / auto-ignored events with no human correction are capped per label — at that label's **Snapshot Limit** (Controls page, default `review.auto_keep_per_label`; 0 = keep forever). Human corrections and false positives are ground truth and kept forever; pending events are never pruned; ignored events are deleted with their snapshots once older than `review.keep_ignored_hours` (default 24, every cycle), and events whose snapshot file is missing are deleted from the database. Force the marker to re-run: `docker exec errata python -c "from errata.db import Database; Database('/data/errata.db').kv_delete('last_prune_at')"` |
| Trigger model rebuild | Train button (GPU/CPU per config), Mac kit, or `./deploy/train.sh`. Use `--export-only` to just rebuild the dataset and `--rebuild` to re-export every correction from scratch (wipes the dataset). The Discord notification at the correction threshold is a reminder, not a trigger |
| Backfill further into history | set `harvest.lookback_hours` (e.g. 168 for a week), clear the watermark, trigger: `docker exec errata python -c "from errata.db import Database; Database('/data/errata.db').kv_delete('last_processed_at')"` then Fetch latest. Snapshots for old events may be expired by Frigate retention (`record` / snapshot retention) — those events are stored without images |
| Reset processed-watermark (re-harvest 24h) | `docker exec errata python -c "from errata.db import Database; Database('/data/errata.db').kv_delete('last_processed_at')"` (or edit the `config` table) |
| Full reset | stop errata, `rm -rf data/` , start again |
| Logs | `docker logs -f errata` |
| DB inspect | `sqlite3 data/errata.db 'select status, count(*) from events group by status;'` |
| Degenerate-box artifact | Full-width, few-pixels-tall boxes at the frame bottom (wide cameras). Check: `docker exec errata python -m errata.trainer --check-boxes` (read-only). Fix: `docker exec errata python -m errata.trainer --skip-degenerate-boxes`, then retrain and activate — human corrections are never touched |
| Merged/oversized boxes (e.g. two cars under one box at night) | Flagged as `oversized` in the review queue; they are never auto-confirmed and never become training pseudo-labels. Confirm correct events manually, or lower `analysis.oversized_box.max_area`. For events already auto-confirmed before the rule existed: `docker exec errata python -m errata.trainer --reanalyze-confirmed`, then retrain |

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| Queue always empty | Check `/errata/api/health` → `last_scheduler_stats`; check `docker logs errata` for Frigate API errors (wrong `api_url`, auth needed) |
| 401 in logs | Set `ERRATA_FRIGATE_USER` / `ERRATA_FRIGATE_PASSWORD` in `.env` and restart |
| 502 from nginx | errata container down or `server.listen_port` mismatch; the resolver needs both containers on the same compose network |
| UI assets / links broken behind proxy | `server.base_path` must equal the nginx location prefix (default `/errata`) |
| Snapshots missing for old events | Auto-confirmed/auto-ignored snapshots with no human correction are pruned nightly to the most recent `review.auto_keep_per_label` per label (0 = never prune); human corrections and false-positive snapshots are kept forever; ignored events expire after `review.keep_ignored_hours`. Events whose snapshot file is missing are deleted from the database |
| Model rejected by Frigate | Confirm `labelmap_path` exists and class count matches; check detector logs; ONNX export imgsz must match `model.width/height` |
| Harvest rescans same window | `last_processed_at` only advances on successful runs; lookback window covers stragglers |
