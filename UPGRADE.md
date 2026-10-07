# Upgrading an existing Errata install

This upgrades Errata in place. Your review database (`data/`) and published
models (`models/`) are never part of the sharable bundle and are left
untouched. Only application code, templates, and the default `config.yaml`
change.

> `data/` and `models/` contain private camera imagery and your trained ONNX
> model. Never include them in a zip you share.

## What's new in this version

- **Degenerate-box guard** — detector boxes that are geometrically impossible
  (a min side below 0.005 of the frame, or an aspect ratio above 20:1 — e.g.
  the full-width slivers at the frame bottom seen on wide cameras) are never
  auto-confirmed and never exported as pseudo-labels. New maintenance CLI:
  `--check-boxes` (read-only report) and `--skip-degenerate-boxes` (sets them
  to `skipped`; human corrections are never touched).
- **Training matches Frigate's inference domain** — Frigate feeds the detector
  square region crops (~1.35x the object's larger side, black-padded when the
  region extends past the frame). The trainer's region crops now replicate
  exactly that; previously they were clamped inside the frame and zoomed 2x,
  so the model never saw the padded-region domain — the source of the
  full-width sliver artifacts — and full-frame training (`region_crops:
  false`) produced whole-frame / giant-box artifacts instead. Keep
  `region_crops: true`; do not train on full frames.
- **Stale pseudo-labels purged** — every dataset export removes old
  `pseudo_*` files first, so the dataset always reflects the current
  auto-confirmed pool (no `--rebuild` needed after changing filters).
- **Review past decisions** — the review queue now has a `View` filter that
  can show already-reviewed events (`Corrected`, `False positive`,
  `Auto-confirmed`, `Ignored`, `Skipped`, or `All`), so a decision can be
  revised after the fact. Corrected cards show the label you chose
  (`→ label`); pick a different one and press **Change** to replace the
  stored correction, or press **Undo review** to drop the correction and
  return the event to the queue (`flagged` if it still has a flag reason,
  otherwise `new`).
  - Re-reviewing is idempotent: the old `corrections` row for that event is
    removed and a fresh unexported one is inserted, so a training run always
    uses your latest decision and never sees duplicate labels.
  - Confirming an auto-`confirmed` event with the same label promotes it to a
    human correction (stronger signal than a pseudo label).
  - The `POST /review/{id}` endpoint gains an `reset` action; an optional
    `return_to` querystring keeps the current filters/View after an action.
- No new config keys and no schema change — the feature uses the existing
  `events` / `corrections` tables. Rebuild the image to pick it up.

### Previous release

- **GPU (XPU) training fixed** — training runs on an Intel Arc GPU when
  present (`training.device: xpu`), with automatic CPU fallback. The image now
  pins a *matched* Intel stack (torch 2.14.1+xpu + Level Zero 1.34 +
  compute-runtime 26.35, downloaded at build time); the previous crash came
  from a mismatched 2025 stack on a 2026 kernel. If your host has no Intel
  GPU, set `training.device: cpu`.
- **Model deployment UI** — the summary page lists every published model and
  which one Frigate is serving. Activating one backs up the Frigate config to
  `data/frigate-config-backups/`, switches the model path via the Frigate API
  (`save_option=restart` — the save triggers the restart), and waits for
  Frigate to come back. Any backup can be reverted the same way. Frigate is
  down for ~30-60s during activation.
- **Training data visibility** — the summary page now shows the on-disk YOLO
  dataset the trainer actually consumes (human vs pseudo labels, background
  frames, and per-class train/val counts), and warns when corrections have
  been collected since the last export and are therefore *not* in the dataset.
  The Train button now re-exports the dataset first, and every published
  model records the dataset snapshot it was trained on ("Trained on" column).
- **New API** — `GET /api/models`, `POST /api/models/activate {model}`,
  `POST /api/models/backups/<id>/revert` (guarded by a deploy lock), and
  `GET /api/dataset/stats`.
- **New config keys** — `frigate.backup_dir`
  (default `/data/frigate-config-backups`) and
  `frigate.published_model_prefix` (default `/config/models/published`).
- **Compose snippet** — now includes `init: true`, GPU passthrough
  (`devices`, `device_cgroup_rules`, `group_add: 109`), `shm_size: 8g`, and
  `YOLO_CONFIG_DIR`. Re-apply it if you are upgrading an existing install.
- The SQLite schema adds a `frigate_backups` table; it migrates itself on
  startup, no manual step.

### Earlier

- **Analyzer / noise rule** — a detection whose GenAI description confirms the
  YOLO label (vocabulary coherence) is no longer flagged as `noise`. Coherent
  bursts (e.g. a cat pacing past the garage camera) are auto-confirmed instead
  of landing in the review queue.
- **Review queue toolbar** — Camera / Label / Reason now filter the displayed
  queue again, and `Set status` + `Apply to matching` is a bulk action that
  applies a status (`skipped`, `confirmed`, `false_positive`, `ignored`) to
  every pending event matching the current filters. Leave the filters on
  `All` and choose `skipped` to set aside the entire remaining queue in one go.
- Per-card actions (`confirm`, `False positive`, `Skip`) are unchanged.

## Before you upgrade

Back up the stateful bits (cheap insurance, and they are gitignored anyway):

```bash
cd /path/to/object-model
stamp=$(date +%Y%m%d-%H%M%S)
cp -a data "data.bak-$stamp"
cp -a models "models.bak-$stamp"
cp config.yaml "config.yaml.bak-$stamp"
[ -f .env ] && cp .env ".env.bak-$stamp"
```

## Upgrade steps

1. **Unpack the new bundle** somewhere temporary, e.g. `/tmp/errata-app`. It
   expands to a folder named `errata-app/`.

2. **Stop the app** (Frigate keeps running; no camera downtime):

   ```bash
   cd /path/to/your/compose/project
   docker compose stop errata
   ```

3. **Overlay the new files** onto your install. The bundle intentionally omits
   `data/`, `models/`, and `.env`, so copying over the top is safe:

   ```bash
   rsync -a --exclude data --exclude models --exclude .env \
     /tmp/errata-app/ /path/to/object-model/
   ```

4. **Reconcile `config.yaml`** — it is included in the bundle and will
   overwrite yours. Diff it against your backup and re-apply any local values
   (`frigate.api_url`, `labels.track` / class order, `server.base_path`,
   thresholds, synonyms, etc.):

   ```bash
   diff -u config.yaml.bak-* /path/to/object-model/config.yaml
   ```

   `.env` is never overwritten; leave your Frigate credentials as they are.

5. **Rebuild and start**:

   ```bash
   docker compose build errata
   docker compose up -d errata
   ```

6. **Verify**:

   ```bash
   docker compose exec errata python -c \
     "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8501/healthz').status)"
   ```

   Then open `https://<your-host>/errata/` and confirm the new **View**
   dropdown (top-left of the filter bar) lists *Pending review / Corrected /
   False positive / Auto-confirmed / Ignored / Skipped / All events*, and that
   a corrected card shows its chosen label with an **Undo review** button.
   The SQLite schema migrates itself on startup — there is no manual migration
   step.

7. **Update nginx / compose only if they changed** in this release. See
   `docs/CONFIGURATION.md` §4 for the current snippets.

## Fixing the degenerate bounding-box artifacts (wide cameras)

Symptom, two related families on wide-aspect cameras (e.g. 1816x816):

- a box spanning the entire bottom of the frame, only a few pixels tall;
- a box covering the whole frame or a giant slab of it (top third, top half).

Cause: Frigate feeds the model square region crops. When a detection is
full-width, that region balloons to ~2300x2300 px, ~84% of it black padding.
Training that never saw that domain hallucinates on it: tight in-frame crops
(`region_crops` with clamping, the old behavior) fire on the content/padding
edge and emit slivers; full-frame training (`region_crops: false`) emits
whole-frame boxes when fed zoomed crops. Frigate's GenAI description then
mentions a vehicle *somewhere* in the scene, the analyzer saw the label's
synonym in the description and auto-confirmed the artifact, and the
pseudo-label exporter would have fed it back into the next training round — a
feedback loop that piles up `confirmed` artifacts over time even though the
active model was never trained on them directly.

This bundle stops the loop (box guard + region crops that replicate Frigate's
region extraction, padding included), but an existing install should also
clear the artifacts already in the database:

1. **Check** (read-only):

   ```bash
   docker exec errata python -m errata.trainer --check-boxes
   ```

2. **Fix** — mark them skipped (excluded from training; snapshots kept; visible
   under *View → Skipped*; human corrections are never touched):

   ```bash
   docker exec errata python -m errata.trainer --skip-degenerate-boxes
   ```

3. **Retrain** (Train button, Mac kit, or `deploy/train.sh`) and **activate**
   the new model on the Summary page. The export purges stale pseudo-label
   files automatically — no `--rebuild` needed.

4. **Verify**: re-run `--check-boxes` (everything should be `skipped`), then
   watch the affected cameras for a day — the slivers should stop. If a model
   still emits them, check that Frigate is actually serving the new ONNX
   (`Summary → Frigate is serving`).

## Rolling back

The upgrade is code-only, so rollback is:

```bash
# restore the previous version of the app files, then
docker compose build errata && docker compose up -d errata
```

If you need the previous database too, stop errata, move the `data.bak-*`
folder back to `data/`, and start errata again.
