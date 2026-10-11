"""End-to-end smoke tests for the object/brand revamp.

Run with: python -m unittest discover -s tests -v
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from errata import basemodels, trainer
from errata import genai_help
from errata.analyzer import Analyzer
from errata.config import load_config, tracked_labels, attribute_labels
from errata.controls import (
    effective_controls,
    effective_for_label,
    reset_all_disabled,
    set_control,
)
from errata.db import Database
from errata.harvester import extract_brands
from errata.vocab import effective_synonyms


def make_config(tmp: str) -> dict:
    cfg = load_config()
    cfg["database"]["path"] = os.path.join(tmp, "errata.db")
    cfg["frigate"]["snapshot_dir"] = os.path.join(tmp, "snap")
    cfg["frigate"]["backup_dir"] = os.path.join(tmp, "backups")
    cfg["training"]["dataset_dir"] = os.path.join(tmp, "dataset")
    cfg["training"]["model_output_dir"] = os.path.join(tmp, "models")
    cfg["training"]["publish_dir"] = os.path.join(tmp, "publish")
    cfg["training"]["base_dir"] = os.path.join(tmp, "base")
    os.makedirs(cfg["frigate"]["snapshot_dir"], exist_ok=True)
    return cfg


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)
        self.db = Database(self.cfg["database"]["path"])

    def add_event(self, eid, label, correct=None, box="[0.1,0.1,0.3,0.3]",
                  description="thing", sub_label=""):
        path = os.path.join(self.cfg["frigate"]["snapshot_dir"], f"{eid}.jpg")
        Image.new("RGB", (320, 240), (40, 40, 40)).save(path)
        self.db.upsert_event({
            "id": eid, "camera": "cam", "label": label, "confidence": 0.9,
            "description": description, "box": box, "snapshot_path": path,
            "sub_label": sub_label, "start_time": 1,
        })
        if correct:
            self.db.insert_correction({
                "event_id": eid, "image_path": path, "correct_label": correct,
                "original_label": label, "confidence": 0.9, "camera": "cam", "box": box,
            })
        return path


class TestTaxonomy(Base):
    def test_track_excludes_brands_and_plates(self):
        track = tracked_labels(self.cfg)
        self.assertNotIn("usps", track)
        self.assertNotIn("fedex", track)
        self.assertNotIn("license_plate", track)
        self.assertIn("car", track)
        self.assertIn("coyote", track)

    def test_attributes_listed(self):
        self.assertEqual(attribute_labels(self.cfg),
                         ["usps", "ups", "fedex", "amazon", "dhl", "gls"])

    def test_lookback_default_zero(self):
        self.assertEqual(self.cfg["harvest"]["lookback_hours"], 0)

    def test_settings_env_expansion(self):
        os.environ["OPENROUTER_API_KEY"] = "unit-test-key"
        try:
            cfg = load_config()
        finally:
            os.environ.pop("OPENROUTER_API_KEY", None)
        self.assertEqual(cfg["review"]["genai_help"]["api_key"], "unit-test-key")

    def test_brand_synonyms_append(self):
        self.db.kv_set("synonyms_override", json.dumps({"coyote": ["brush wolf"]}))
        syn = effective_synonyms(self.cfg, self.db)
        self.assertIn("wolf", syn["coyote"])          # baseline kept
        self.assertIn("brush wolf", syn["coyote"])     # addition appended


class TestHarvesterBrands(unittest.TestCase):
    def test_extract_from_sub_label_and_data(self):
        ev = {"sub_label": "usps", "data": {"delivery": "fedex"}}
        self.assertEqual(extract_brands(ev, ["usps", "ups", "fedex"]), ["fedex", "usps"])

    def test_ignores_unknown(self):
        ev = {"sub_label": "person", "data": {"x": "random"}}
        self.assertEqual(extract_brands(ev, ["usps"]), [])


class TestAnalyzer(Base):
    def test_candidate_search_surfaces_label(self):
        self.add_event("e1", "dog", description="a coyote on the lawn")
        set_control(self.cfg, self.db, "coyote", search=True)
        stats = Analyzer(self.cfg, self.db).run()
        self.assertEqual(stats["candidate"], 1)
        row = self.db.get_event("e1")
        self.assertEqual(row["status"], "flagged")
        self.assertEqual(row["flag_reason"], "candidate")
        self.assertEqual(row["suggested_label"], "coyote")

    def test_machine_review_off_skips_clean_events(self):
        self.add_event("e2", "car", description="a car")
        set_control(self.cfg, self.db, "car", collect_mode="review_only")
        Analyzer(self.cfg, self.db).run()
        # Human Review Only: a clean event is ignored, then dropped (never reviewed).
        self.assertIsNone(self.db.get_event("e2"))

    def test_machine_review_on_confirms_clean(self):
        self.add_event("e3", "car", description="a car")
        Analyzer(self.cfg, self.db).run()
        self.assertEqual(self.db.get_event("e3")["status"], "confirmed")


class TestCollectMode(Base):
    def test_off_routes_to_ignored(self):
        self.add_event("e1", "car", description="a car")
        set_control(self.cfg, self.db, "car", collect_mode="off")
        Analyzer(self.cfg, self.db).run()
        row = self.db.get_event("e1")
        self.assertEqual(row["status"], "ignored")
        self.assertIsNotNone(row["reviewed_at"])

    def test_ignored_purge_query_and_delete(self):
        self.add_event("old", "car", description="a car")
        self.add_event("new", "car", description="a car")
        with self.db.connect() as conn:
            conn.execute("UPDATE events SET status='ignored', reviewed_at=0 WHERE id='old'")
        rows = self.db.ignored_for_purge(24)
        self.assertEqual([r["id"] for r in rows], ["old"])
        self.db.delete_events(["old"])
        self.assertIsNone(self.db.get_event("old"))
        self.assertIsNotNone(self.db.get_event("new"))

    def test_queue_offset_and_total(self):
        for i in range(5):
            self.add_event(f"q{i}", "car", description="a car")
        total = self.db.queue_total(status="all")
        self.assertEqual(total, 5)
        page1 = self.db.queue(status="all", limit=2, offset=0)
        page2 = self.db.queue(status="all", limit=2, offset=2)
        self.assertEqual(len(page1), 2)
        self.assertEqual(len(page2), 2)
        self.assertNotEqual({r["id"] for r in page1}, {r["id"] for r in page2})


class TestExport(Base):
    def test_drop_brands_hold_candidates_export_objects(self):
        self.add_event("e1", "dog", correct="dog")
        self.add_event("e2", "car", correct="usps")     # attribute -> dropped
        self.add_event("e3", "dog", correct="coyote")   # candidate -> held
        set_control(self.cfg, self.db, "coyote",
                    human_verifications="collect", machine_verifications="collect")
        counts = trainer.export_dataset(self.cfg, self.db)
        self.assertGreaterEqual(counts["exported"], 1)
        labels = (Path(self.cfg["training"]["dataset_dir"]) / "labels.txt").read_text().split()
        self.assertIn("dog", labels)
        self.assertNotIn("coyote", labels)
        self.assertNotIn("usps", labels)
        held = [r["correct_label"] for r in self.db.corrections_unexported()]
        self.assertEqual(held, ["coyote"])

    def test_zero_enabled_classes_refuses(self):
        self.add_event("e1", "dog", correct="dog")
        reset_all_disabled(self.cfg, self.db,
                           tracked_labels(self.cfg) + attribute_labels(self.cfg))
        self.assertEqual(trainer.trained_class_map(self.cfg, self.db), {})
        counts = trainer.export_dataset(self.cfg, self.db)
        self.assertEqual(counts["exported"], 0)


class TestControls(Base):
    def test_reset_sets_disabled_and_future_labels_disabled(self):
        reset_all_disabled(self.cfg, self.db,
                           tracked_labels(self.cfg) + attribute_labels(self.cfg))
        cfg = effective_for_label(self.cfg, self.db, "car")
        self.assertEqual(cfg["collect_mode"], "off")
        self.assertEqual(cfg["human_verifications"], "collect")
        self.assertEqual(cfg["machine_verifications"], "collect")
        # a "new" label not in the reset list inherits disabled via the flag
        fresh = effective_controls(self.cfg, self.db, ["brand_new"])["brand_new"]
        self.assertEqual(fresh["collect_mode"], "off")
        self.assertEqual(fresh["human_verifications"], "collect")
        self.assertEqual(fresh["machine_verifications"], "collect")


class TestBaseModels(Base):
    def test_seed_is_idempotent(self):
        basemodels.seed_builtin_models(self.db)
        n = len(self.db.base_models())
        basemodels.seed_builtin_models(self.db)
        self.assertEqual(len(self.db.base_models()), n)
        self.assertGreater(n, 0)

    def test_frigate_plus_labelmap_ordering(self):
        info = {"labelMap": {"2": "deer", "0": "person", "1": "dhl"}}
        self.assertEqual(basemodels._plus_label_list(info), ["person", "dhl", "deer"])

    def test_model_layout_prefers_declared_meta(self):
        row = {"meta": json.dumps({"inputShape": "nchw"})}
        self.assertEqual(basemodels._model_layout(row, "/nonexistent.onnx"), "nchw")

    def test_patch_model_config_sets_extra_keys(self):
        text = "model:\n  path: /old.onnx\n  input_tensor: nhwc\n  width: 320\n"
        new, changed = trainer.patch_model_config(
            text, "/new.onnx", width=640, height=640, input_tensor="nchw",
            extra={"labelmap_path": "/labels.txt", "input_dtype": "float", "model_type": "yolo-generic"},
        )
        self.assertTrue(changed)
        self.assertIn("path: /new.onnx", new)
        self.assertIn("input_tensor: nchw", new)
        self.assertIn("width: 640", new)
        self.assertIn("labelmap_path: /labels.txt", new)
        self.assertIn("input_dtype: float", new)
        self.assertIn("model_type: yolo-generic", new)

    def test_labels_from_names_orders_by_index(self):
        self.assertEqual(
            basemodels._labels_from_names({2: "deer", 0: "person", 1: "car"}),
            ["person", "car", "deer"],
        )
        self.assertEqual(basemodels._labels_from_names(None), [])

    def test_export_activate_rejects_non_variant(self):
        from errata import scheduler as sched_mod
        from errata import webapp as webapp_mod
        from starlette.testclient import TestClient

        sched_mod.Scheduler.start = lambda self: None
        sched_mod.Scheduler.stop = lambda self: None
        app = webapp_mod.create_app(self.cfg)
        with TestClient(app) as c:
            r = c.post("/errata/base-models/not-a-model/export-activate", data={"imgsz": 320},
                       follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn("not%20a%20packaged%20YOLO%20variant", r.headers["location"])

    def test_base_models_not_pruned_by_keep_versions(self):
        self.db.base_model_insert("custom", "upload", "pt", "/x/custom.pt")
        # keep_versions only touches published errata_*.onnx; base table intact
        self.assertIsNotNone(self.db.base_model_get("custom"))


class TestSnapshotLimit(Base):
    def _confirm(self, eid, label, start):
        self.add_event(eid, label, description="a car")
        self.db.set_status(eid, "confirmed")
        with self.db.connect() as conn:
            conn.execute("UPDATE events SET start_time=? WHERE id=?", (start, eid))

    def test_per_label_cap(self):
        for i in range(5):
            self._confirm(f"c{i}", "car", i)
        rows = self.db.prune_candidates({"car": 2}, 150)
        self.assertEqual(sorted(r["id"] for r in rows), ["c0", "c1", "c2"])

    def test_zero_keeps_forever(self):
        for i in range(5):
            self._confirm(f"c{i}", "car", i)
        self.assertEqual(self.db.prune_candidates({"car": 0}, 150), [])

    def test_default_cap(self):
        for i in range(4):
            self._confirm(f"c{i}", "car", i)
        rows = self.db.prune_candidates({}, 2)
        self.assertEqual(sorted(r["id"] for r in rows), ["c0", "c1"])

    def test_controls_carry_keep(self):
        set_control(self.cfg, self.db, "car", keep=42)
        self.assertEqual(effective_for_label(self.cfg, self.db, "car")["keep"], 42)
        set_control(self.cfg, self.db, "car", keep=None)
        self.assertIsNone(effective_for_label(self.cfg, self.db, "car")["keep"])


class TestGenAIHelp(Base):
    def test_disabled_without_key(self):
        self.cfg["review"]["genai_help"]["api_key"] = ""
        self.assertFalse(genai_help.help_enabled(self.cfg))

    def test_enabled_with_key(self):
        self.cfg["review"]["genai_help"]["api_key"] = "test-key"
        self.assertTrue(genai_help.help_enabled(self.cfg))

    def test_analyze_parses_structured_response(self):
        self.cfg["review"]["genai_help"]["api_key"] = "test-key"
        path = self.add_event("e1", "cat", description="a cat")

        class Resp:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": json.dumps({
                    "matches": True, "label": "cat", "description": "a gray cat",
                    "confidence": 0.9,
                    "box": {"found": True, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
                })}}]}

        orig = genai_help.requests.post
        genai_help.requests.post = lambda *a, **k: Resp()
        try:
            res = genai_help.analyze(self.cfg, path, "object", "cat", ["cat", "dog"], "m",
                                     box="[0.2, 0.3, 0.1, 0.1]")
        finally:
            genai_help.requests.post = orig
        self.assertTrue(res["ok"])
        self.assertTrue(res["matches"])
        self.assertEqual(res["label"], "cat")
        self.assertTrue(res["box"]["found"])
        self.assertAlmostEqual(res["box"]["w"], 0.3)


class TestPreview(Base):
    def test_preview_fit_has_no_black_padding(self):
        from errata.imaging import region_crop_pil, region_crop_pil_fit

        img = Image.new("RGB", (1816, 816), (50, 60, 70))
        box = (0.7836, 0.8946, 0.0402, 0.0944)  # bottom edge of a wide frame
        padded, _ = region_crop_pil(img, box, 640, scale=1.33)
        fitted, _ = region_crop_pil_fit(img, box, 640, scale=1.33)
        self.assertTrue(any(c[1] == (0, 0, 0) for c in padded.getcolors(maxcolors=1 << 20)))
        self.assertFalse(any(c[1] == (0, 0, 0) for c in fitted.getcolors(maxcolors=1 << 20)))


class TestControlsPage(Base):
    def test_page_renders_new_labels_and_limit(self):
        from errata import scheduler as sched_mod
        from errata import webapp as webapp_mod
        from starlette.testclient import TestClient

        sched_mod.Scheduler.start = lambda self: None
        sched_mod.Scheduler.stop = lambda self: None
        app = webapp_mod.create_app(self.cfg)
        with TestClient(app) as c:
            html = c.get("/errata/controls").text
        self.assertEqual(html.count('class="bulk"'), 4)
        self.assertIn("Frigate Description Search", html)
        self.assertIn("Human Verifications", html)
        self.assertIn("Machine Verifications", html)
        self.assertIn("Snapshot Limit", html)
        self.assertIn("Monitor Only (Ignore)", html)
        self.assertNotIn("Auto-approve", html)


class TestResetDb(Base):
    def test_reset_wipes_learned_data_keeps_base_models(self):
        basemodels.seed_builtin_models(self.db)
        self.add_event("e1", "car", correct="car")
        self.db.insert_brand("e1", "usps")
        self.db.reset_training_data()
        self.assertEqual(self.db.queue_count("all"), 0)
        self.assertEqual(self.db.brand_count("all"), 0)
        self.assertEqual(self.db.corrections_count(), 0)
        self.assertGreater(len(self.db.base_models()), 0)


class TestVerificationCounts(Base):
    def test_human_and_machine_counts(self):
        from errata.trainer import verification_counts

        self.add_event("e1", "car", correct="car", description="a car")
        self.add_event("e2", "car", description="a car")
        self.db.set_status("e2", "confirmed")
        counts = verification_counts(self.cfg, self.db)
        self.assertEqual(counts["car"]["human"], 1)
        self.assertGreaterEqual(counts["car"]["machine"], 1)

    def test_event_genai_roundtrip(self):
        self.add_event("e1", "car", description="a car")
        self.db.set_event_genai("e1", json.dumps({"object|car|m": {"created_at": 1}}))
        self.assertIn("object|car|m", self.db.get_event("e1")["genai"])


class TestGenAICache(unittest.TestCase):
    def test_cache_key_and_most_recent(self):
        cache = {
            genai_help.cache_key("object", "cat", "m1"): {"created_at": 1, "label": "cat"},
            genai_help.cache_key("object", "cat", "m2"): {"created_at": 2, "label": "dog"},
            genai_help.cache_key("brand", "usps", "m1"): {"created_at": 3, "label": "usps"},
        }
        key, entry = genai_help.most_recent(cache, "object", "cat")
        self.assertEqual(entry["label"], "dog")
        self.assertEqual(genai_help.most_recent(cache, "object", "dog"), None)
        self.assertEqual(genai_help.most_recent(cache, "brand", "usps")[1]["label"], "usps")

    def test_load_cache_handles_garbage(self):
        self.assertEqual(genai_help.load_cache(None), {})
        self.assertEqual(genai_help.load_cache("not json"), {})
        self.assertEqual(genai_help.load_cache("[1,2]"), {})
        self.assertEqual(genai_help.load_cache('{"a": 1}'), {"a": 1})


class TestGenAIEndpoint(Base):
    def test_second_call_is_cached(self):
        from errata import scheduler as sched_mod
        from errata import webapp as webapp_mod
        from starlette.testclient import TestClient

        self.cfg["review"]["genai_help"]["api_key"] = "test-key"
        model = self.cfg["review"]["genai_help"]["default_model"]
        self.add_event("e1", "cat", description="a cat", box="[0.1,0.1,0.3,0.3]")
        calls = {"n": 0}

        def fake(cfg, path, kind, expected, allowed, model=None, box=None):
            calls["n"] += 1
            return {"ok": True, "model": model, "matches": True, "label": "cat",
                    "description": "a cat", "confidence": 0.9,
                    "box": {"found": True, "x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3}}

        orig = genai_help.analyze
        genai_help.analyze = fake
        sched_mod.Scheduler.start = lambda self: None
        sched_mod.Scheduler.stop = lambda self: None
        try:
            app = webapp_mod.create_app(self.cfg)
            with TestClient(app) as c:
                r1 = c.post("/errata/api/genai_help/e1", json={"kind": "object", "model": model}).json()
                r2 = c.post("/errata/api/genai_help/e1", json={"kind": "object", "model": model}).json()
        finally:
            genai_help.analyze = orig
        self.assertFalse(r1["cached"])
        self.assertTrue(r2["cached"])
        self.assertEqual(calls["n"], 1)
        self.assertIsNotNone(r1["detector_crop"])
        self.assertIsNotNone(r1["genai_crop"])


class TestPreviewGeometry(unittest.TestCase):
    def test_box_in_crop_matches_fit(self):
        from errata.imaging import box_in_crop, preview_geometry, region_crop_pil_fit

        img = Image.new("RGB", (1816, 816), (10, 20, 30))
        box = (0.7836, 0.8946, 0.0402, 0.0944)
        x0, y0, side = preview_geometry(box, *img.size, scale=1.33)
        _crop, box_c = region_crop_pil_fit(img, box, 640, scale=1.33)
        self.assertEqual(box_in_crop(box, x0, y0, side, *img.size), box_c)


class TestVerifications(Base):
    def test_train_only_stops_new_machine_events(self):
        self.add_event("e1", "car", description="a car")
        set_control(self.cfg, self.db, "car", collect_mode="all", machine_verifications="train_only")
        Analyzer(self.cfg, self.db).run()
        self.assertEqual(self.db.get_event("e1")["status"], "ignored")

    def test_train_only_is_trainable_and_pseudo(self):
        self.add_event("e1", "car", description="a car")
        self.db.set_status("e1", "confirmed")
        set_control(self.cfg, self.db, "car", machine_verifications="train_only")
        self.assertIn("car", trainer.trained_class_map(self.cfg, self.db))
        pseudo = trainer._select_pseudo_labels(self.cfg, self.db, {"car": 0})
        self.assertEqual([r["id"] for r in pseudo], ["e1"])

    def test_missing_tracked_labels(self):
        pub = Path(self.cfg["training"]["publish_dir"]) / "published"
        pub.mkdir(parents=True, exist_ok=True)
        (pub / "errata_test.onnx").write_bytes(b"")
        (pub / "errata_test.labels.txt").write_text("cat\n")
        missing = trainer.missing_tracked_labels(self.cfg, "errata_test.onnx")
        self.assertIn("car", missing)
        self.assertNotIn("cat", missing)


class TestPurge(Base):
    def _client(self):
        from errata import scheduler as sched_mod
        from errata import webapp as webapp_mod
        from starlette.testclient import TestClient

        sched_mod.Scheduler.start = lambda self: None
        sched_mod.Scheduler.stop = lambda self: None
        return TestClient(webapp_mod.create_app(self.cfg))

    def test_human_purge_deletes_corrections(self):
        self.add_event("e1", "cat", correct="cat")
        self.assertEqual(self.db.corrections_count(), 1)
        with self._client() as c:
            r = c.post("/errata/controls", data={
                "present_cat": "1", "collect_mode_cat": "all",
                "human_verifications_cat": "purge", "machine_verifications_cat": "collect",
            }, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.db.corrections_count(), 0)
        self.assertEqual(effective_for_label(self.cfg, self.db, "cat")["human_verifications"], "collect")

    def test_machine_purge_deletes_confirmed_events(self):
        self.add_event("e1", "car", description="a car")
        self.db.set_status("e1", "confirmed")
        with self._client() as c:
            c.post("/errata/controls", data={
                "present_car": "1", "collect_mode_car": "all",
                "human_verifications_car": "collect", "machine_verifications_car": "purge",
            })
        self.assertIsNone(self.db.get_event("e1"))


if __name__ == "__main__":
    unittest.main()
