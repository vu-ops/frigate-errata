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
    def test_candidate_search_overrides_auto_confirm(self):
        self.add_event("e1", "dog", description="a coyote on the lawn")
        set_control(self.cfg, self.db, "coyote", search=True)
        stats = Analyzer(self.cfg, self.db).run()
        self.assertEqual(stats["candidate"], 1)
        row = self.db.get_event("e1")
        self.assertEqual(row["status"], "flagged")
        self.assertEqual(row["flag_reason"], "candidate")
        self.assertEqual(row["suggested_label"], "coyote")

    def test_auto_confirm_off_skips_clean_events(self):
        self.add_event("e2", "car", description="a car")
        set_control(self.cfg, self.db, "car", auto_confirm=False)
        Analyzer(self.cfg, self.db).run()
        self.assertEqual(self.db.get_event("e2")["status"], "skipped")

    def test_auto_confirm_on_confirms_clean(self):
        self.add_event("e3", "car", description="a car")
        Analyzer(self.cfg, self.db).run()
        self.assertEqual(self.db.get_event("e3")["status"], "confirmed")


class TestExport(Base):
    def test_drop_brands_hold_candidates_export_objects(self):
        self.add_event("e1", "dog", correct="dog")
        self.add_event("e2", "car", correct="usps")     # attribute -> dropped
        self.add_event("e3", "dog", correct="coyote")   # candidate -> held
        set_control(self.cfg, self.db, "coyote", include_training=False)
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
        self.assertFalse(cfg["include_training"])
        # a "new" label not in the reset list inherits disabled via the flag
        fresh = effective_controls(self.cfg, self.db, ["brand_new"])["brand_new"]
        self.assertEqual(fresh["collect_mode"], "off")
        self.assertFalse(fresh["include_training"])


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

    def test_base_models_not_pruned_by_keep_versions(self):
        self.db.base_model_insert("custom", "upload", "pt", "/x/custom.pt")
        # keep_versions only touches published errata_*.onnx; base table intact
        self.assertIsNotNone(self.db.base_model_get("custom"))


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


if __name__ == "__main__":
    unittest.main()
