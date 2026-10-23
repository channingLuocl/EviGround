"""Current category, inference, evaluation, and run-identity boundaries."""

from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
from eviground.api.client import APIError

from eviground.categories.hypotheses import (
    gate_categories,
    load_category_objects,
    prepare_tables,
)
from eviground.categories.retrieval import (
    match_retrieval_categories,
)
from eviground.cli.ground import run_case, save_manifest
from eviground.evaluation.metrics import bbox_iou_3d
from eviground.evaluation.reporting import report, summarize
from eviground.evaluation.run_storage import ensure_run_writable
from eviground.grounding import pipeline, react
from eviground.grounding.recovery import sample_recovery, actual_attempts
from eviground.query.parser import parse_query
from eviground.rendering.overview import validate_decision
from tests.helpers import make_export, parsed, response


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source, self.scenes, self.rows = make_export(self.root)
        self.tables = self.root / "tables"
        self.out = self.root / "run"
        env = patch.dict(
            os.environ,
            {
                "EVIGROUND_CATEGORY_ROOT": str(self.tables),
                "EVIGROUND_API_PROVIDER": "openai",
                "EVIGROUND_SYNTAX_BACKEND": "spacy",
            },
        )
        env.start()
        self.addCleanup(env.stop)

    def prepare(self, k=5):
        prepare_tables(self.source, self.scenes, self.tables, ["synthetic"], k)
        return load_category_objects(self.tables / "synthetic" / f"objects_top{k}.json")

    def test_top1_and_top5_have_different_retrieval_pools(self):
        top1 = self.prepare(1)
        top5 = self.prepare(5)
        self.assertEqual(
            [o["bbox_id"] for o in match_retrieval_categories(["chair"], top1)], [2]
        )
        self.assertEqual(
            [o["bbox_id"] for o in match_retrieval_categories(["chair"], top5)], [1, 2]
        )
        self.assertEqual(
            [h["rank"] for h in top5[0]["category_hypotheses"]], [1, 2, 3, 4, 5]
        )
        self.assertEqual(top5[0]["category_hypotheses"][-1]["score"], -0.2)
        self.assertAlmostEqual(top5[0]["category_hypotheses"][-1]["gap_from_top1"], 3.6)

    def test_fusion_scores_have_no_probability_gate(self):
        hypotheses = deepcopy(self.rows[0]["top5"])
        for row in hypotheses:
            row["score"] -= 100
        retained, _ = gate_categories(hypotheses, 5)
        self.assertEqual(len(retained), 5)
        self.assertTrue(all(h["score"] < 0 for h in retained))

    def test_malformed_categories_and_scores_are_rejected(self):
        for field, value in [
            ("category", "stool"),
            ("score", float("nan")),
            ("rank", 5),
        ]:
            rows = deepcopy(self.rows[0]["top5"])
            rows[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                gate_categories(rows)
        with self.assertRaises(ValueError):
            gate_categories(self.rows[0]["top5"], score_type="probability")
        with self.assertRaises(ValueError):
            gate_categories(self.rows[0]["top5"], 3)

    def test_verified_zip_and_directory_match(self):
        original = self.prepare()
        archive = self.root / "predictions.zip"
        with ZipFile(archive, "w") as z:
            for p in self.source.iterdir():
                z.write(p, "export/" + p.name)
        alternate = self.root / "zip_tables"
        prepare_tables(archive, self.scenes, alternate, ["synthetic"], 5)
        self.assertEqual(
            load_category_objects(alternate / "synthetic/objects_top5.json"), original
        )

    def test_changed_export_hash_is_rejected(self):
        with (self.source / "predictions.json").open("a") as f:
            f.write(" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.prepare()

    def test_changed_point_cloud_is_rejected(self):
        with (self.scenes / "synthetic/scene.npz").open("ab") as f:
            f.write(b" ")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.prepare()

    def test_tampered_prepared_hypotheses_are_rejected(self):
        self.prepare()
        path = self.tables / "synthetic/objects_top5.json"
        data = json.loads(path.read_text())
        data["objects"][0]["category_hypotheses"][1]["rank"] = 1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            load_category_objects(path)

    def test_supplemental_categories_are_scrubbed(self):
        objects = self.prepare()
        supplemental = pipeline.structural_objects(self.scenes / "synthetic", objects)
        self.assertEqual(
            supplemental,
            [
                {
                    "bbox_id": 99,
                    "target": "object",
                    "bbox_3d": [0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
                }
            ],
        )
        self.assertNotIn("SECRET", json.dumps(objects + supplemental))

    def test_actual_render_measure_finish_trajectory_has_no_gt_input(self):
        self.prepare()
        query = "the chair nearest to the table"

        def action(kind, **updates):
            return {
                "action": kind,
                "reason": "compare visible candidates",
                "focus_ids": [],
                "context": None,
                "azimuth_deg": None,
                "elevation_deg": None,
                "frame": None,
                "predicted_id": None,
                "evidence": [],
                "unresolved": [],
                **updates,
            }

        finish = action(
            "finish",
            predicted_id=1,
            evidence=[
                {
                    "source_quote": query,
                    "object_ids": [1, 3],
                    "observation": "visually checked chair",
                }
            ],
        )
        actions = [
            finish,
            action("inspect", focus_ids=[1, 2], context="isolated"),
            action("measure", focus_ids=[1, 3]),
            finish,
        ]
        payloads = []

        def fake_chat(messages, **kwargs):
            payloads.append(messages)
            return response(actions[len(payloads) - 1])

        raw = parsed(anchors=["table"])
        raw["Relations"] = [
            {
                "target_category": "chair",
                "anchor_category": "table",
                "anchor_categories": [],
                "relation": "nearest",
                "scope": "",
                "phrase": "nearest to the table",
            }
        ]
        read_text = Path.read_text

        def guarded_read(path, *args, **kwargs):
            if path.name == "objects_gt.json":
                raise AssertionError("Inference read evaluation boxes")
            return read_text(path, *args, **kwargs)

        with (
            patch("eviground.query.parser.chat", return_value=response(raw)),
            patch("eviground.grounding.react.chat", side_effect=fake_chat),
            patch.object(Path, "read_text", guarded_read),
        ):
            result = pipeline.run_query(self.scenes / "synthetic", query, self.out)
        self.assertEqual(result["predicted_id"], 1)
        self.assertEqual(result["api_calls"], 5)
        self.assertEqual(
            result["active_trace"]["react_observations"][0]["tool"], "finish_check"
        )
        self.assertEqual(
            [x["action"] for x in result["active_trace"]["react_actions"]],
            ["finish", "inspect", "measure", "finish"],
        )
        self.assertTrue(all(Path(p).is_file() for p in result["images"]))
        sent = json.dumps(payloads)
        self.assertNotIn("SECRET", sent)
        self.assertNotIn("evaluation_only", sent)
        self.assertFalse(result["gt_categories_supplied_to_model"])
        self.assertFalse(result["query_target_id_supplied_to_model"])
        self.assertIsNone(sample_recovery.get())
        self.assertFalse((self.out / "pending_inference.json").exists())

    def test_empty_category_pool_abstains_without_react(self):
        self.prepare()
        with (
            patch(
                "eviground.grounding.pipeline.parse_query",
                return_value=parsed("zeppelin"),
            ),
            patch("eviground.grounding.pipeline.ground") as model,
        ):
            result = pipeline.run_query(
                self.scenes / "synthetic", "the zeppelin", self.out
            )
        model.assert_not_called()
        self.assertIsNone(result["predicted_id"])

    def test_api_failure_keeps_diagnostic_and_restores_context(self):
        self.prepare()
        with (
            patch(
                "eviground.grounding.pipeline.parse_query",
                side_effect=ValueError("synthetic failure"),
            ),
            self.assertRaises(ValueError),
        ):
            pipeline.run_query(self.scenes / "synthetic", "the chair", self.out)
        self.assertEqual(
            json.loads((self.out / "run.json").read_text())["error_stage"],
            "parse_query",
        )
        self.assertFalse((self.out / "prediction.json").exists())
        self.assertIsNone(sample_recovery.get())

    def test_measurement_direction_overlap_and_invalid_frame(self):
        objects = self.prepare()
        result = react.measure(
            [1, 3],
            objects,
            parsed(anchors=["table"]),
            objects[:2],
            "the chair",
            {"origin_ids": [3], "facing_ids": [3]},
        )
        self.assertIsNotNone(result["observer_frame_error"])
        row = next(
            p for p in result["directed_pair_measurements"] if p["object_id"] == 1
        )
        self.assertEqual(row["delta_xyz_m"], [-4.0, 0.0, 0.0])
        self.assertEqual(row["xy_center_m"], 4.0)
        self.assertEqual(row["xy_box_gap_m"], 3.0)
        self.assertEqual(row["xy_overlap_fraction"], 0.0)

    def test_failed_api_attempts_are_recorded_once_for_parser_and_react(self):
        self.prepare()
        for failed_stage in ("parse_query", "react_decision"):
            directory = self.root / failed_stage
            diagnostic = {"failure_kind": "timeout", "api_attempts": [{}, {}]}

            def fail_ground(*args):
                args[8].append({"stage": "react_decision"})
                raise APIError("synthetic timeout", diagnostic=diagnostic)

            parse_result = parsed()
            with (
                self.subTest(stage=failed_stage),
                patch(
                    "eviground.grounding.pipeline.parse_query",
                    side_effect=(
                        APIError("synthetic timeout", diagnostic=diagnostic)
                        if failed_stage == "parse_query"
                        else None
                    ),
                    return_value=parse_result,
                ),
                patch("eviground.grounding.pipeline.ground", side_effect=fail_ground),
                self.assertRaises(APIError),
            ):
                pipeline.run_query(self.scenes / "synthetic", "the chair", directory)
            record = json.loads((directory / "run.json").read_text())
            self.assertEqual(record["error_stage"], failed_stage)
            self.assertEqual(sum(actual_attempts(c) for c in record["calls"]), 2)

    def test_observer_frame_uses_perpendicular_right_axis(self):
        objects = self.prepare()
        frame = react.observer_frame(
            {"origin_ids": [1], "facing_ids": [3]}, objects, objects
        )
        self.assertEqual(frame["forward_xy"], [1.0, 0.0])
        self.assertEqual(frame["right_xy"], [0.0, -1.0])

    def test_parser_has_no_legacy_mode_or_answer_argument(self):
        self.assertEqual(
            list(inspect.signature(parse_query).parameters),
            ["query", "calls", "scene_categories"],
        )
        self.assertEqual(
            list(inspect.signature(pipeline.run_query).parameters),
            ["scene_dir", "query", "out", "top_k", "review_limit"],
        )

    def test_invalid_final_id_or_unquoted_evidence_is_rejected(self):
        for data in [
            {"predicted_id": 99, "reason": "bad", "evidence": []},
            {
                "predicted_id": 1,
                "reason": "bad",
                "evidence": [
                    {
                        "source_quote": "invented",
                        "object_ids": [1],
                        "observation": "unknown",
                    }
                ],
            },
        ]:
            with self.assertRaises(ValueError):
                validate_decision(data, "the chair", [1], [1, 2])

    def test_null_and_error_stay_in_full_denominator(self):
        rows = [
            {"predicted_id": 1, "correct": True, "iou_3d": 1.0},
            {"predicted_id": None, "correct": False},
            {"predicted_id": None, "correct": False, "error": "failed"},
        ]
        summary = summarize(rows, 3)
        self.assertEqual(summary["accuracy"], 1 / 3)
        self.assertEqual(
            (summary["null_count"], summary["errors"], summary["abstained"]), (2, 1, 1)
        )
        partial = summarize(rows[:1], 3)
        self.assertEqual(partial["accuracy"], 1 / 3)
        self.assertEqual(partial["processed_accuracy"], 1.0)
        self.assertFalse(partial["complete"])

    def test_no_cache_marker_can_enable_cumulative_evaluation(self):
        self.out.mkdir()
        (self.out / "correct_cache_sync.json").write_text("{}")
        rows = [
            {
                "csv_row": 1,
                "predicted_id": None,
                "correct": False,
                "easy": False,
                "view_dep": True,
            }
        ]
        summary = report(self.out, rows, 1)
        self.assertEqual(summary["evaluation_policy"], "single_pass")
        self.assertEqual(summary["accuracy"], 0.0)
        self.assertEqual(
            json.loads((self.out / "results.json").read_text())["groups"]["easy"][
                "False"
            ]["correct"],
            0,
        )

    def test_duplicate_evaluation_rows_are_rejected(self):
        self.out.mkdir()
        row = {"csv_row": 1, "correct": False}
        with self.assertRaisesRegex(ValueError, "unique"):
            report(self.out, [row, row], 2)

    def test_same_run_resume_keeps_correct_wrong_and_null_equally(self):
        self.out.mkdir()
        for i, choice in enumerate([1, 2, None], 1):
            row = {
                "csv_row": i,
                "scene": "synthetic",
                "query": "the chair",
                "target_id": 1,
            }
            directory = self.out / f"q{i:05d}"
            directory.mkdir()
            saved = {
                **row,
                "category_top_k": 5,
                "predicted_id": choice,
                "correct": choice == 1,
            }
            (directory / "evaluation.json").write_text(json.dumps(saved))
            with patch("eviground.cli.ground.subprocess.run") as worker:
                self.assertEqual(run_case(row, self.scenes, self.out, 5, 3), saved)
                worker.assert_not_called()

    def test_saved_evaluation_for_different_input_is_rejected(self):
        self.out.mkdir()
        directory = self.out / "q00001"
        directory.mkdir()
        row = {"csv_row": 1, "scene": "synthetic", "query": "the chair", "target_id": 1}
        (directory / "evaluation.json").write_text(
            json.dumps({**row, "query": "changed", "category_top_k": 5})
        )
        with self.assertRaisesRegex(ValueError, "different inputs"):
            run_case(row, self.scenes, self.out, 5, 3)

    def test_new_run_does_not_import_results_from_other_directory(self):
        self.out.mkdir()
        old = self.root / "another_run"
        old.mkdir()
        row = {"csv_row": 1, "scene": "synthetic", "query": "the chair", "target_id": 1}
        (old / "evaluation.json").write_text(
            json.dumps({**row, "predicted_id": 1, "correct": True})
        )
        with patch(
            "eviground.cli.ground.subprocess.run", side_effect=OSError("offline")
        ):
            result = run_case(row, self.scenes, self.out, 5, 3)
        self.assertFalse(result["correct"])
        self.assertIsNone(result["predicted_id"])

    def test_manifest_rejects_changed_inputs_code_or_settings(self):
        self.out.mkdir()
        path = self.out / "manifest.json"
        save_manifest(path, {"code": "one", "top_k": 5})
        save_manifest(path, {"code": "one", "top_k": 5})
        for setting in [{"code": "two", "top_k": 5}, {"code": "one", "top_k": 1}]:
            with self.assertRaisesRegex(ValueError, "changed"):
                save_manifest(path, setting)

    def test_results_without_run_manifest_cannot_seed_a_new_run(self):
        self.out.mkdir()
        (self.out / "q00001").mkdir()
        with self.assertRaisesRegex(ValueError, "without a matching run manifest"):
            save_manifest(self.out / "manifest.json", {"code": "current", "top_k": 5})

    def test_frozen_output_cannot_be_modified_through_symlink(self):
        self.out.mkdir()
        (self.out / "FROZEN.json").write_text("{}")
        alias = self.root / "alias"
        alias.symlink_to(self.out, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "read-only"):
            ensure_run_writable(alias / "nested")

    def test_iou_uses_geometry_only(self):
        box = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
        self.assertEqual(bbox_iou_3d(box, box), 1.0)
        self.assertEqual(bbox_iou_3d(box, [4.0, 0.0, 0.0, 2.0, 2.0, 2.0]), 0.0)
