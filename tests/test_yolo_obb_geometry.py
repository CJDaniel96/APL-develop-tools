"""OBB measurement, geometry-limit and review-report regression tests."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from ultralytics.engine.results import Results

from scripts import e2e_infer
from scripts.yolo_classify import (
    Decision,
    Detection,
    ObbRules,
    classify_detections,
    detection_edge_distance,
    main,
    measure_ok_objects,
    obb_angle_degrees,
    unique_output_targets,
)


def obb(
    center: tuple[float, float],
    width: float = 20,
    height: float = 6,
    angle: float = 0,
    confidence: float = 0.9,
) -> Detection:
    """Construct a rectangle from known geometry, independent of YOLO."""
    radians = math.radians(angle)
    ux, uy = math.cos(radians) * width / 2, math.sin(radians) * width / 2
    vx, vy = -math.sin(radians) * height / 2, math.cos(radians) * height / 2
    cx, cy = center
    corners = (
        (cx - ux - vx, cy - uy - vy),
        (cx + ux - vx, cy + uy - vy),
        (cx + ux + vx, cy + uy + vy),
        (cx - ux + vx, cy - uy + vy),
    )
    return Detection(
        5, "good", confidence,
        (min(x for x, _ in corners), min(y for _, y in corners),
         max(x for x, _ in corners), max(y for _, y in corners)),
        corners,
    )


def classify(items: list[Detection], **limits: float) -> Decision:
    return classify_detections(
        items, 5, len(items), (200, 100), 0.5,
        obb_rules=ObbRules(**limits),
    )


class ObbMeasurementTest(unittest.TestCase):
    def test_angle_uses_long_axis_and_half_turn_equivalence(self) -> None:
        for angle in (0, 30, -30, 89, -89, 90, 120, 179, 210):
            with self.subTest(angle=angle):
                self.assertAlmostEqual(
                    obb_angle_degrees(obb((100, 50), angle=angle)),
                    (angle + 90) % 180 - 90,
                )
        self.assertAlmostEqual(
            obb_angle_degrees(obb((100, 50), 6, 20, 20)), -70,
        )

    def test_angle_is_independent_of_corner_start_and_winding(self) -> None:
        item = obb((100, 50), angle=37)
        assert item.polygon is not None
        for offset in range(4):
            points = item.polygon[offset:] + item.polygon[:offset]
            for polygon in (points, tuple(reversed(points))):
                self.assertAlmostEqual(
                    obb_angle_degrees(replace(item, polygon=polygon)), 37,
                )

    def test_center_distance_uses_original_image_coordinates(self) -> None:
        result = measure_ok_objects(
            [obb((103, 54), angle=30)], (200, 100), 1, ObbRules(),
        )
        item = result.objects[0]
        self.assertAlmostEqual(item.angle_deg, 30)
        self.assertAlmostEqual(item.offset_x_px, 3)
        self.assertAlmostEqual(item.offset_y_px, 4)
        self.assertAlmostEqual(item.center_distance_px, 5)
        self.assertIsNone(result.min_edge_distance_px)
        self.assertEqual(result.pairs, ())

    def test_shortest_gap_can_end_inside_an_edge(self) -> None:
        first = obb((20, 20), 20, 10)
        second = obb((35, 20), 4, 4)
        self.assertAlmostEqual(detection_edge_distance(first, second), 3)
        self.assertAlmostEqual(detection_edge_distance(second, first), 3)

    def test_rotated_gap_uses_actual_edges(self) -> None:
        first = obb((80, 50), 40, 4, 45)
        shift = 7 / math.sqrt(2)  # 4 px height plus a 3 px gap.
        second = obb((80 - shift, 50 + shift), 40, 4, 45)
        self.assertLess(first.bbox[0], second.bbox[2])
        self.assertLess(second.bbox[0], first.bbox[2])
        self.assertLess(first.bbox[1], second.bbox[3])
        self.assertLess(second.bbox[1], first.bbox[3])
        self.assertAlmostEqual(detection_edge_distance(first, second), 3)

    def test_diagonal_gap_uses_nearest_vertices(self) -> None:
        self.assertAlmostEqual(
            detection_edge_distance(
                obb((20, 20), 10, 10), obb((33, 34), 10, 10)
            ),
            5,
        )

    def test_touch_overlap_crossing_and_containment_have_zero_gap(self) -> None:
        cases = (
            (obb((20, 20), 10, 10), obb((30, 20), 10, 10)),
            (obb((20, 20), 10, 10), obb((30, 30), 10, 10)),
            (obb((20, 20), 10, 10), obb((25, 20), 10, 10)),
            (obb((20, 20), 40, 4, 45), obb((20, 20), 40, 4, -45)),
            (obb((20, 20), 40, 20, 30), obb((20, 20), 4, 2, 30)),
        )
        for first, second in cases:
            with self.subTest(first=first, second=second):
                self.assertEqual(detection_edge_distance(first, second), 0)


class ObbRuleTest(unittest.TestCase):
    def test_measurements_are_automatic_without_enabling_limits(self) -> None:
        decision = classify_detections(
            [obb((100, 50), angle=30)], 5, 1, (200, 100), 0.25,
        )
        self.assertTrue(decision.is_ok)
        self.assertIsNotNone(decision.measurements)
        self.assertIn("angle=30.00deg", decision.reason)

    def test_angle_target_and_tolerance(self) -> None:
        items = [obb((100, 50), angle=30)]
        self.assertTrue(classify(
            items, angle_target_deg=25, angle_tolerance_deg=5,
        ).is_ok)
        decision = classify(items, angle_tolerance_deg=5)
        self.assertFalse(decision.is_ok)
        self.assertEqual(decision.folder, "_ok_rule")
        self.assertIn("angle 30.00deg deviates", decision.reason)

    def test_angle_comparison_wraps_at_180_degrees(self) -> None:
        items = [obb((100, 50), angle=-89)]
        self.assertTrue(classify(
            items, angle_target_deg=89, angle_tolerance_deg=2,
        ).is_ok)
        self.assertFalse(classify(
            items, angle_target_deg=89, angle_tolerance_deg=1.9,
        ).is_ok)
        self.assertTrue(classify(
            [obb((100, 50))], angle_target_deg=180, angle_tolerance_deg=0,
        ).is_ok)

    def test_center_distance_limit_is_inclusive(self) -> None:
        items = [obb((103, 54))]
        self.assertTrue(classify(items, max_center_distance_px=5).is_ok)
        decision = classify(items, max_center_distance_px=4.9)
        self.assertFalse(decision.is_ok)
        self.assertIn("centre distance 5.00px exceeds 4.90px", decision.reason)

    def test_shortest_gap_limit_is_inclusive(self) -> None:
        items = [obb((80, 50)), obb((103, 50))]
        self.assertTrue(classify(items, min_edge_distance_px=3).is_ok)
        decision = classify(items, min_edge_distance_px=3.1)
        self.assertFalse(decision.is_ok)
        self.assertIn("OK#1/OK#2 edge distance 3.00px", decision.reason)

    def test_all_pairs_are_measured_and_closest_pair_causes_ng(self) -> None:
        items = [obb((50, 50)), obb((85, 50)), obb((110, 50))]
        decision = classify(items, min_edge_distance_px=6)
        self.assertFalse(decision.is_ok)
        self.assertIn("OK#2/OK#3", decision.reason)
        assert decision.measurements is not None
        self.assertEqual(len(decision.measurements.pairs), 3)
        self.assertEqual(decision.measurements.min_edge_distance_px, 5)

    def test_contact_is_ng_only_with_a_positive_minimum_gap(self) -> None:
        items = [obb((80, 50)), obb((100, 50))]
        self.assertTrue(classify(items, min_edge_distance_px=0).is_ok)
        self.assertFalse(classify(items, min_edge_distance_px=0.1).is_ok)

    def test_duplicate_boxes_do_not_trigger_a_false_gap_failure(self) -> None:
        items = [obb((100, 50)), obb((100.1, 50), confidence=0.4)]
        decision = classify_detections(
            items, 5, 1, (200, 100), 0.25,
            obb_rules=ObbRules(min_edge_distance_px=10),
        )
        self.assertTrue(decision.is_ok)
        measurement = decision.measurements
        assert measurement is not None
        self.assertEqual(measurement.raw_ok_count, 2)
        self.assertEqual(measurement.effective_ok_count, 1)
        self.assertEqual(len(measurement.objects), 1)
        self.assertEqual(measurement.objects[0].confidence, 0.9)
        self.assertEqual(measurement.pairs, ())

    def test_count_failure_still_preserves_measurements(self) -> None:
        decision = classify_detections(
            [obb((50, 50)), obb((110, 50))], 5, 1, (200, 100), 0.5,
        )
        self.assertFalse(decision.is_ok)
        assert decision.measurements is not None
        self.assertEqual(decision.measurements.effective_ok_count, 2)

    def test_every_enabled_geometry_failure_is_reported(self) -> None:
        decision = classify(
            [obb((110, 50), angle=30)],
            angle_tolerance_deg=5, max_center_distance_px=5,
        )
        self.assertIn("angle", decision.reason)
        self.assertIn("centre distance 10.00px exceeds", decision.reason)


class ObbCliTest(unittest.TestCase):
    def arguments(self, root: Path) -> list[str]:
        return [
            "--model", "unused.pt", "--source", str(root / "part.png"),
            "--output-dir", str(root / "out"),
            "--ok-label", "good", "--ok-count", "2",
            "--center-tolerance", "0.5",
        ]

    def result(self, source: Path, empty: bool = False) -> Results:
        data = torch.empty((0, 7)) if empty else torch.tensor([
            [80, 100, 30, 10, 0, 0.9, 5],
            [120, 100, 30, 10, 0, 0.8, 5],
        ])
        return Results(
            np.zeros((200, 200, 3), dtype=np.uint8),
            path=str(source), names={5: "good"}, obb=data,
        )

    def test_cli_saves_measured_ok_and_ng_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "part.png"
            Image.new("RGB", (200, 200)).save(source)
            result = self.result(source)
            with patch("ultralytics.YOLO") as model_type:
                model = model_type.return_value
                model.names, model.task = result.names, "obb"
                model.predict.return_value = [result]
                for gap, category in ((10, "OK"), (12, "NG/_ok_rule")):
                    with self.subTest(gap=gap):
                        with self.assertLogs("yolo_classify", level="INFO"):
                            exit_code = main(self.arguments(root) + [
                                "--ok-min-edge-distance", str(gap),
                                "--ok-angle-tolerance", "5",
                                "--ok-max-center-distance", "30",
                            ])
                        self.assertEqual(exit_code, 0)
                        category_dir = root / "out" / category
                        payload = json.loads((
                            category_dir / "measurements" / "part.json"
                        ).read_text())
                        self.assertEqual(payload["image"], "part.png")
                        self.assertEqual(payload["verdict"], category[:2])
                        measured = payload["measurements"]
                        self.assertEqual(measured["min_edge_distance_px"], 10)
                        self.assertEqual(len(measured["objects"]), 2)
                        self.assertEqual(measured["objects"][0][
                            "center_distance_px"
                        ], 20)
                        self.assertEqual(measured["rules"][
                            "min_edge_distance_px"
                        ], gap)
                        self.assertTrue((
                            category_dir / "original" / "part.xml"
                        ).exists())
                        self.assertTrue((
                            category_dir / "inference" / "part.png"
                        ).exists())

    def test_empty_obb_result_writes_valid_empty_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "part.png"
            Image.new("RGB", (200, 200)).save(source)
            result = self.result(source, empty=True)
            with patch("ultralytics.YOLO") as model_type:
                model_type.return_value.names = result.names
                model_type.return_value.predict.return_value = [result]
                with self.assertLogs("yolo_classify", level="INFO"):
                    self.assertEqual(main(self.arguments(root)), 0)
            payload = json.loads((
                root / "out/NG/_no_detection/measurements/part.json"
            ).read_text())
            self.assertEqual(payload["measurements"]["objects"], [])
            self.assertEqual(payload["measurements"]["pairs"], [])
            self.assertIsNone(payload["measurements"]["min_edge_distance_px"])

    def test_dry_run_logs_measurements_without_creating_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "part.png"
            Image.new("RGB", (200, 200)).save(source)
            result = self.result(source)
            with patch("ultralytics.YOLO") as model_type:
                model_type.return_value.names = result.names
                model_type.return_value.predict.return_value = [result]
                with self.assertLogs("yolo_classify", level="INFO") as logs:
                    self.assertEqual(main(self.arguments(root) + [
                        "--dry-run",
                    ]), 0)
            self.assertIn(
                "minimum edge distance=10.00px", " ".join(logs.output)
            )
            self.assertFalse((root / "out").exists())

    def test_invalid_geometry_limits_fail_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "part.png").touch()
            with patch("ultralytics.YOLO") as model_type:
                for option, value in (
                    ("--ok-angle-target", "nan"),
                    ("--ok-angle-tolerance", "91"),
                    ("--ok-angle-tolerance", "-1"),
                    ("--ok-max-center-distance", "inf"),
                    ("--ok-min-edge-distance", "-0.1"),
                ):
                    with self.subTest(option=option, value=value):
                        with self.assertLogs("yolo_classify", level="ERROR"):
                            self.assertEqual(main(self.arguments(root) + [
                                option, value,
                            ]), 2)
                model_type.assert_not_called()

    def test_geometry_limits_reject_a_regular_detection_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "part.png"
            Image.new("RGB", (200, 200)).save(source)
            with patch("ultralytics.YOLO") as model_type:
                model = model_type.return_value
                model.names, model.task = {5: "good"}, "detect"
                with self.assertLogs("yolo_classify", level="ERROR"):
                    self.assertEqual(main(self.arguments(root) + [
                        "--ok-min-edge-distance", "10",
                    ]), 2)
                model.predict.assert_not_called()

    def test_measurement_path_participates_in_collision_handling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            category = Path(temporary)
            measurement = category / "measurements/part.json"
            measurement.parent.mkdir()
            measurement.touch()
            self.assertIsNone(unique_output_targets(
                category, "part.png", set(), "skip", True,
            ))
            targets = unique_output_targets(
                category, "part.png", set(), "suffix", True,
            )
            assert targets is not None and targets.measurements is not None
            self.assertEqual(len(targets.all()), 4)
            self.assertEqual(targets.measurements.name, "part_1.json")
            self.assertEqual(targets.original.name, "part_1.png")
            self.assertEqual(targets.annotation.name, "part_1.xml")

    def test_e2e_gate_and_reports_keep_structured_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "part.png"
            Image.new("RGB", (200, 200)).save(source)
            result = self.result(source)
            args = e2e_infer.parse_args([
                "--input", str(source), "--output-dir", str(root / "out"),
                "--yolo-model", "unused.pt", "--ok-label", "good",
                "--ok-count", "2", "--center-tolerance", "0.5",
                "--ok-min-edge-distance", "12",
                "--anomaly-model", "dinomaly",
                "--anomaly-model-dir", str(root), "--dry-run",
            ])
            self.assertIsNone(e2e_infer.validate_args(args))
            trace = e2e_infer.ImageTrace(
                source=str(source), relative=source.name,
                oriented=str(source), stage="oriented",
            )
            stats = e2e_infer.Stats()
            with patch("ultralytics.YOLO") as model_type:
                model = model_type.return_value
                model.names, model.task = result.names, "obb"
                model.predict.return_value = [result]
                passed = e2e_infer.run_yolo_stage([trace], args, stats)
                self.assertEqual(passed, [])
            self.assertEqual(trace.yolo_verdict, "NG")
            json_path, csv_path = e2e_infer.write_report([trace], stats, args)
            payload = json.loads(json_path.read_text())
            self.assertEqual(payload["images"][0]["yolo_measurements"][
                "min_edge_distance_px"
            ], 10)
            with csv_path.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(json.loads(row["yolo_measurements"])[
                "min_edge_distance_px"
            ], 10)


if __name__ == "__main__":
    unittest.main()
