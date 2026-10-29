"""Tests for YOLO AOI classification rules."""

from __future__ import annotations

import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from ultralytics.engine.results import Results

from scripts.yolo_classify import (
    Detection,
    bbox_is_contained,
    bbox_is_in_image_center,
    classify_detections,
    detection_is_contained,
    detection_iou,
    detections_from_result,
    main,
    parse_args,
    save_review_artifacts,
    unique_output_targets,
    write_pascal_voc,
)


def detection(
    class_id: int,
    label: str,
    bbox: tuple[float, float, float, float],
) -> Detection:
    return Detection(class_id, label, 0.9, bbox)


class GeometryTest(unittest.TestCase):
    diamond = Detection(
        5, "good", 0.9, (20, 20, 80, 80),
        ((50, 20), (80, 50), (50, 80), (20, 50)),
    )

    def test_center_uses_bbox_center(self) -> None:
        self.assertTrue(
            bbox_is_in_image_center((40, 40, 60, 60), (100, 100), 0.1)
        )
        self.assertFalse(
            bbox_is_in_image_center((0, 0, 10, 10), (100, 100), 0.1)
        )

    def test_containment_requires_all_edges_inside(self) -> None:
        self.assertTrue(bbox_is_contained((20, 20, 30, 30), (10, 10, 40, 40)))
        self.assertFalse(bbox_is_contained((5, 20, 30, 30), (10, 10, 40, 40)))


    def test_rotated_containment_ignores_empty_aabb_corners(self) -> None:
        inner = detection(1, "scratch", (21, 21, 29, 29))
        self.assertTrue(bbox_is_contained(inner.bbox, self.diamond.bbox))
        self.assertFalse(detection_is_contained(inner, self.diamond))

    def test_rotated_containment_requires_all_corners_inside(self) -> None:
        inner = detection(1, "scratch", (25, 25, 55, 55))
        self.assertTrue(bbox_is_contained(inner.bbox, self.diamond.bbox))
        self.assertFalse(detection_is_contained(inner, self.diamond))

    def test_contained_obb_uses_its_actual_corners(self) -> None:
        inner = Detection(
            1, "scratch", 0.9, (23, 23, 77, 77),
            ((50, 23), (77, 50), (50, 77), (23, 50)),
        )
        self.assertTrue(detection_is_contained(inner, self.diamond))
        reversed_outer = Detection(
            5, "good", 0.9, self.diamond.bbox,
            tuple(reversed(self.diamond.polygon)),
        )
        self.assertTrue(detection_is_contained(inner, reversed_outer))

    def test_rotated_boundary_is_inclusive(self) -> None:
        self.assertTrue(detection_is_contained(self.diamond, self.diamond))
        inner = detection(1, "scratch", (40, 30, 60, 50))
        self.assertTrue(detection_is_contained(inner, self.diamond))

    def test_obb_inside_regular_box(self) -> None:
        outer = detection(5, "good", (10, 10, 90, 90))
        self.assertTrue(detection_is_contained(self.diamond, outer))

    def test_degenerate_obb_does_not_contain_detections(self) -> None:
        outer = Detection(
            5, "good", 0.9, (50, 20, 50, 80),
            ((50, 20), (50, 80), (50, 80), (50, 20)),
        )
        self.assertFalse(detection_is_contained(self.diamond, outer))


class OverlapGeometryTest(unittest.TestCase):
    crossed_boxes = (
        Detection(
            5, "good", 0.9, (15, 15, 85, 85),
            ((20, 15), (85, 80), (80, 85), (15, 20)),
        ),
        Detection(
            5, "good", 0.8, (15, 15, 85, 85),
            ((80, 15), (85, 20), (20, 85), (15, 80)),
        ),
    )

    def test_regular_iou(self) -> None:
        first = detection(5, "good", (0, 0, 4, 2))
        self.assertEqual(detection_iou(first, first), 1.0)
        self.assertEqual(
            detection_iou(first, detection(5, "good", (4, 0, 6, 2))), 0.0,
        )
        self.assertAlmostEqual(
            detection_iou(first, detection(5, "good", (2, 0, 6, 2))), 1 / 3,
        )

    def test_rotated_iou_supports_containment_and_either_winding(self) -> None:
        outer = GeometryTest.diamond
        inner = Detection(
            5, "good", 0.8, (23, 23, 77, 77),
            ((50, 23), (23, 50), (50, 77), (77, 50)),
        )
        self.assertAlmostEqual(detection_iou(outer, inner), 0.81)
        self.assertAlmostEqual(detection_iou(inner, outer), 0.81)
        self.assertEqual(detection_iou(outer, outer), 1.0)

    def test_crossed_obbs_do_not_use_identical_enclosing_boxes(self) -> None:
        first, second = self.crossed_boxes
        self.assertEqual(first.bbox, second.bbox)
        self.assertAlmostEqual(detection_iou(first, second), 0.04)

    def test_mixed_regular_and_obb_iou(self) -> None:
        diamond = GeometryTest.diamond
        enclosing = detection(5, "good", diamond.bbox)
        self.assertEqual(detection_iou(diamond, enclosing), 0.5)

    def test_zero_area_boxes_have_zero_iou(self) -> None:
        for item in (
            detection(5, "good", (0, 0, 0, 10)),
            Detection(
                5, "good", 0.9, (0, 0, 0, 10),
                ((0, 0), (0, 10), (0, 10), (0, 0)),
            ),
        ):
            with self.subTest(item=item):
                self.assertEqual(detection_iou(item, item), 0.0)


class ResultConversionTest(unittest.TestCase):
    names = {5: "good", 1: "scratch"}

    def test_regular_results_keep_existing_boxes(self) -> None:
        result = Results(
            np.zeros((100, 100, 3), dtype=np.uint8),
            path="part.jpg", names=self.names,
            boxes=torch.tensor([[20, 30, 80, 70, 0.9, 5]]),
        )
        items = detections_from_result(result, self.names)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].class_id, 5)
        self.assertEqual(items[0].label, "good")
        self.assertAlmostEqual(items[0].confidence, 0.9)
        self.assertEqual(items[0].bbox, (20, 30, 80, 70))
        self.assertIsNone(items[0].polygon)

    def test_obb_results_preserve_geometry_and_classification(self) -> None:
        result = Results(
            np.zeros((100, 100, 3), dtype=np.uint8),
            path="part.jpg", names=self.names,
            obb=torch.tensor([
                [50, 50, 40, 40, math.pi / 4, 0.9, 5],
                [25, 25, 4, 4, math.pi / 4, 0.8, 1],
            ]),
        )
        self.assertIsNone(result.boxes)
        items = detections_from_result(result, self.names)

        self.assertEqual([item.class_id for item in items], [5, 1])
        self.assertEqual([item.label for item in items], ["good", "scratch"])
        self.assertAlmostEqual(items[0].confidence, 0.9)
        self.assertEqual(len(items[0].polygon), 4)
        for actual, expected in zip(
            items[0].bbox,
            (50 - 20 * math.sqrt(2),) * 2
            + (50 + 20 * math.sqrt(2),) * 2,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=4)
        decision = classify_detections(items, 5, 1, (100, 100), 0.25)
        self.assertTrue(decision.is_ok)

    def test_numpy_obb_results_and_unknown_class(self) -> None:
        result = Results(
            np.zeros((100, 100, 3), dtype=np.uint8),
            path="part.jpg", names=self.names,
            obb=np.array([[50, 50, 60, 20, 0.5, 0.8, 9]]),
        )
        items = detections_from_result(result, self.names)
        self.assertEqual(items[0].label, "class_9")
        self.assertEqual(len(items[0].polygon), 4)

    def test_empty_results_for_both_model_types(self) -> None:
        for kwargs in (
            {"boxes": torch.empty((0, 6))},
            {"obb": torch.empty((0, 7))},
        ):
            with self.subTest(kind=next(iter(kwargs))):
                result = Results(
                    np.zeros((100, 100, 3), dtype=np.uint8),
                    path="part.jpg", names=self.names, **kwargs,
                )
                self.assertEqual(detections_from_result(result, self.names), [])


class ClassificationTest(unittest.TestCase):
    image_size = (100, 100)
    ok_box = (20.0, 20.0, 80.0, 80.0)

    def classify(
        self,
        items: list[Detection],
        ok_count: int = 1,
        ok_overlap_iou: float = 0.8,
    ):
        return classify_detections(
            items,
            ok_class_id=5,
            ok_count=ok_count,
            image_size=self.image_size,
            center_tolerance=0.25,
            ok_overlap_iou=ok_overlap_iou,
        )

    def test_centered_expected_ok_is_ok(self) -> None:
        result = self.classify([detection(5, "good", self.ok_box)])

        self.assertTrue(result.is_ok)
        self.assertEqual(result.folder, "OK")

    def test_ng_inside_ok_bbox_is_ng(self) -> None:
        result = self.classify(
            [
                detection(5, "good", self.ok_box),
                detection(3, "scratch", (30, 30, 40, 40)),
                detection(1, "crack", (50, 50, 60, 60)),
            ]
        )

        self.assertFalse(result.is_ok)
        self.assertEqual(result.folder, "crack")

    def test_ng_outside_ok_bbox_is_ignored(self) -> None:
        result = self.classify(
            [
                detection(5, "good", self.ok_box),
                detection(1, "crack", (0, 0, 10, 10)),
            ]
        )

        self.assertTrue(result.is_ok)

    def test_ng_inside_rotated_ok_is_ng(self) -> None:
        result = self.classify([
            GeometryTest.diamond,
            detection(1, "crack", (40, 40, 60, 60)),
        ])
        self.assertFalse(result.is_ok)
        self.assertEqual(result.folder, "crack")

    def test_ng_in_aabb_corner_outside_rotated_ok_is_ignored(self) -> None:
        result = self.classify([
            GeometryTest.diamond,
            detection(1, "crack", (21, 21, 29, 29)),
        ])
        self.assertTrue(result.is_ok)

    def test_no_ok_uses_lowest_ng_class_id(self) -> None:
        result = self.classify(
            [
                detection(7, "dent", (0, 0, 10, 10)),
                detection(2, "scratch", (90, 90, 99, 99)),
            ]
        )

        self.assertFalse(result.is_ok)
        self.assertEqual(result.folder, "scratch")

    def test_wrong_ok_count_without_ng_uses_rule_folder(self) -> None:
        result = self.classify(
            [
                detection(5, "good", self.ok_box),
                detection(5, "good", (30, 30, 70, 70)),
            ]
        )

        self.assertFalse(result.is_ok)
        self.assertEqual(result.folder, "_ok_rule")

    def test_off_center_ok_is_ng(self) -> None:
        result = self.classify(
            [detection(5, "good", (0, 0, 10, 10))]
        )

        self.assertFalse(result.is_ok)
        self.assertEqual(result.folder, "_ok_rule")

    def test_no_detections_uses_diagnostic_folder(self) -> None:
        result = self.classify([])

        self.assertFalse(result.is_ok)
        self.assertEqual(result.folder, "_no_detection")


    def test_overlapping_ok_boxes_count_as_one_without_mutating(self) -> None:
        items = [
            detection(5, "good", self.ok_box),
            detection(5, "good", (21, 21, 81, 81)),
            detection(5, "good", (19, 19, 79, 79)),
        ]
        original = list(items)
        result = self.classify(items)
        self.assertTrue(result.is_ok)
        self.assertIn("counted 3 overlapping OK boxes as 1", result.reason)
        self.assertEqual(items, original)

    def test_near_identical_obbs_count_as_one(self) -> None:
        items = [
            GeometryTest.diamond,
            Detection(
                5, "good", 0.8, (23, 23, 77, 77),
                ((50, 23), (77, 50), (50, 77), (23, 50)),
            ),
        ]
        self.assertTrue(self.classify(items).is_ok)

    def test_crossed_obbs_are_not_counted_as_one(self) -> None:
        result = self.classify(list(OverlapGeometryTest.crossed_boxes))
        self.assertFalse(result.is_ok)
        self.assertEqual(result.folder, "_ok_rule")

    def test_duplicate_boxes_plus_a_separate_object_remain_ng(self) -> None:
        result = self.classify([
            detection(5, "good", self.ok_box),
            detection(5, "good", self.ok_box),
            detection(5, "good", (0, 0, 10, 10)),
        ])
        self.assertFalse(result.is_ok)

    def test_chain_of_overlaps_does_not_merge_separate_endpoints(self) -> None:
        items = [
            detection(5, "good", (20, 20, 80, 80)),
            detection(5, "good", (24, 20, 84, 80)),
            detection(5, "good", (28, 20, 88, 80)),
        ]
        self.assertGreater(detection_iou(items[0], items[1]), 0.8)
        self.assertGreater(detection_iou(items[1], items[2]), 0.8)
        result = self.classify(items)
        self.assertFalse(result.is_ok)
        self.assertIn("below overlap threshold", result.reason)

    def test_overlap_threshold_is_configurable_and_inclusive(self) -> None:
        items = [
            detection(5, "good", self.ok_box),
            detection(5, "good", (20, 20, 95, 80)),
        ]
        self.assertTrue(self.classify(items, ok_overlap_iou=0.8).is_ok)
        self.assertFalse(self.classify(items, ok_overlap_iou=0.81).is_ok)
        self.assertFalse(self.classify(items, ok_overlap_iou=0).is_ok)

    def test_highest_confidence_box_determines_center(self) -> None:
        inside = Detection(5, "good", 0.95, (16, 20, 36, 40))
        outside = Detection(5, "good", 0.4, (14, 20, 34, 40))
        for items in ([inside, outside], [outside, inside]):
            with self.subTest(items=items):
                self.assertTrue(self.classify(items).is_ok)

    def test_duplicate_off_center_objects_remain_ng(self) -> None:
        result = self.classify([
            detection(5, "good", (0, 0, 10, 10)),
            detection(5, "good", (0, 0, 10, 10)),
        ])
        self.assertFalse(result.is_ok)
        self.assertIn("outside image centre region", result.reason)

    def test_ng_inside_lower_confidence_duplicate_still_counts(self) -> None:
        result = self.classify([
            Detection(5, "good", 0.95, self.ok_box),
            Detection(5, "good", 0.4, (18, 18, 82, 82)),
            detection(1, "crack", (18.5, 30, 19.5, 40)),
        ])
        self.assertFalse(result.is_ok)
        self.assertEqual(result.folder, "crack")
        self.assertIn("NG bbox contained", result.reason)

    def test_multiple_expected_objects_keep_original_count_rules(self) -> None:
        items = [detection(5, "good", self.ok_box)] * 2
        self.assertTrue(self.classify(items, ok_count=2).is_ok)
        self.assertFalse(self.classify(items * 2, ok_count=2).is_ok)


class ClassificationCliTest(unittest.TestCase):
    def arguments(self, root: Path) -> list[str]:
        return [
            "--model", "unused.pt", "--source", str(root / "part.png"),
            "--output-dir", str(root / "out"),
            "--ok-label", "good", "--ok-count", "1",
        ]

    def test_default_overlap_threshold(self) -> None:
        args = parse_args(self.arguments(Path("/tmp")))
        self.assertEqual(args.ok_overlap_iou, 0.8)

    def test_cli_applies_threshold_and_keeps_all_review_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "part.png"
            Image.new("RGB", (100, 100)).save(source)
            result = Results(
                np.zeros((100, 100, 3), dtype=np.uint8),
                path=str(source), names={5: "good"},
                boxes=torch.tensor([
                    [20, 20, 80, 80, 0.9, 5],
                    [21, 21, 81, 81, 0.8, 5],
                ]),
            )
            with patch("ultralytics.YOLO") as model_type:
                model_type.return_value.names = result.names
                model_type.return_value.predict.return_value = [result]
                for threshold, category in ((0.8, "OK"), (0, "NG/_ok_rule")):
                    with self.subTest(threshold=threshold):
                        with self.assertLogs("yolo_classify", level="INFO"):
                            exit_code = main(self.arguments(root) + [
                                "--ok-overlap-iou", str(threshold),
                            ])
                        self.assertEqual(exit_code, 0)
                        annotation = (
                            root / "out" / category / "original" / "part.xml"
                        )
                        objects = ET.parse(annotation).getroot().findall(
                            "object"
                        )
                        self.assertEqual(len(objects), 2)

    def test_invalid_overlap_threshold_skips_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "part.png").touch()
            with patch("ultralytics.YOLO") as model_type:
                for value in ("-0.1", "1.1", "nan", "inf"):
                    with self.subTest(value=value):
                        with self.assertLogs("yolo_classify", level="ERROR"):
                            exit_code = main(self.arguments(root) + [
                                "--ok-overlap-iou", value,
                            ])
                        self.assertEqual(exit_code, 2)
                model_type.assert_not_called()


class ReviewArtifactTest(unittest.TestCase):
    def test_obb_review_outputs_plot_and_enclosing_voc_box(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "part.png"
            Image.new("RGB", (100, 100)).save(source)
            result = Results(
                np.zeros((100, 100, 3), dtype=np.uint8),
                path=str(source), names={5: "good"},
                obb=torch.tensor([
                    [50, 50, 40, 40, math.pi / 4, 0.9, 5],
                ]),
            )
            targets = unique_output_targets(
                Path(temporary) / "OK", source.name, set(), "suffix",
            )
            assert targets is not None
            save_review_artifacts(
                result, source, targets,
                detections_from_result(result, result.names), (100, 100),
            )

            self.assertTrue(all(path.exists() for path in targets.all()))
            self.assertEqual(source.read_bytes(), targets.original.read_bytes())
            with Image.open(targets.inference) as preview:
                self.assertEqual(preview.size, (100, 100))
                self.assertIsNotNone(preview.getbbox())
            root = ET.parse(targets.annotation).getroot()
            self.assertEqual(root.findtext("object/name"), "good")
            self.assertEqual(root.findtext("object/bndbox/xmin"), "21")
            self.assertEqual(root.findtext("object/bndbox/ymin"), "21")
            self.assertEqual(root.findtext("object/bndbox/xmax"), "79")
            self.assertEqual(root.findtext("object/bndbox/ymax"), "79")

    def test_voc_xml_contains_image_metadata_and_all_detections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / "OK" / "original" / "part.jpg"
            annotation = original.with_suffix(".xml")
            items = [
                detection(5, "component_ok", (10.2, 20.8, 50.1, 80.9)),
                detection(1, "scratch & dent", (-2.0, 30.0, 105.0, 40.0)),
            ]

            write_pascal_voc(
                annotation,
                original,
                image_size=(100, 90),
                depth=3,
                detections=items,
            )

            root = ET.parse(annotation).getroot()
            self.assertEqual(root.findtext("folder"), "original")
            self.assertEqual(root.findtext("filename"), "part.jpg")
            self.assertEqual(root.findtext("size/width"), "100")
            self.assertEqual(root.findtext("size/height"), "90")
            self.assertEqual(root.findtext("size/depth"), "3")
            objects = root.findall("object")
            self.assertEqual(
                [item.findtext("name") for item in objects],
                ["component_ok", "scratch & dent"],
            )
            self.assertEqual(objects[0].findtext("bndbox/xmin"), "10")
            self.assertEqual(objects[0].findtext("bndbox/ymin"), "20")
            self.assertEqual(objects[0].findtext("bndbox/xmax"), "51")
            self.assertEqual(objects[0].findtext("bndbox/ymax"), "81")
            self.assertEqual(objects[1].findtext("truncated"), "1")
            self.assertEqual(objects[1].findtext("bndbox/xmin"), "0")
            self.assertEqual(objects[1].findtext("bndbox/xmax"), "100")

    def test_related_outputs_receive_the_same_collision_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            category = Path(temporary) / "NG" / "scratch"
            base = unique_output_targets(
                category,
                "part.jpg",
                used=set(),
                on_exists="suffix",
            )
            self.assertIsNotNone(base)
            assert base is not None

            suffixed = unique_output_targets(
                category,
                "part.png",
                used=set(base.all()),
                on_exists="suffix",
            )
            self.assertIsNotNone(suffixed)
            assert suffixed is not None
            self.assertEqual(suffixed.inference.name, "part_1.png")
            self.assertEqual(suffixed.original.name, "part_1.png")
            self.assertEqual(suffixed.annotation.name, "part_1.xml")


if __name__ == "__main__":
    unittest.main()
