"""Tests for YOLO AOI classification rules."""

from __future__ import annotations

import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from ultralytics.engine.results import Results

from scripts.yolo_classify import (
    Detection,
    bbox_is_contained,
    bbox_is_in_image_center,
    classify_detections,
    detection_is_contained,
    detections_from_result,
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
    ):
        return classify_detections(
            items,
            ok_class_id=5,
            ok_count=ok_count,
            image_size=self.image_size,
            center_tolerance=0.25,
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


class ReviewArtifactTest(unittest.TestCase):
    def test_obb_review_outputs_plot_and_enclosing_voc_box(self) -> None:
        from PIL import Image

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
