"""Tests for YOLO AOI classification rules."""

from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.yolo_classify import (
    Detection,
    bbox_is_contained,
    bbox_is_in_image_center,
    classify_detections,
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
