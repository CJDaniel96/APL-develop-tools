"""Tests for the end-to-end pipeline helpers that need no model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.e2e_infer import (
    ImageTrace,
    Stats,
    _scored_path,
    filter_by_light,
    normalize_anomaly_model,
    orient_images,
    parse_image_size,
    resolve_checkpoint,
    save_yolo_ok_crop,
)
from scripts.yolo_classify import Detection


def write_image(path: Path, size: tuple[int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)
    return path


class ModelNameTest(unittest.TestCase):
    def test_canonical_names_pass_through(self) -> None:
        for name in ("dinomaly", "patchcore", "anomaly_dino"):
            self.assertEqual(normalize_anomaly_model(name), name)

    def test_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_anomaly_model("AnomalyDINO"), "anomaly_dino")
        self.assertEqual(normalize_anomaly_model("patch-core"), "patchcore")

    def test_unknown_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_anomaly_model("padim")


class ImageSizeTest(unittest.TestCase):
    def test_none_and_empty(self) -> None:
        self.assertIsNone(parse_image_size(None))
        self.assertIsNone(parse_image_size(""))

    def test_square_and_pair(self) -> None:
        self.assertEqual(parse_image_size("448"), (448, 448))
        self.assertEqual(parse_image_size("256x320"), (256, 320))


class LightFilterTest(unittest.TestCase):
    def pairs(self, *names: str) -> list[tuple[Path, Path]]:
        return [(Path("/in") / name, Path(name)) for name in names]

    def test_all_keeps_every_image_including_unparsable_names(self) -> None:
        stats = Stats()
        pairs = self.pairs("C1_1_SolderLight.jpg", "anything.jpg")
        self.assertEqual(filter_by_light(pairs, "all", stats), pairs)
        self.assertEqual(stats.other_light, 0)
        self.assertEqual(stats.light_unparsed, 0)

    def test_only_the_requested_light_is_kept(self) -> None:
        stats = Stats()
        pairs = self.pairs(
            "C1_1_SolderLight.jpg",
            "C1_1_UniformLight.jpg",
            "U5_A_3_SolderLight.jpg",
        )
        selected = filter_by_light(pairs, "SolderLight", stats)
        self.assertEqual(
            [source.name for source, _ in selected],
            ["C1_1_SolderLight.jpg", "U5_A_3_SolderLight.jpg"],
        )
        self.assertEqual(stats.other_light, 1)

    def test_dedup_suffix_still_resolves(self) -> None:
        stats = Stats()
        pairs = self.pairs("C1_1_SolderLight_1.jpg")
        self.assertEqual(len(filter_by_light(pairs, "SolderLight", stats)), 1)

    def test_unparsable_names_are_counted_and_skipped(self) -> None:
        stats = Stats()
        pairs = self.pairs("image.jpg", "C1_1_SolderLight.jpg")
        selected = filter_by_light(pairs, "SolderLight", stats)
        self.assertEqual(len(selected), 1)
        self.assertEqual(stats.light_unparsed, 1)


class OrientationTest(unittest.TestCase):
    def test_portrait_is_rotated_and_landscape_is_left_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            portrait = write_image(root / "in" / "a.png", (40, 80))
            landscape = write_image(root / "in" / "sub" / "b.png", (80, 40))
            pairs = [
                (portrait, Path("a.png")),
                (landscape, Path("sub/b.png")),
            ]
            stats = Stats()

            traces = orient_images(
                pairs=pairs,
                work_dir=root / "work",
                target="landscape",
                clockwise=True,
                dry_run=False,
                stats=stats,
            )

            self.assertEqual(stats.images_seen, 2)
            self.assertEqual(stats.rotated, 1)
            self.assertTrue(traces[0].rotated)
            rotated = Path(traces[0].oriented)
            self.assertTrue(rotated.exists())
            with Image.open(rotated) as image:
                self.assertEqual(image.size, (80, 40))

            self.assertFalse(traces[1].rotated)
            self.assertEqual(Path(traces[1].oriented), landscape)

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            portrait = write_image(root / "in" / "a.png", (40, 80))
            stats = Stats()

            traces = orient_images(
                pairs=[(portrait, Path("a.png"))],
                work_dir=root / "work",
                target="landscape",
                clockwise=True,
                dry_run=True,
                stats=stats,
            )

            self.assertEqual(stats.rotated, 1)
            self.assertFalse(Path(traces[0].oriented).exists())

    def test_unreadable_image_is_recorded_as_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = root / "in" / "broken.png"
            broken.parent.mkdir(parents=True)
            broken.write_text("not an image")
            stats = Stats()

            traces = orient_images(
                pairs=[(broken, Path("broken.png"))],
                work_dir=root / "work",
                target="landscape",
                clockwise=True,
                dry_run=False,
                stats=stats,
            )

            self.assertEqual(stats.orient_errors, 1)
            self.assertEqual(traces[0].stage, "orient_error")
            self.assertTrue(traces[0].error)


class CheckpointResolutionTest(unittest.TestCase):
    def test_class_sub_directory_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wanted = root / "R1005" / "weights.ckpt"
            wanted.parent.mkdir(parents=True)
            wanted.touch()
            (root / "C0402").mkdir()
            (root / "C0402" / "weights.ckpt").touch()

            self.assertEqual(resolve_checkpoint(root, "R1005", None), wanted)

    def test_missing_class_directory_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(Path(tmp), "missing", None)

    def test_explicit_checkpoint_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit = root / "explicit.ckpt"
            explicit.touch()
            self.assertEqual(
                resolve_checkpoint(root / "unused", "any", explicit), explicit
            )

    def test_newest_checkpoint_without_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old.ckpt"
            new = root / "nested" / "new.ckpt"
            new.parent.mkdir()
            old.touch()
            new.touch()
            import os

            os.utime(old, (1_000_000, 1_000_000))
            os.utime(new, (2_000_000, 2_000_000))
            self.assertEqual(resolve_checkpoint(root, None, None), new)


class TraceTest(unittest.TestCase):
    def test_fail_marks_the_stage(self) -> None:
        trace = ImageTrace(source="a.png", relative="a.png")
        trace.fail("yolo", "boom")
        self.assertEqual(trace.stage, "yolo_error")
        self.assertEqual(trace.error, "boom")

    def test_traces_are_hashable_by_identity(self) -> None:
        first = ImageTrace(source="a.png", relative="a.png")
        second = ImageTrace(source="a.png", relative="a.png")
        self.assertEqual(len({first, second}), 2)


class YoloOkCropTest(unittest.TestCase):
    def test_crop_uses_ok_bbox_and_is_preferred_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_image(root / "source.png", (100, 80))
            target = root / "work" / "crop.png"
            detections = [
                Detection(3, "component_ok", 0.9, (10.2, 20.8, 60.1, 70.2)),
                Detection(4, "outside_ng", 0.8, (0.0, 0.0, 5.0, 5.0)),
            ]

            box = save_yolo_ok_crop(
                source, target, detections, ok_class_id=3, image_size=(100, 80)
            )

            self.assertEqual(box, (10, 20, 61, 71))
            with Image.open(target) as crop:
                self.assertEqual(crop.size, (51, 51))
            trace = ImageTrace(
                source=str(source),
                relative=source.name,
                oriented=str(source),
                yolo_crop=str(target),
            )
            self.assertEqual(_scored_path(trace), target)

            target.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "crop does not exist"):
                _scored_path(trace)

    def test_multiple_ok_boxes_use_their_enclosing_rectangle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_image(root / "source.png", (100, 80))
            target = root / "crop.png"
            detections = [
                Detection(3, "component_ok", 0.9, (-2.0, 5.0, 20.0, 30.0)),
                Detection(3, "component_ok", 0.8, (60.0, 40.0, 105.0, 90.0)),
            ]

            box = save_yolo_ok_crop(
                source, target, detections, ok_class_id=3, image_size=(100, 80)
            )

            self.assertEqual(box, (0, 5, 100, 80))
            with Image.open(target) as crop:
                self.assertEqual(crop.size, (100, 75))

    def test_missing_ok_bbox_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_image(root / "source.png", (20, 20))
            with self.assertRaisesRegex(ValueError, "no OK-class"):
                save_yolo_ok_crop(
                    source,
                    root / "crop.png",
                    [Detection(4, "ng", 0.9, (1.0, 1.0, 5.0, 5.0))],
                    ok_class_id=3,
                    image_size=(20, 20),
                )


if __name__ == "__main__":
    unittest.main()
