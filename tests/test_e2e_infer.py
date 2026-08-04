"""Tests for the end-to-end pipeline helpers that need no model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.e2e_infer import (
    ImageTrace,
    Stats,
    normalize_anomaly_model,
    orient_images,
    parse_image_size,
    resolve_checkpoint,
)


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


if __name__ == "__main__":
    unittest.main()
