"""Tests for the component-list AOI crop CLI."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.crop_components import (
    FilenameInfo,
    main,
    parse_filename,
)


class ParseFilenameTest(unittest.TestCase):
    """Filename parsing tests."""

    def test_component_name_may_contain_underscores(self) -> None:
        components = {"ic_top": "IC_TOP", "ic": "IC"}

        result = parse_filename(
            "12_153045_20260728_M01_IC_TOP_3_IC_TOP_3_UniformLight",
            components,
            ignore_case=True,
        )

        self.assertEqual(
            result,
            FilenameInfo(
                component="IC_TOP",
                board_1="3",
                board_2="3",
                light="UniformLight",
            ),
        )

    def test_component_not_in_list_is_not_selected(self) -> None:
        result = parse_filename(
            "12_153045_20260728_M01_R10_3_R10_3_SolderLight",
            {"c1": "C1"},
            ignore_case=True,
        )

        self.assertIsNone(result)


class CliIntegrationTest(unittest.TestCase):
    """End-to-end crop tests with a small generated image and XML."""

    def test_crops_bbox_from_same_name_xml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "DATA"
            image_dir = input_dir / "NG"
            xml_dir = input_dir / "XML"
            output_dir = root / "OUT"
            image_dir.mkdir(parents=True)
            xml_dir.mkdir()

            stem = (
                "12_153045_20260728_M01_"
                "IC_TOP_3_IC_TOP_3_UniformLight"
            )
            source = image_dir / f"{stem}.jpg"
            Image.new("RGB", (100, 80), color=(10, 20, 30)).save(source)

            xml = (
                "<Root><Panel><Board>"
                '<Component CompName="IC_TOP_3"><CompImage>'
                '<Image X1="10" Y1="20" X2="50" Y2="60" />'
                "</CompImage></Component>"
                "</Board></Panel></Root>"
            )
            (xml_dir / f"{stem}.xml").write_text(xml, encoding="utf-8")
            component_list = root / "components.txt"
            component_list.write_text("IC_TOP\n", encoding="utf-8")

            exit_code = main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--component-list",
                    str(component_list),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            target = output_dir / "NG" / source.name
            self.assertEqual(exit_code, 0)
            self.assertTrue(target.is_file())
            with Image.open(target) as cropped:
                self.assertEqual(cropped.size, (40, 40))


if __name__ == "__main__":
    unittest.main()
