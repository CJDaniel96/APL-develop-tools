"""Tests for the component-name and Type list AOI crop CLI."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.crop_components import (
    FilenameInfo,
    main,
    parse_args,
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


class TypeListIntegrationTest(unittest.TestCase):
    """Type filtering tests using real CLI arguments, images, and XML."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.input_dir = root / "DATA"
        self.xml_dir = self.input_dir / "XML"
        self.output_dir = root / "OUT"
        image_dir = self.input_dir / "NG"
        image_dir.mkdir(parents=True)
        self.xml_dir.mkdir()
        self.stem = (
            "12_153045_20260728_M01_IC_TOP_3_IC_TOP_3_UniformLight"
        )
        self.source = image_dir / f"{self.stem}.jpg"
        self.target = self.output_dir / "NG" / self.source.name
        self.xml_path = self.xml_dir / f"{self.stem}.xml"
        Image.new("RGB", (100, 80), color=(10, 20, 30)).save(self.source)
        self.type_list = root / "types.txt"
        self.type_list.write_text("32-350177-01\n", encoding="utf-8")

    def write_xml(self, components: str) -> None:
        """Writes Component nodes inside the expected XML hierarchy."""
        self.xml_path.write_text(
            f"<Root><Panel><Board>{components}</Board></Panel></Root>",
            encoding="utf-8",
        )

    def run_cli(self, *extra: str) -> int:
        """Runs Type selection with any additional CLI options."""
        return main([
            "--input-dir", str(self.input_dir),
            "--type-list", str(self.type_list),
            "--output-dir", str(self.output_dir),
            *extra,
        ])

    def test_crops_requested_type_with_bom_and_multiple_list_values(
        self,
    ) -> None:
        self.type_list.write_text(
            "\ufeff\nOTHER_TYPE\n32-350177-01\n32-350177-01\n",
            encoding="utf-8",
        )
        self.write_xml(
            '<Component CompName="IC_TOP_3" Type="32-350177-01">'
            '<CompImage><Image X1="10" Y1="20" X2="50" Y2="60" />'
            "</CompImage></Component>"
        )

        self.assertEqual(self.run_cli(), 0)

        with Image.open(self.target) as cropped:
            self.assertEqual(cropped.size, (40, 40))

    def test_requested_type_on_another_component_does_not_select_image(
        self,
    ) -> None:
        self.write_xml(
            '<Component CompName="R10_3" Type="32-350177-01">'
            '<CompImage><Image X1="0" Y1="0" X2="5" Y2="5" />'
            "</CompImage></Component>"
            '<Component CompName="IC_TOP_3" Type="OTHER_TYPE">'
            '<CompImage><Image X1="10" Y1="20" X2="50" Y2="60" />'
            "</CompImage></Component>"
        )

        self.assertEqual(self.run_cli(), 0)
        self.assertFalse(self.output_dir.exists())

    def test_missing_type_is_not_selected(self) -> None:
        self.write_xml(
            '<Component CompName="IC_TOP_3"><CompImage>'
            '<Image X1="10" Y1="20" X2="50" Y2="60" />'
            "</CompImage></Component>"
        )

        self.assertEqual(self.run_cli(), 0)
        self.assertFalse(self.output_dir.exists())

    def test_unrequested_type_skips_even_without_an_unambiguous_image(
        self,
    ) -> None:
        for image_nodes in (
            "",
            '<Image X1="10" Y1="20" X2="50" Y2="60" />'
            '<Image X1="0" Y1="0" X2="5" Y2="5" />',
        ):
            with self.subTest(image_nodes=image_nodes):
                self.write_xml(
                    '<Component CompName="IC_TOP_3" Type="OTHER_TYPE">'
                    f"<CompImage>{image_nodes}</CompImage></Component>"
                )

                self.assertEqual(self.run_cli(), 0)
                self.assertFalse(self.output_dir.exists())

    def test_type_value_is_case_sensitive_unless_requested(self) -> None:
        self.type_list.write_text("ic-part\n", encoding="utf-8")
        self.write_xml(
            '<Component CompName="IC_TOP_3" Type="IC-PART"><CompImage>'
            '<Image X1="10" Y1="20" X2="50" Y2="60" />'
            "</CompImage></Component>"
        )

        self.assertEqual(self.run_cli(), 0)
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(self.run_cli("--ignore-case"), 0)
        self.assertTrue(self.target.is_file())

    def test_ignore_case_also_matches_xml_component_name(self) -> None:
        self.write_xml(
            '<Component CompName="ic_top_3" Type="32-350177-01">'
            '<CompImage><Image X1="10" Y1="20" X2="50" Y2="60" />'
            "</CompImage></Component>"
        )

        self.assertEqual(self.run_cli("--ignore-case"), 0)
        self.assertTrue(self.target.is_file())

    def test_type_child_element_with_namespace_and_lowercase_tags(self) -> None:
        self.write_xml(
            '<a:component xmlns:a="urn:aoi">'
            "<a:compname>IC_TOP_3</a:compname>"
            "<a:type> 32-350177-01 </a:type><a:compimage>"
            '<a:image X1="10" Y1="20" X2="50" Y2="60" />'
            "</a:compimage></a:component>"
        )

        self.assertEqual(self.run_cli(), 0)
        self.assertTrue(self.target.is_file())

    def test_source_filename_is_resolved_before_filtering_type(self) -> None:
        self.write_xml(
            '<Component CompName="IC_TOP_3" Type="32-350177-01">'
            '<CompImage><Image FileName="another_image.jpg" '
            'Light="UniformLight" X1="0" Y1="0" X2="5" Y2="5" />'
            "</CompImage></Component>"
            '<Component CompName="IC_TOP_3" Type="OTHER_TYPE">'
            f'<CompImage><Image PicPath="C:\\DATA\\{self.source.name}" '
            'X1="10" Y1="20" X2="50" Y2="60" />'
            "</CompImage></Component>"
        )

        self.assertEqual(self.run_cli(), 0)
        self.assertFalse(self.output_dir.exists())

    def test_dry_run_does_not_create_output_directory(self) -> None:
        self.write_xml(
            '<Component CompName="IC_TOP_3" Type="32-350177-01">'
            '<CompImage><Image X1="10" Y1="20" X2="50" Y2="60" />'
            "</CompImage></Component>"
        )

        self.assertEqual(self.run_cli("--dry-run"), 0)
        self.assertFalse(self.output_dir.exists())

    def test_missing_xml_returns_failure(self) -> None:
        with self.assertLogs("crop_components", level="WARNING"):
            self.assertEqual(self.run_cli(), 1)
        self.assertFalse(self.output_dir.exists())

    def test_malformed_xml_returns_failure(self) -> None:
        self.xml_path.write_text("<Root>", encoding="utf-8")

        with self.assertLogs("crop_components", level="WARNING"):
            self.assertEqual(self.run_cli(), 1)
        self.assertFalse(self.output_dir.exists())

    def test_missing_component_returns_failure(self) -> None:
        self.write_xml(
            '<Component CompName="R10_3" Type="32-350177-01">'
            '<CompImage><Image X1="10" Y1="20" X2="50" Y2="60" />'
            "</CompImage></Component>"
        )

        with self.assertLogs("crop_components", level="WARNING"):
            self.assertEqual(self.run_cli(), 1)
        self.assertFalse(self.output_dir.exists())

    def test_ambiguous_image_nodes_return_failure(self) -> None:
        self.write_xml(
            '<Component CompName="IC_TOP_3" Type="32-350177-01">'
            '<CompImage><Image X1="10" Y1="20" X2="50" Y2="60" />'
            '<Image X1="0" Y1="0" X2="5" Y2="5" />'
            "</CompImage></Component>"
        )

        with self.assertLogs("crop_components", level="WARNING"):
            self.assertEqual(self.run_cli(), 1)
        self.assertFalse(self.output_dir.exists())

    def test_invalid_type_list_returns_usage_error(self) -> None:
        for content in (None, "\n  \n"):
            with self.subTest(content=content):
                if content is None:
                    self.type_list.unlink()
                else:
                    self.type_list.write_text(content, encoding="utf-8")
                with self.assertLogs("crop_components", level="ERROR"):
                    self.assertEqual(self.run_cli(), 2)
                self.assertFalse(self.output_dir.exists())


class SelectionArgumentsTest(unittest.TestCase):
    """Selection modes must be explicit and mutually exclusive."""

    def test_requires_exactly_one_selection_list(self) -> None:
        base = ["-i", "DATA", "-o", "OUT"]
        for selection in (
            [],
            ["-c", "components.txt", "-t", "types.txt"],
        ):
            with self.subTest(selection=selection):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        parse_args([*base, *selection])
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
