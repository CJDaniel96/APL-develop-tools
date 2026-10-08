"""Tests for the component-name and Type list AOI crop CLI."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
import xml.etree.ElementTree as ET
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

    def test_short_filename_with_component_underscores_and_light_suffix(
        self,
    ) -> None:
        result = parse_filename(
            "IC_TOP_3_SolderLight_2",
            {"ic": "IC", "ic_top": "IC_TOP"},
            ignore_case=True,
        )

        self.assertEqual(
            result,
            FilenameInfo("IC_TOP", "3", "3", "SolderLight_2"),
        )


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

    def test_short_filename_with_same_name_xml(self) -> None:
        self.source = self.source.rename(
            self.source.with_name("IC_TOP_3_UniformLight.jpg")
        )
        self.target = self.output_dir / "NG" / self.source.name
        self.xml_path = self.xml_dir / f"{self.source.stem}.xml"
        self.write_xml(
            '<Component CompName="IC_TOP_3" Type="32-350177-01">'
            '<CompImage><Image X1="10" Y1="20" X2="50" Y2="60" />'
            "</CompImage></Component>"
        )

        self.assertEqual(self.run_cli(), 0)
        self.assertTrue(self.target.is_file())


class MapExportIntegrationTest(unittest.TestCase):
    """Regression tests for short MAP names and differently named XML."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input_dir = self.root / "DATA"
        self.export_dir = self.input_dir / "1001" / "T1Post_XML"
        self.xml_dir = self.export_dir / "XML"
        self.xml_dir.mkdir(parents=True)
        self.output_dir = self.root / "CROPPED"
        self.type_list = self.root / "types.txt"
        self.type_list.write_text("TARGET_TYPE\n", encoding="utf-8")
        self.source = self.add_image("20261001 235938T1")

    def add_image(self, board: str) -> Path:
        """Creates a short image name under a real machine-export path."""
        source = (
            self.export_dir / "MAP" / "20261001" / "T1"
            / "7043-120095-01B" / board / "L25_1_SolderLight.jpg"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 80), color=(10, 20, 30)).save(source)
        return source

    def write_reference(
        self,
        source: Path,
        xml_name: str = "20261001_235938.xml",
        type_value: str = "TARGET_TYPE",
        box: tuple[int, int, int, int] = (10, 20, 50, 60),
        xml_dir: Path | None = None,
        path_on_comp_image: bool = False,
    ) -> Path:
        """Writes XML with an old machine prefix and the image's MAP path."""
        xml_path = (xml_dir or self.xml_dir) / xml_name
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        root = ET.Element("Root")
        component = ET.SubElement(
            root, "Component", CompName="L25_1", Type=type_value
        )
        comp_image = ET.SubElement(component, "CompImage")
        image_node = ET.SubElement(comp_image, "Image")
        for name, value in zip(("X1", "Y1", "X2", "Y2"), box):
            image_node.set(name, str(value))
        path_node = comp_image if path_on_comp_image else image_node
        relative = source.relative_to(self.export_dir)
        path_node.set(
            "PicPath", "C:\\OldMachine\\T1Post_XML\\"
            + "\\".join(relative.parts).lower()
        )
        ET.ElementTree(root).write(xml_path, encoding="utf-8")
        return xml_path

    def run_cli(self, *extra: str) -> int:
        """Runs Type selection over the parent of nested machine exports."""
        return main([
            "-i", str(self.input_dir),
            "-t", str(self.type_list),
            "-o", str(self.output_dir),
            *extra,
        ])

    def target(self, source: Path) -> Path:
        """Returns the expected source-relative crop path."""
        return self.output_dir / source.relative_to(self.input_dir)

    def test_short_name_with_spaces_and_nested_xml_is_cropped(self) -> None:
        self.write_reference(self.source)

        with self.assertLogs("crop_components", level="INFO") as logs:
            self.assertEqual(self.run_cli(), 0)

        self.assertFalse(any(
            "Unrecognized image filename" in message for message in logs.output
        ))
        with Image.open(self.target(self.source)) as cropped:
            self.assertEqual(cropped.size, (40, 40))

    def test_same_filename_on_different_boards_uses_full_map_path(self) -> None:
        other = self.add_image("20261001 235939T1")
        self.write_reference(self.source, "first.xml")
        self.write_reference(other, "second.xml", type_value="OTHER_TYPE")

        self.assertEqual(self.run_cli(), 0)
        self.assertTrue(self.target(self.source).is_file())
        self.assertFalse(self.target(other).exists())

    def test_basename_match_on_another_board_does_not_crop(self) -> None:
        other = self.add_image("20261001 235939T1")
        self.write_reference(other)

        with self.assertLogs("crop_components", level="WARNING"):
            self.assertEqual(self.run_cli(), 1)

        self.assertFalse(self.target(self.source).exists())
        self.assertTrue(self.target(other).is_file())

    def test_identical_regions_in_multiple_xml_files_are_deduplicated(
        self,
    ) -> None:
        self.write_reference(self.source, "first.xml")
        self.write_reference(self.source, "second.xml")

        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(len(list(self.output_dir.rglob("*.jpg"))), 1)

    def test_distinct_requested_regions_get_separate_output_names(self) -> None:
        self.write_reference(self.source, "first.xml")
        self.write_reference(
            self.source, "second.xml", box=(0, 0, 20, 30)
        )
        self.write_reference(
            self.source, "third.xml", type_value="OTHER_TYPE"
        )

        self.assertEqual(self.run_cli(), 0)

        target = self.target(self.source)
        with Image.open(target) as first:
            self.assertEqual(first.size, (40, 40))
        with Image.open(target.with_name(f"{target.stem}_1.jpg")) as second:
            self.assertEqual(second.size, (20, 30))
        self.assertEqual(len(list(self.output_dir.rglob("*.jpg"))), 2)

    def test_component_name_list_also_supports_short_map_names(self) -> None:
        self.write_reference(self.source)
        component_list = self.root / "components.txt"
        component_list.write_text("L25\n", encoding="utf-8")

        result = main([
            "-i", str(self.input_dir),
            "-c", str(component_list),
            "-o", str(self.output_dir),
        ])

        self.assertEqual(result, 0)
        self.assertTrue(self.target(self.source).is_file())

    def test_explicit_xml_directory_is_used(self) -> None:
        external_xml = self.root / "ExternalXML"
        self.write_reference(self.source, xml_dir=external_xml)

        self.assertEqual(self.run_cli("-x", str(external_xml)), 0)
        self.assertTrue(self.target(self.source).is_file())

    def test_pic_path_on_comp_image_is_supported(self) -> None:
        self.write_reference(self.source, path_on_comp_image=True)

        self.assertEqual(self.run_cli(), 0)
        self.assertTrue(self.target(self.source).is_file())

    def test_unrequested_type_skips_without_opening_image(self) -> None:
        self.write_reference(self.source, type_value="OTHER_TYPE")
        self.source.write_bytes(b"not an image")

        self.assertEqual(self.run_cli(), 0)
        self.assertFalse(self.output_dir.exists())

    def test_dry_run_does_not_create_output_directory(self) -> None:
        self.write_reference(self.source)

        self.assertEqual(self.run_cli("--dry-run"), 0)
        self.assertFalse(self.output_dir.exists())

    def test_output_images_and_xml_are_excluded(self) -> None:
        self.output_dir = self.input_dir / "CROPPED"
        self.write_reference(self.source)
        output_xml = self.output_dir / "XML"
        output_xml.mkdir(parents=True)
        (output_xml / "broken.xml").write_text("<Root>", encoding="utf-8")
        old_crop = self.output_dir / self.source.relative_to(self.input_dir)
        old_crop.parent.mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(old_crop)

        self.assertEqual(self.run_cli("--on-exists", "overwrite"), 0)

        with Image.open(old_crop) as cropped:
            self.assertEqual(cropped.size, (40, 40))
        self.assertEqual(len(list(self.output_dir.rglob("*.jpg"))), 1)


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
