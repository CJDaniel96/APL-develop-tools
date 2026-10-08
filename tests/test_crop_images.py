"""Integration tests for AOI crop layouts, names and source mappings."""

from __future__ import annotations

import csv
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from scripts.crop_images import main


class CropImagesTest(unittest.TestCase):
    """Exercises the CLI with generated XML and real image files."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.image_dir = self.root / "IMG"
        self.xml_dir = self.root / "XML"
        self.xml_dir.mkdir()
        self.image_dir.mkdir()
        self.xml = ET.Element("Panel")
        self.board = ET.SubElement(
            self.xml, "Board", BoardSN="BoardA", imulti="3",
        )
        self.xml_path = self.xml_dir / "inspection.xml"

    def add_image(
        self,
        filename: str = "IC_TOP_3_UniformLight.png",
        component_fields: dict[str, str] | None = None,
        image_fields: dict[str, str] | None = None,
        comp_image_fields: dict[str, str] | None = None,
        board_folder: str = "BoardA",
        mode: str = "RGB",
        color: str = "red",
    ) -> Path:
        """Adds a source image and its enclosing Component/Image nodes."""
        relative = Path("MAP/20261008/T1/Product") / board_folder / filename
        source = self.image_dir / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, (40, 30), color=color).save(source)
        if component_fields is None:
            component_fields = {
                "CompName": "IC_TOP_3",
                "Type": "32-350177-01",
                "PackageType": "IC",
            }
        component = ET.SubElement(self.board, "Component", component_fields)
        comp_image = ET.SubElement(
            component, "CompImage", comp_image_fields or {},
        )
        attributes = {
            "PicPath": "C:\\machine\\" + str(relative).replace("/", "\\"),
            "X1": "10", "Y1": "5", "X2": "30", "Y2": "20",
        }
        attributes.update(image_fields or {})
        ET.SubElement(comp_image, "Image", attributes)
        return relative

    def run_cli(self, output: Path, *options: str) -> int:
        """Saves the XML fixture and runs the crop CLI."""
        ET.ElementTree(self.xml).write(self.xml_path, encoding="utf-8")
        return main([
            "--xml-dir", str(self.xml_dir),
            "--image-dir", str(self.image_dir),
            "--output-dir", str(output),
            *options,
        ])

    def manifest_rows(self, manifest: Path) -> list[dict[str, str]]:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_all_layout_and_filename_combinations(self) -> None:
        relative = self.add_image()
        layouts = {
            "source": relative.parent,
            "comp-name": Path("IC_TOP_3"),
            "type": Path("32-350177-01"),
            "package-type": Path("IC"),
            "comp-type-package": Path("IC_TOP_3/32-350177-01/IC"),
        }
        filenames = {
            "original": relative.name,
            "type-package-component": (
                "32-350177-01_IC_IC_TOP_3_UniformLight.jpg"
            ),
            "package-component": "IC_IC_TOP_3_UniformLight.jpg",
            "type-component": "32-350177-01_IC_TOP_3_UniformLight.jpg",
        }
        for layout, parent in layouts.items():
            for name_format, filename in filenames.items():
                with self.subTest(layout=layout, filename=name_format):
                    output = self.root / layout / name_format
                    manifest = output / "manifest.csv"
                    self.assertEqual(self.run_cli(
                        output,
                        "--output-layout", layout,
                        "--filename-format", name_format,
                        "--manifest", str(manifest),
                    ), 0)
                    target = output / parent / filename
                    with Image.open(target) as image:
                        self.assertEqual(image.size, (20, 15))
                        self.assertEqual(
                            image.format,
                            "PNG" if name_format == "original" else "JPEG",
                        )
                        self.assertGreater(image.getpixel((0, 0))[0], 240)
                    rows = self.manifest_rows(manifest)
                    self.assertEqual(len(rows), 1)
                    row = rows[0]
                    self.assertEqual(row["CompName"], "IC_TOP_3")
                    self.assertEqual(row["ComponentName"], "IC_TOP")
                    self.assertEqual(row["PadID"], "3")
                    self.assertEqual(row["Light"], "UniformLight")
                    self.assertEqual(row["BoardSN"], "BoardA")
                    self.assertEqual(row["BoardIndex"], "3")
                    self.assertEqual(
                        row["source_relative"], relative.as_posix(),
                    )
                    self.assertEqual(row["output_path"], str(target))
                    self.assertEqual(row["xml_path"], str(self.xml_path))
                    self.assertEqual(row["X1"], "10")
                    self.assertEqual(row["Y2"], "20")
                    self.assertEqual(row["missing_fields"], "")

    def test_default_preserves_source_path_without_extra_files(self) -> None:
        relative = self.add_image(component_fields={})
        output = self.root / "OUT"
        self.assertEqual(self.run_cli(output), 0)
        self.assertEqual(
            [path.relative_to(output) for path in output.rglob("*")
             if path.is_file()],
            [relative],
        )

    def test_metadata_stays_with_each_component(self) -> None:
        self.add_image(
            "R1_3_SolderLight.png",
            {"CompName": "R1_3", "Type": "R-PART", "PackageType": "R"},
        )
        self.add_image(
            "C1_3_SolderLight.png",
            {"CompName": "C1_3", "Type": "C-PART", "PackageType": "C"},
        )
        output = self.root / "OUT"
        self.assertEqual(self.run_cli(
            output, "--output-layout", "comp-type-package",
            "--filename-format", "type-package-component",
        ), 0)
        self.assertTrue((
            output / "R1_3/R-PART/R/R-PART_R_R1_3_SolderLight.jpg"
        ).is_file())
        self.assertTrue((
            output / "C1_3/C-PART/C/C-PART_C_C1_3_SolderLight.jpg"
        ).is_file())

    def test_namespaced_child_fields_ignore_case(self) -> None:
        self.add_image()
        for node in list(self.xml.iter()):
            node.tag = "{urn:aoi}" + node.tag.lower()
            attributes = dict(node.attrib)
            node.attrib.clear()
            for key, value in attributes.items():
                ET.SubElement(node, "{urn:aoi}" + key.upper()).text = value
        output = self.root / "OUT"
        self.assertEqual(self.run_cli(
            output, "--output-layout", "comp-type-package",
            "--filename-format", "package-component",
        ), 0)
        self.assertTrue((
            output / "IC_TOP_3/32-350177-01/IC/IC_IC_TOP_3_UniformLight.jpg"
        ).is_file())

    def test_xml_pad_and_light_override_filename(self) -> None:
        self.add_image(
            filename="IC_TOP_3_UniformLight_2.png",
            image_fields={"PadID": "7", "LightSource": "solderlight"},
            comp_image_fields={"PadID": "8", "Light": "UniformLight"},
        )
        output = self.root / "OUT"
        self.assertEqual(self.run_cli(
            output, "--output-layout", "type",
            "--filename-format", "package-component",
        ), 0)
        self.assertTrue((
            output / "32-350177-01/IC_IC_TOP_7_SolderLight.jpg"
        ).is_file())

    def test_filename_suffix_and_component_underscores(self) -> None:
        self.add_image(filename="IC_TOP_3_UniformLight_2.png")
        output = self.root / "OUT"
        self.assertEqual(self.run_cli(
            output, "--filename-format", "package-component",
        ), 0)
        self.assertTrue((output / (
            "MAP/20261008/T1/Product/BoardA/IC_IC_TOP_3_UniformLight.jpg"
        )).is_file())

    def test_unconfirmed_compname_suffix_is_preserved(self) -> None:
        self.add_image(component_fields={
            "CompName": "OTHER_COMPONENT_9", "Type": "PART",
        })
        output = self.root / "OUT"
        self.assertEqual(self.run_cli(
            output, "--output-layout", "type",
            "--filename-format", "type-component",
        ), 0)
        self.assertTrue((
            output / "PART/PART_OTHER_COMPONENT_9_3_UniformLight.jpg"
        ).is_file())

    def test_missing_fields_use_placeholders_and_manifest(self) -> None:
        self.add_image(filename="unparsed.png", component_fields={
            "CompName": "IC_TOP",
        })
        output = self.root / "OUT"
        manifest = output / "manifest.csv"
        with self.assertLogs("crop_images", level="WARNING") as logs:
            self.assertEqual(self.run_cli(
                output, "--output-layout", "comp-type-package",
                "--filename-format", "package-component",
                "--manifest", str(manifest),
            ), 0)
        self.assertIn("missing Type", logs.output[0])
        self.assertTrue((output / (
            "IC_TOP/_missing_Type/_missing_PackageType/"
            "_missing_PackageType_IC_TOP__missing_PadID__missing_Light.jpg"
        )).is_file())
        row = self.manifest_rows(manifest)[0]
        self.assertEqual(
            row["missing_fields"], "Type;PackageType;PadID;Light",
        )

    def test_only_selected_modes_require_metadata(self) -> None:
        self.add_image(component_fields={"Type": "PART"})
        output = self.root / "OUT"
        with self.assertLogs("crop_images", level="INFO") as logs:
            self.assertEqual(self.run_cli(
                output, "--output-layout", "type",
            ), 0)
        self.assertFalse(any("WARNING" in line for line in logs.output))
        self.assertTrue((output / "PART/IC_TOP_3_UniformLight.png").is_file())

    def test_metadata_values_cannot_create_extra_path_levels(self) -> None:
        self.add_image(component_fields={
            "CompName": "../IC:TOP", "Type": "..", "PackageType": "CON",
        })
        output = self.root / "OUT"
        self.assertEqual(self.run_cli(
            output, "--output-layout", "comp-type-package",
            "--filename-format", "type-package-component",
        ), 0)
        self.assertTrue((output / (
            ".._IC_TOP/_invalid_Type/_CON/"
            "_invalid_Type__CON_.._IC_TOP_3_UniformLight.jpg"
        )).is_file())

    def test_grouped_collisions_and_existing_file_policies(self) -> None:
        self.add_image(color="red")
        self.add_image(board_folder="BoardB", color="blue")
        filename = "IC_IC_TOP_3_UniformLight.jpg"
        for policy in ("suffix", "skip", "overwrite"):
            with self.subTest(policy=policy):
                output = self.root / policy
                folder = output / "IC"
                folder.mkdir(parents=True)
                target = folder / filename
                Image.new("RGB", (1, 1), color="green").save(target)
                self.assertEqual(self.run_cli(
                    output, "--output-layout", "package-type",
                    "--filename-format", "package-component",
                    "--on-exists", policy,
                ), 0)
                expected_count = {"suffix": 3, "skip": 1, "overwrite": 2}
                self.assertEqual(len(list(folder.glob("*.jpg"))),
                                 expected_count[policy])
                with Image.open(target) as image:
                    expected_size = (
                        (20, 15) if policy == "overwrite" else (1, 1)
                    )
                    self.assertEqual(
                        image.size, expected_size,
                    )
                if policy == "overwrite":
                    second = folder / "IC_IC_TOP_3_UniformLight_1.jpg"
                    with Image.open(second) as image:
                        self.assertGreater(image.getpixel((0, 0))[2], 240)

    def test_dry_run_creates_no_images_or_manifest(self) -> None:
        relative = self.add_image()
        source = self.image_dir / relative
        original = source.read_bytes()
        output = self.root / "OUT"
        self.assertEqual(self.run_cli(
            output, "--output-layout", "type",
            "--filename-format", "type-component",
            "--manifest", str(output / "manifest.csv"), "--dry-run",
        ), 0)
        self.assertFalse(output.exists())
        self.assertEqual(source.read_bytes(), original)

    def test_transparent_png_is_reencoded_as_rgb_jpeg(self) -> None:
        self.add_image(mode="RGBA")
        output = self.root / "OUT"
        self.assertEqual(self.run_cli(
            output, "--output-layout", "type",
            "--filename-format", "type-component",
        ), 0)
        with Image.open(
            output / "32-350177-01/32-350177-01_IC_TOP_3_UniformLight.jpg"
        ) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.mode, "RGB")

    def test_input_cannot_be_used_as_output(self) -> None:
        relative = self.add_image()
        source = self.image_dir / relative
        original = source.read_bytes()
        self.assertEqual(self.run_cli(self.image_dir), 2)
        self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
