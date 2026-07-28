#!/usr/bin/env python3
"""Crop requested AOI components from images using matching XML metadata.

The input directory is expected to contain an ``XML`` directory and image
files (images may also live in nested directories such as ``NG``). Component
names are read one-per-line from a text file. An image is selected when its
name follows this form and either component token occurs in the list::

    {id}_{timestamp}_{date}_{machine}_{component}_{board}_\
{component}_{board}_{light}.jpg

For a selected image, an XML file with the same stem is located below the XML
directory. The script finds the ``Component`` whose ``CompName`` is
``{component}_{board}``, reads X1/Y1/X2/Y2 from its ``Image`` element, and
writes the cropped image below the output directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path

from PIL import Image, UnidentifiedImageError

_LOGGER = logging.getLogger("crop_components")
_DEFAULT_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)
_COORD_FIELDS = ("X1", "Y1", "X2", "Y2")
_IMAGE_NAME_FIELDS = (
    "PicPath",
    "PicName",
    "FileName",
    "ImageName",
    "Name",
    "Path",
)


@dataclass(frozen=True)
class FilenameInfo:
    """The component-related fields parsed from an AOI image name."""

    component: str
    board_1: str
    board_2: str
    light: str

    @property
    def xml_component_names(self) -> tuple[str, ...]:
        """Returns possible CompName values in filename order."""
        first = f"{self.component}_{self.board_1}"
        second = f"{self.component}_{self.board_2}"
        return (first,) if first == second else (first, second)


@dataclass
class Stats:
    """Counters describing one run."""

    images_seen: int = 0
    selected: int = 0
    written: int = 0
    skipped_existing: int = 0
    not_requested: int = 0
    unparsed_names: int = 0
    missing_xml: int = 0
    ambiguous_xml: int = 0
    xml_errors: int = 0
    missing_component: int = 0
    ambiguous_image_node: int = 0
    bad_bbox: int = 0
    image_errors: int = 0

    def render(self) -> str:
        """Returns all counters as an indented summary."""
        return "\n".join(
            f"  {field.name:<22}: {getattr(self, field.name)}"
            for field in fields(self)
        )

    def has_errors(self) -> bool:
        """Returns whether any selected image failed to produce a crop."""
        return any(
            (
                self.missing_xml,
                self.ambiguous_xml,
                self.xml_errors,
                self.missing_component,
                self.ambiguous_image_node,
                self.bad_bbox,
                self.image_errors,
            )
        )


def _localname(tag: str) -> str:
    """Strips an XML namespace from a tag or attribute name."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _get_field(element: ET.Element, name: str) -> str | None:
    """Reads an attribute or direct child, ignoring case and namespaces."""
    wanted = name.casefold()
    for key, value in element.attrib.items():
        if _localname(key).casefold() == wanted:
            stripped = value.strip()
            return stripped or None
    for child in element:
        if _localname(child.tag).casefold() == wanted:
            stripped = (child.text or "").strip()
            return stripped or None
    return None


def load_components(path: Path, ignore_case: bool) -> dict[str, str]:
    """Loads non-empty component names, preserving their first spelling."""
    lookup: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as component_file:
        for line_number, line in enumerate(component_file, start=1):
            component = line.strip()
            if not component:
                continue
            key = component.casefold() if ignore_case else component
            if key in lookup:
                _LOGGER.debug(
                    "Ignoring duplicate component on line %d: %s",
                    line_number,
                    component,
                )
                continue
            lookup[key] = component
    return lookup


def parse_filename(
    stem: str,
    components: dict[str, str],
    ignore_case: bool,
) -> FilenameInfo | None:
    """Parses a selected component from a filename stem.

    The first four underscore-separated fields are fixed metadata fields.
    Component names are tested longest-first, which makes names containing
    underscores and names that prefix another requested component safe.
    """
    prefix = stem.split("_", 4)
    if len(prefix) != 5 or any(not value for value in prefix[:4]):
        return None
    tail = prefix[4]
    compare_tail = tail.casefold() if ignore_case else tail

    ordered = sorted(components.items(), key=lambda item: len(item[0]), reverse=True)
    for key, component in ordered:
        component_prefix = f"{key}_"
        if not compare_tail.startswith(component_prefix):
            continue

        remainder = tail[len(component) + 1 :]
        board_1, separator, remainder = remainder.partition("_")
        if not separator or not board_1:
            continue

        compare_remainder = (
            remainder.casefold() if ignore_case else remainder
        )
        if not compare_remainder.startswith(component_prefix):
            continue

        remainder = remainder[len(component) + 1 :]
        board_2, separator, light = remainder.partition("_")
        if not separator or not board_2 or not light:
            continue
        return FilenameInfo(component, board_1, board_2, light)
    return None


def _looks_like_known_filename(stem: str) -> bool:
    """Returns whether a stem has at least the four fixed prefix fields."""
    parts = stem.split("_", 4)
    return len(parts) == 5 and all(parts[:4])


def find_images(
    input_dir: Path,
    xml_dir: Path,
    output_dir: Path,
    extensions: tuple[str, ...],
) -> list[Path]:
    """Finds images recursively while excluding XML and output directories."""
    excluded = [xml_dir.resolve()]
    resolved_input = input_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output.is_relative_to(resolved_input):
        excluded.append(resolved_output)
    images: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in extensions:
            continue
        resolved = path.resolve()
        if any(resolved.is_relative_to(directory) for directory in excluded):
            continue
        images.append(path)
    return images


def index_xml_files(xml_dir: Path) -> dict[str, list[Path]]:
    """Indexes XML files by case-insensitive filename stem."""
    index: dict[str, list[Path]] = {}
    for path in sorted(xml_dir.rglob("*")):
        if path.is_file() and path.suffix.casefold() == ".xml":
            index.setdefault(path.stem.casefold(), []).append(path)
    return index


def _matching_xml(
    source: Path,
    input_dir: Path,
    xml_candidates: list[Path],
    xml_dir: Path,
) -> Path | None:
    """Chooses an XML path, using mirrored relative paths when necessary."""
    if len(xml_candidates) == 1:
        return xml_candidates[0]
    if not xml_candidates:
        return None

    source_relative = source.relative_to(input_dir)
    possible_parents = [source_relative.parent]
    if source_relative.parts and source_relative.parts[0].casefold() == "ng":
        possible_parents.append(Path(*source_relative.parts[1:-1]))

    for parent in possible_parents:
        wanted = (xml_dir / parent / f"{source.stem}.xml").resolve()
        exact = [path for path in xml_candidates if path.resolve() == wanted]
        if len(exact) == 1:
            return exact[0]
    return None


def _matches(value: str, expected: str, ignore_case: bool) -> bool:
    """Compares two XML/file values according to the requested case policy."""
    if ignore_case:
        return value.casefold() == expected.casefold()
    return value == expected


def _node_filename_values(element: ET.Element) -> list[str]:
    """Collects possible image names/paths stored on an XML element."""
    values: list[str] = []
    for field in _IMAGE_NAME_FIELDS:
        value = _get_field(element, field)
        if value:
            values.append(value)
    return values


def _value_matches_source(value: str, source: Path) -> bool:
    """Checks whether a Windows/POSIX XML path names the source image."""
    basename = value.strip().strip('"').replace("\\", "/").rsplit("/", 1)[-1]
    return (
        basename.casefold() == source.name.casefold()
        or Path(basename).stem.casefold() == source.stem.casefold()
    )


def find_image_node(
    root: ET.Element,
    source: Path,
    info: FilenameInfo,
    ignore_case: bool,
) -> tuple[ET.Element | None, str]:
    """Finds the Image node for the requested Component.

    Exact source filename metadata is preferred. If that is absent, light
    metadata/text is used. A single remaining Image node is unambiguous.

    Returns:
        ``(node, reason)``. ``node`` is None when no component exists or when
        multiple Image nodes cannot safely be distinguished.
    """
    wanted_names = info.xml_component_names
    candidates: list[tuple[ET.Element, ET.Element]] = []
    component_found = False

    for component in root.iter():
        if _localname(component.tag).casefold() != "component":
            continue
        comp_name = _get_field(component, "CompName")
        if comp_name is None or not any(
            _matches(comp_name, wanted, ignore_case) for wanted in wanted_names
        ):
            continue
        component_found = True

        for comp_image in component:
            if _localname(comp_image.tag).casefold() != "compimage":
                continue
            for image_node in comp_image.iter():
                if image_node is comp_image:
                    continue
                if _localname(image_node.tag).casefold() == "image":
                    candidates.append((comp_image, image_node))

    if not component_found:
        return None, "missing_component"
    if not candidates:
        return None, "missing_component"

    filename_matches = [
        image_node
        for comp_image, image_node in candidates
        if any(
            _value_matches_source(value, source)
            for value in (
                *_node_filename_values(comp_image),
                *_node_filename_values(image_node),
            )
        )
    ]
    if len(filename_matches) == 1:
        return filename_matches[0], "ok"
    if len(filename_matches) > 1:
        return None, "ambiguous_image_node"

    light_matches = [
        image_node
        for comp_image, image_node in candidates
        if any(
            info.light.casefold() == value.casefold()
            or f"_{info.light.casefold()}_" in f"_{value.casefold()}_"
            for value in (
                *comp_image.attrib.values(),
                *image_node.attrib.values(),
            )
        )
    ]
    if len(light_matches) == 1:
        return light_matches[0], "ok"
    if len(candidates) == 1:
        return candidates[0][1], "ok"
    return None, "ambiguous_image_node"


def region_box(
    image_node: ET.Element,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Reads, normalizes, and clamps X1/Y1/X2/Y2 to the image bounds."""
    values: dict[str, str] = {}
    for field in _COORD_FIELDS:
        value = _get_field(image_node, field)
        if value is None:
            return None
        values[field] = value
    try:
        x1, y1, x2, y2 = (
            int(round(float(values[field]))) for field in _COORD_FIELDS
        )
    except ValueError:
        return None

    left, right = sorted((x1, x2))
    upper, lower = sorted((y1, y2))
    left = max(0, left)
    upper = max(0, upper)
    right = min(width, right)
    lower = min(height, lower)
    if right <= left or lower <= upper:
        return None
    return left, upper, right, lower


def _unique_output(
    target: Path,
    used: set[Path],
    on_exists: str,
) -> Path | None:
    """Selects an output path without unintended overwrites."""
    if target not in used:
        if not target.exists() or on_exists == "overwrite":
            return target
        if on_exists == "skip":
            return None

    number = 1
    while True:
        candidate = target.with_name(
            f"{target.stem}_{number}{target.suffix}"
        )
        if candidate not in used and not candidate.exists():
            return candidate
        number += 1


def process_image(
    source: Path,
    input_dir: Path,
    xml_dir: Path,
    output_dir: Path,
    components: dict[str, str],
    xml_index: dict[str, list[Path]],
    ignore_case: bool,
    on_exists: str,
    dry_run: bool,
    used_outputs: set[Path],
    stats: Stats,
) -> None:
    """Selects and crops one source image, recording failures in stats."""
    stats.images_seen += 1
    info = parse_filename(source.stem, components, ignore_case)
    if info is None:
        if _looks_like_known_filename(source.stem):
            stats.not_requested += 1
        else:
            stats.unparsed_names += 1
            _LOGGER.warning("Unrecognized image filename: %s", source)
        return
    stats.selected += 1

    xml_candidates = xml_index.get(source.stem.casefold(), [])
    xml_path = _matching_xml(source, input_dir, xml_candidates, xml_dir)
    if xml_path is None:
        if xml_candidates:
            stats.ambiguous_xml += 1
            _LOGGER.warning(
                "Several XML files match %s and none mirrors its path: %s",
                source,
                ", ".join(str(path) for path in xml_candidates),
            )
        else:
            stats.missing_xml += 1
            _LOGGER.warning("No same-name XML found for %s", source)
        return

    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as error:
        stats.xml_errors += 1
        _LOGGER.warning("Cannot parse XML %s: %s", xml_path, error)
        return

    image_node, reason = find_image_node(
        root, source, info, ignore_case
    )
    if image_node is None:
        if reason == "ambiguous_image_node":
            stats.ambiguous_image_node += 1
            _LOGGER.warning(
                "Multiple Image nodes match component %s in %s; "
                "none identifies %s",
                "/".join(info.xml_component_names),
                xml_path,
                source.name,
            )
        else:
            stats.missing_component += 1
            _LOGGER.warning(
                "CompName %s with an Image node not found in %s",
                "/".join(info.xml_component_names),
                xml_path,
            )
        return

    try:
        with Image.open(source) as image:
            image.load()
            box = region_box(image_node, image.width, image.height)
            if box is None:
                stats.bad_bbox += 1
                _LOGGER.warning(
                    "Invalid bbox for %s in %s", source, xml_path
                )
                return

            relative = source.relative_to(input_dir)
            target = output_dir / relative
            final = _unique_output(target, used_outputs, on_exists)
            if final is None:
                stats.skipped_existing += 1
                _LOGGER.info("Exists, skipping: %s", target)
                return
            used_outputs.add(final)

            if dry_run:
                stats.written += 1
                _LOGGER.info(
                    "[dry-run] %s %s -> %s", source, box, final
                )
                return

            final.parent.mkdir(parents=True, exist_ok=True)
            image.crop(box).save(final)
            stats.written += 1
            _LOGGER.info("%s %s -> %s", source.name, box, final)
    except (UnidentifiedImageError, OSError) as error:
        stats.image_errors += 1
        _LOGGER.warning("Cannot crop/write %s: %s", source, error)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Crop components listed in a text file from AOI images, using "
            "same-name XML files for bbox coordinates."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        type=Path,
        help="Root containing XML/, NG/, and/or image files.",
    )
    parser.add_argument(
        "-c",
        "--component-list",
        required=True,
        type=Path,
        help="UTF-8 text file containing one component name per line.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Crop output root; source-relative directories are preserved.",
    )
    parser.add_argument(
        "-x",
        "--xml-dir",
        type=Path,
        help="XML directory; defaults to <input-dir>/XML.",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=list(_DEFAULT_EXTENSIONS),
        metavar="EXT",
        help="Image extensions to scan.",
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Match component and CompName values case-insensitively.",
    )
    parser.add_argument(
        "--on-exists",
        choices=("suffix", "skip", "overwrite"),
        default="suffix",
        help="What to do when an output image already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report crops without writing files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logs.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show warnings and errors.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Runs the component crop CLI."""
    args = parse_args(argv)
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    input_dir = args.input_dir.resolve()
    xml_dir = (
        args.xml_dir.resolve()
        if args.xml_dir is not None
        else input_dir / "XML"
    )
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        _LOGGER.error("Input directory does not exist: %s", input_dir)
        return 2
    if not xml_dir.is_dir():
        _LOGGER.error("XML directory does not exist: %s", xml_dir)
        return 2
    if output_dir == input_dir:
        _LOGGER.error("Output directory must differ from input directory")
        return 2
    if not args.component_list.is_file():
        _LOGGER.error(
            "Component list does not exist: %s", args.component_list
        )
        return 2

    try:
        components = load_components(
            args.component_list, args.ignore_case
        )
    except (OSError, UnicodeError) as error:
        _LOGGER.error(
            "Cannot read component list %s: %s",
            args.component_list,
            error,
        )
        return 2
    if not components:
        _LOGGER.error("Component list contains no non-empty names")
        return 2

    extensions = tuple(
        extension.casefold()
        if extension.startswith(".")
        else f".{extension.casefold()}"
        for extension in args.ext
    )
    images = find_images(input_dir, xml_dir, output_dir, extensions)
    xml_index = index_xml_files(xml_dir)
    _LOGGER.info(
        "Loaded %d component(s), found %d image(s) and %d XML file(s)",
        len(components),
        len(images),
        sum(len(paths) for paths in xml_index.values()),
    )

    stats = Stats()
    used_outputs: set[Path] = set()
    for source in images:
        process_image(
            source=source,
            input_dir=input_dir,
            xml_dir=xml_dir,
            output_dir=output_dir,
            components=components,
            xml_index=xml_index,
            ignore_case=args.ignore_case,
            on_exists=args.on_exists,
            dry_run=args.dry_run,
            used_outputs=used_outputs,
            stats=stats,
        )

    _LOGGER.info(
        "Done%s. Summary:\n%s",
        " (dry-run)" if args.dry_run else "",
        stats.render(),
    )
    return 1 if stats.has_errors() else 0


if __name__ == "__main__":
    sys.exit(main())
