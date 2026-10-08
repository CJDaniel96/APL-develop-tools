#!/usr/bin/env python3
"""Crops image regions defined by AOI machine-output XML files.

Recursively scans an XML directory for XML files. Each XML holds one or more
``<Image>`` nodes (nested root > Panel > Board > Component > CompImage >
Image) that carry a ``PicPath`` plus an ``X1``/``Y1``/``X2``/``Y2`` region.
For every Image node this script:

  1. Reverse-engineers the machine ``PicPath`` into a path *relative* to the
     input image directory, of the form::

         MAP/{YYYYMMDD}/{StationID}/{ProductName}/{BoardID}/{Image}

     by anchoring on the ``MAP`` path segment. ``BoardID`` may contain
     spaces.
  2. Loads the source image from ``<image-dir>/<relative path>``.
  3. Crops the ``(X1, Y1, X2, Y2)`` box.
  4. Saves the crop using the selected directory layout and filename format.
     By default the source path and filename are preserved. Metadata layouts
     use the enclosing Component's CompName, Type and PackageType. If several
     regions map to one output file, later crops get a numeric suffix.

``PicPath`` / ``X1`` / ... are read whether stored as XML attributes or as
child elements, ignoring namespaces and letter case.

Run inside the project's uv environment, e.g.::

    uv run scripts/crop_images.py \
        --xml-dir  ./XML \
        --image-dir ./IMG \
        --output-dir ./OUT
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path

from PIL import Image, UnidentifiedImageError

_LOGGER = logging.getLogger("crop_images")

# Fields we pull out of each <Image> node.
_COORD_FIELDS = ("X1", "Y1", "X2", "Y2")
_PIC_PATH_FIELD = "PicPath"
_IMAGE_TAG = "Image"
_OUTPUT_LAYOUTS = {
    "source": (),
    "comp-name": ("CompName",),
    "type": ("Type",),
    "package-type": ("PackageType",),
    "comp-type-package": ("CompName", "Type", "PackageType"),
}
_FILENAME_FORMATS = {
    "original": (),
    "type-package-component": (
        "Type", "PackageType", "ComponentName", "PadID", "Light",
    ),
    "package-component": ("PackageType", "ComponentName", "PadID", "Light"),
    "type-component": ("Type", "ComponentName", "PadID", "Light"),
}
_LIGHTS = {
    name.lower(): name
    for name in ("SolderLight", "UniformLight", "LowAngleLight")
}
_MANIFEST_FIELDS = (
    "xml_path", "source_path", "source_relative", "output_path",
    "CompName", "Type", "PackageType", "ComponentName", "PadID", "Light",
    "BoardSN", "BoardIndex", "X1", "Y1", "X2", "Y2", "missing_fields",
)


# --------------------------------------------------------------------------- #
# XML helpers (namespace / attribute-vs-element / case tolerant)
# --------------------------------------------------------------------------- #
def _localname(tag: str) -> str:
    """Strips the namespace from an XML tag.

    Args:
        tag: A possibly namespaced tag, e.g. ``{ns}Image``.

    Returns:
        The local part of the tag, e.g. ``Image``.
    """
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _get_field(elem: ET.Element, name: str) -> str | None:
    """Reads a field from an element's attributes, then its children.

    Matching is case-insensitive and namespace-insensitive.

    Args:
        elem: The XML element to read from.
        name: Name of the field to look up, e.g. ``PicPath``.

    Returns:
        The stripped field value, or None when the field is absent or empty.
    """
    target = name.lower()
    for key, value in elem.attrib.items():
        if _localname(key).lower() == target:
            value = value.strip()
            return value or None
    for child in elem:
        if _localname(child.tag).lower() == target:
            text = (child.text or "").strip()
            return text or None
    return None


def _ancestors(
    elem: ET.Element, parents: dict[ET.Element, ET.Element]
) -> tuple[ET.Element, ...]:
    """Returns an element's ancestors, nearest first.

    Args:
        elem: Element whose enclosing metadata is needed.
        parents: Mapping of each child element to its parent.

    Returns:
        Ancestors ordered from the parent to the root.
    """
    result = []
    while elem in parents:
        elem = parents[elem]
        result.append(elem)
    return tuple(result)


def _filename_fields(stem: str) -> tuple[str, str, str] | None:
    """Reads Component Name, Pad ID and light from a source filename.

    Component names can contain underscores. Numeric collision suffixes after
    the light are accepted. No part of the XML CompName is blindly removed.

    Args:
        stem: Source filename without its extension.

    Returns:
        Component Name, Pad ID and canonical light, or None if unparseable.
    """
    tokens = stem.split("_")
    for index in range(len(tokens) - 1, 1, -1):
        light = _LIGHTS.get(tokens[index].lower())
        if light and all(token.isdigit() for token in tokens[index + 1:]):
            component = "_".join(tokens[:index - 1])
            pad_id = tokens[index - 1]
            if component and pad_id:
                return component, pad_id, light
    return None


def _image_metadata(
    elem: ET.Element, source: Path, ancestors: tuple[ET.Element, ...]
) -> dict[str, str | None]:
    """Reads enclosing XML metadata and filename fallbacks for this image.

    CompName, Type and PackageType always come from the nearest Component.
    PadID and Light/LightSource prefer Image, CompImage, then Component fields.
    Component Name only omits a CompName suffix when the source filename
    confirms the same Component Name and Pad ID.

    Args:
        elem: Image node carrying the crop and optional image metadata.
        source: Resolved source image path used for filename fallbacks.
        ancestors: Enclosing XML nodes, nearest first.

    Returns:
        Raw component and board fields plus resolved naming fields.
    """
    component = next(
        (node for node in ancestors
         if _localname(node.tag).lower() == "component"),
        None,
    )
    comp_image = next(
        (node for node in ancestors
         if _localname(node.tag).lower() == "compimage"),
        None,
    )
    board = next(
        (node for node in ancestors
         if _localname(node.tag).lower() == "board"),
        None,
    )
    metadata = {
        name: _get_field(component, name) if component is not None else None
        for name in ("CompName", "Type", "PackageType")
    }
    nodes = [node for node in (elem, comp_image, component) if node is not None]
    pad_id = next(
        (value for node in nodes if (value := _get_field(node, "PadID"))),
        None,
    )
    light = next(
        (value for node in nodes for field in ("Light", "LightSource")
         if (value := _get_field(node, field))),
        None,
    )
    component_name = metadata["CompName"]
    parsed = _filename_fields(source.stem)
    if parsed is not None:
        filename_component, filename_pad, filename_light = parsed
        pad_id = pad_id or filename_pad
        light = light or filename_light
        if component_name is None:
            component_name = filename_component
        elif (
            component_name.casefold()
            == f"{filename_component}_{filename_pad}".casefold()
        ):
            component_name = component_name[:-(len(filename_pad) + 1)]
    if light is not None:
        light = _LIGHTS.get(light.lower(), light)
    metadata.update({
        "ComponentName": component_name,
        "PadID": pad_id,
        "Light": light,
        "BoardSN": _get_field(board, "BoardSN") if board is not None else None,
        "BoardIndex": (
            _get_field(board, "imulti") if board is not None else None
        ),
    })
    return metadata


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def _split_pic_path(pic_path: str) -> list[str]:
    """Splits a Windows or POSIX PicPath into clean segments.

    Args:
        pic_path: The raw ``PicPath`` value read from the XML.

    Returns:
        The path segments, with quotes, empty segments and ``.`` removed.
    """
    normalized = pic_path.strip().strip('"').replace("\\", "/")
    return [seg for seg in normalized.split("/") if seg not in ("", ".")]


def _relative_from_anchor(parts: list[str], anchor: str) -> list[str] | None:
    """Returns the path segments from the last anchor segment onward.

    Args:
        parts: Path segments, as produced by ``_split_pic_path``.
        anchor: The segment to anchor on, matched case-insensitively.

    Returns:
        The segments starting at the last occurrence of the anchor, or None
        if the anchor is not present.
    """
    anchor_l = anchor.lower()
    indices = [i for i, seg in enumerate(parts) if seg.lower() == anchor_l]
    if not indices:
        return None
    return parts[indices[-1]:]


def _resolve_source(root: Path, parts: list[str]) -> Path | None:
    """Resolves ``root/parts`` on disk, tolerating case differences.

    Windows-origin paths are case-insensitive; when running on a
    case-sensitive filesystem we fall back to a per-segment
    case-insensitive match.

    Args:
        root: The image directory the segments are relative to.
        parts: The relative path segments.

    Returns:
        The resolved path, or None if it cannot be found.
    """
    exact = root.joinpath(*parts)
    if exact.exists():
        return exact

    current = root
    for seg in parts:
        if not current.is_dir():
            return None
        match: Path | None = None
        try:
            for entry in current.iterdir():
                if entry.name.lower() == seg.lower():
                    match = entry
                    break
        except OSError:
            return None
        if match is None:
            return None
        current = match
    return current if current.exists() else None


def _unique_output(
    target: Path, used: set[Path], on_exists: str
) -> Path | None:
    """Picks the final output path, honoring the on_exists policy.

    Within a single run two distinct regions never overwrite each other: if
    the path was already produced this run it always gets a numeric suffix.
    For a path that already exists on disk from a *previous* run the
    on_exists policy applies.

    Args:
        target: The desired output path.
        used: Output paths already produced by this run.
        on_exists: Policy for a path that exists on disk from a previous
            run: ``suffix``, ``skip`` or ``overwrite``.

    Returns:
        The path to write to, or None when the policy is ``skip`` and the
        target already exists.
    """
    produced_this_run = target in used
    exists_on_disk = target.exists()

    if not produced_this_run:
        if not exists_on_disk:
            return target
        if on_exists == "overwrite":
            return target
        if on_exists == "skip":
            return None
        # on_exists == "suffix": fall through to suffixing

    stem, suffix = target.stem, target.suffix
    n = 1
    while True:
        candidate = target.with_name(f"{stem}_{n}{suffix}")
        if candidate not in used and not candidate.exists():
            return candidate
        n += 1


def _safe_segment(value: str | None, field: str) -> str:
    """Makes one metadata value safe as a folder or filename segment.

    Args:
        value: Raw XML or parsed filename value.
        field: Field name used for missing or invalid value placeholders.

    Returns:
        One portable path segment, without directory separators.
    """
    if not value:
        return f"_missing_{field}"
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", value)
    cleaned = cleaned.strip().rstrip(" .")
    if not cleaned:
        return f"_invalid_{field}"
    if re.fullmatch(
        r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?",
        cleaned,
        flags=re.IGNORECASE,
    ):
        cleaned = f"_{cleaned}"
    return cleaned


def _output_target(
    output_dir: Path,
    relative: list[str],
    metadata: dict[str, str | None],
    output_layout: str,
    filename_format: str,
) -> tuple[Path, tuple[str, ...]]:
    """Builds an output path and lists metadata missing for its modes.

    Args:
        output_dir: Root for the cropped images.
        relative: Source path segments starting at the configured anchor.
        metadata: Raw XML and resolved naming fields for this image.
        output_layout: Selected folder layout.
        filename_format: Selected naming format.

    Returns:
        Desired output path and the required fields missing from metadata.
    """
    folder_fields = _OUTPUT_LAYOUTS[output_layout]
    name_fields = _FILENAME_FORMATS[filename_format]
    required = dict.fromkeys((*folder_fields, *name_fields))
    missing = tuple(field for field in required if not metadata.get(field))
    if output_layout == "source":
        parent = output_dir.joinpath(*relative[:-1])
    else:
        parent = output_dir.joinpath(*(
            _safe_segment(metadata[field], field) for field in folder_fields
        ))
    if filename_format == "original":
        filename = relative[-1]
    else:
        filename = "_".join(
            _safe_segment(metadata[field], field) for field in name_fields
        ) + ".jpg"
    return parent / filename, missing


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """Writes this run's successful crops to a UTF-8 CSV with a header.

    Args:
        path: Destination CSV, replaced if it already exists.
        rows: Source mapping rows from successfully saved crops.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    """Counters describing the outcome of a run.

    Attributes:
        xml_files: XML files parsed successfully.
        xml_errors: XML files that could not be parsed.
        image_nodes: ``<Image>`` nodes encountered.
        written: Crops written (or that would be written, when dry-running).
        skipped_existing: Nodes skipped because the output already existed.
        no_pic_path: Nodes without a usable ``PicPath``.
        no_anchor: Nodes whose ``PicPath`` lacked the anchor segment.
        missing_source: Nodes whose source image was not found on disk.
        bad_region: Nodes with a missing, non-numeric or empty region.
        missing_metadata: Crops missing fields required by their output modes.
        read_errors: Source images that could not be read or cropped.
    """

    xml_files: int = 0
    xml_errors: int = 0
    image_nodes: int = 0
    written: int = 0
    skipped_existing: int = 0
    no_pic_path: int = 0
    no_anchor: int = 0
    missing_source: int = 0
    bad_region: int = 0
    missing_metadata: int = 0
    read_errors: int = 0

    def render(self) -> str:
        """Returns the counters as an indented, one-per-line summary."""
        return "\n".join(
            f"  {f.name:<18}: {getattr(self, f.name)}" for f in fields(self)
        )


# --------------------------------------------------------------------------- #
# Core processing
# --------------------------------------------------------------------------- #
def _region_box(
    elem: ET.Element, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Builds a clamped, normalized crop box from a node's X1/Y1/X2/Y2.

    Args:
        elem: The ``<Image>`` element carrying the coordinates.
        width: Width of the source image, used to clamp the box.
        height: Height of the source image, used to clamp the box.

    Returns:
        A ``(left, upper, right, lower)`` box clamped to the image bounds,
        or None if the coordinates are missing, non-numeric, or describe an
        empty region.
    """
    raw: dict[str, str] = {}
    for name in _COORD_FIELDS:
        value = _get_field(elem, name)
        if value is None:
            _LOGGER.warning("Image node missing %s; skipping region", name)
            return None
        raw[name] = value
    try:
        x1, y1, x2, y2 = (
            int(round(float(raw[name]))) for name in _COORD_FIELDS
        )
    except ValueError:
        _LOGGER.warning(
            "Image node has non-numeric coordinates %s; skipping", raw
        )
        return None

    left, right = sorted((x1, x2))
    upper, lower = sorted((y1, y2))
    left = max(0, left)
    upper = max(0, upper)
    right = min(width, right)
    lower = min(height, lower)
    if right <= left or lower <= upper:
        _LOGGER.warning(
            "Empty/out-of-bounds region (%s,%s,%s,%s) in %dx%d image; skipping",
            x1, y1, x2, y2, width, height,
        )
        return None
    return left, upper, right, lower


def process_image_node(
    elem: ET.Element,
    image_dir: Path,
    output_dir: Path,
    anchor: str,
    on_exists: str,
    dry_run: bool,
    used_outputs: set[Path],
    stats: Stats,
    *,
    ancestors: tuple[ET.Element, ...] = (),
    xml_path: Path | None = None,
    output_layout: str = "source",
    filename_format: str = "original",
    manifest_rows: list[dict[str, object]] | None = None,
) -> None:
    """Crops the region described by a single ``<Image>`` node.

    Failures are logged and counted rather than raised, so that one bad node
    does not abort the run.

    Args:
        elem: The ``<Image>`` element to process.
        image_dir: Root directory holding the source image tree.
        output_dir: Root directory the crop is written under.
        anchor: Path segment used to reverse-engineer ``PicPath``.
        on_exists: Policy for an output path that already exists on disk:
            ``suffix``, ``skip`` or ``overwrite``.
        dry_run: If True, report the crop without writing it.
        used_outputs: Output paths already produced by this run. Mutated in
            place to reserve the path chosen for this node.
        stats: Counters, updated in place.
        ancestors: Enclosing XML nodes, nearest first.
        xml_path: XML path to include in the optional source manifest.
        output_layout: Directory layout selected from _OUTPUT_LAYOUTS.
        filename_format: Naming mode selected from _FILENAME_FORMATS.
        manifest_rows: Optional list receiving successfully saved crop rows.
    """
    stats.image_nodes += 1

    pic_path = _get_field(elem, _PIC_PATH_FIELD)
    if not pic_path:
        _LOGGER.warning("Image node without %s; skipping", _PIC_PATH_FIELD)
        stats.no_pic_path += 1
        return

    parts = _split_pic_path(pic_path)
    relative = _relative_from_anchor(parts, anchor)
    if relative is None:
        _LOGGER.warning(
            "No %r anchor in PicPath %r; skipping", anchor, pic_path
        )
        stats.no_anchor += 1
        return

    source = _resolve_source(image_dir, relative)
    if source is None:
        _LOGGER.warning(
            "Source image not found: %s", image_dir.joinpath(*relative)
        )
        stats.missing_source += 1
        return

    try:
        with Image.open(source) as img:
            img.load()
            box = _region_box(elem, img.width, img.height)
            if box is None:
                stats.bad_region += 1
                return

            metadata = _image_metadata(elem, source, ancestors)
            target, missing = _output_target(
                output_dir, relative, metadata, output_layout, filename_format,
            )
            if missing:
                stats.missing_metadata += 1
                _LOGGER.warning(
                    "%s: missing %s; using _missing_ placeholders",
                    source.name, ", ".join(missing),
                )
            final = _unique_output(target, used_outputs, on_exists)
            if final is None:
                _LOGGER.info("Exists, skipping: %s", target)
                stats.skipped_existing += 1
                return
            if final.resolve() == source.resolve():
                _LOGGER.warning("Output would overwrite its source: %s", source)
                stats.read_errors += 1
                return
            used_outputs.add(final)

            if dry_run:
                _LOGGER.info("[dry-run] %s %s -> %s", source.name, box, final)
                stats.written += 1
                return

            crop = img.crop(box)
            final.parent.mkdir(parents=True, exist_ok=True)
            if filename_format == "original":
                crop.save(final)
            else:
                crop.convert("RGB").save(final, format="JPEG", quality=95)
            _LOGGER.debug("Wrote %s %s -> %s", source.name, box, final)
            stats.written += 1
            if manifest_rows is not None:
                manifest_rows.append({
                    "xml_path": str(xml_path.resolve()) if xml_path else "",
                    "source_path": str(source.resolve()),
                    "source_relative": Path(*relative).as_posix(),
                    "output_path": str(final.resolve()),
                    **metadata,
                    **dict(zip(_COORD_FIELDS, box)),
                    "missing_fields": ";".join(missing),
                })
    except (UnidentifiedImageError, OSError) as exc:
        _LOGGER.warning("Failed to read/crop %s: %s", source, exc)
        stats.read_errors += 1


def process_xml_file(
    xml_path: Path,
    image_dir: Path,
    output_dir: Path,
    anchor: str,
    on_exists: str,
    dry_run: bool,
    used_outputs: set[Path],
    stats: Stats,
    *,
    output_layout: str = "source",
    filename_format: str = "original",
    manifest_rows: list[dict[str, object]] | None = None,
) -> None:
    """Processes every ``<Image>`` node in one XML file.

    A file that cannot be parsed is logged and counted rather than raised.

    Args:
        xml_path: The XML file to read.
        image_dir: Root directory holding the source image tree.
        output_dir: Root directory the crops are written under.
        anchor: Path segment used to reverse-engineer ``PicPath``.
        on_exists: Policy for an output path that already exists on disk:
            ``suffix``, ``skip`` or ``overwrite``.
        dry_run: If True, report the crops without writing them.
        used_outputs: Output paths already produced by this run. Mutated in
            place.
        stats: Counters, updated in place.
        output_layout: Directory layout selected from _OUTPUT_LAYOUTS.
        filename_format: Naming mode selected from _FILENAME_FORMATS.
        manifest_rows: Optional list receiving successfully saved crop rows.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        _LOGGER.error("Cannot parse %s: %s", xml_path, exc)
        stats.xml_errors += 1
        return
    stats.xml_files += 1

    root = tree.getroot()
    parents = {child: parent for parent in root.iter() for child in parent}
    image_nodes = [
        el
        for el in root.iter()
        if _localname(el.tag).lower() == _IMAGE_TAG.lower()
    ]
    _LOGGER.debug("%s: %d Image node(s)", xml_path, len(image_nodes))
    for node in image_nodes:
        process_image_node(
            node, image_dir, output_dir, anchor, on_exists, dry_run,
            used_outputs, stats,
            ancestors=_ancestors(node, parents),
            xml_path=xml_path,
            output_layout=output_layout,
            filename_format=filename_format,
            manifest_rows=manifest_rows,
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Crop image regions defined by AOI XML files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-x", "--xml-dir", required=True, type=Path,
        help="Directory searched recursively for *.xml files.",
    )
    parser.add_argument(
        "-i", "--image-dir", required=True, type=Path,
        help="Root image directory (contains the MAP\\... tree).",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, type=Path,
        help="Root output directory for the selected layout.",
    )
    parser.add_argument(
        "--output-layout", choices=tuple(_OUTPUT_LAYOUTS), default="source",
        help="Directory layout: source path, CompName, Type, PackageType, "
             "or CompName/Type/PackageType.",
    )
    parser.add_argument(
        "--filename-format", choices=tuple(_FILENAME_FORMATS),
        default="original",
        help="Keep the original name, or prefix Component Name/Pad ID/light "
             "with Type/PackageType, PackageType, or Type and save as JPEG.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None, metavar="CSV",
        help="Optional CSV source mapping for this run's saved crops. "
             "Replaces the CSV if it exists; dry-run does not write it.",
    )
    parser.add_argument(
        "--anchor", default="MAP",
        help="Path segment used to reverse-engineer PicPath.",
    )
    parser.add_argument(
        "--on-exists", choices=("suffix", "skip", "overwrite"),
        default="suffix",
        help="What to do when an output file already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be cropped without writing.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose (DEBUG) logging."
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Only warnings and errors."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Runs the crop over every XML file under the given XML directory.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code: 0 after crop processing, 1 if manifest writing
        fails, or 2 for invalid input/output directory or manifest settings.
    """
    args = parse_args(argv)
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    if not args.xml_dir.is_dir():
        _LOGGER.error("XML directory does not exist: %s", args.xml_dir)
        return 2
    if not args.image_dir.is_dir():
        _LOGGER.error("Image directory does not exist: %s", args.image_dir)
        return 2
    if args.image_dir.resolve() == args.output_dir.resolve():
        _LOGGER.error("Input and output directories must be different")
        return 2
    if args.manifest is not None and args.manifest.suffix.lower() != ".csv":
        _LOGGER.error("--manifest must have a .csv extension")
        return 2

    xml_files = sorted(args.xml_dir.rglob("*.xml"))
    if not xml_files:
        _LOGGER.warning("No XML files found under %s", args.xml_dir)
        return 0
    _LOGGER.info("Found %d XML file(s) under %s", len(xml_files), args.xml_dir)

    stats = Stats()
    used_outputs: set[Path] = set()
    manifest_rows: list[dict[str, object]] | None = (
        [] if args.manifest is not None else None
    )
    for xml_path in xml_files:
        process_xml_file(
            xml_path, args.image_dir, args.output_dir, args.anchor,
            args.on_exists, args.dry_run, used_outputs, stats,
            output_layout=args.output_layout,
            filename_format=args.filename_format,
            manifest_rows=manifest_rows,
        )

    if args.manifest is not None and not args.dry_run:
        try:
            _write_manifest(args.manifest, manifest_rows or [])
        except OSError as exc:
            _LOGGER.error("Could not write manifest %s: %s", args.manifest, exc)
            return 1
        _LOGGER.info("Source manifest: %s", args.manifest)

    _LOGGER.info(
        "Done%s. Summary:\n%s",
        " (dry-run)" if args.dry_run else "", stats.render(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
