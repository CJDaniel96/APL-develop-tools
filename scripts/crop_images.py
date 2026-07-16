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
  4. Saves the crop to ``<output-dir>/<same relative path>``, preserving the
     directory structure. If several regions map to one output file, later
     crops get a ``_1``, ``_2`` ... suffix so nothing is overwritten.

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
import logging
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
        elem: The ``<Image>`` element to read from.
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
) -> None:
    """Crops the region described by a single ``<Image>`` node.

    Failures are logged and counted rather than raised, so that one bad node
    does not abort the run.

    Args:
        elem: The ``<Image>`` element to process.
        image_dir: Root directory holding the source image tree.
        output_dir: Root directory the crop is written under, mirroring the
            source tree.
        anchor: Path segment used to reverse-engineer ``PicPath``.
        on_exists: Policy for an output path that already exists on disk:
            ``suffix``, ``skip`` or ``overwrite``.
        dry_run: If True, report the crop without writing it.
        used_outputs: Output paths already produced by this run. Mutated in
            place to reserve the path chosen for this node.
        stats: Counters, updated in place.
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

            target = output_dir.joinpath(*relative)
            final = _unique_output(target, used_outputs, on_exists)
            if final is None:
                _LOGGER.info("Exists, skipping: %s", target)
                stats.skipped_existing += 1
                return
            used_outputs.add(final)

            if dry_run:
                _LOGGER.info("[dry-run] %s %s -> %s", source.name, box, final)
                stats.written += 1
                return

            crop = img.crop(box)
            final.parent.mkdir(parents=True, exist_ok=True)
            crop.save(final)
            _LOGGER.debug("Wrote %s %s -> %s", source.name, box, final)
            stats.written += 1
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
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        _LOGGER.error("Cannot parse %s: %s", xml_path, exc)
        stats.xml_errors += 1
        return
    stats.xml_files += 1

    root = tree.getroot()
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
        help="Root output directory; mirrors the MAP\\... tree.",
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
        A process exit code: 0 on success, 2 if an input directory is
        missing.
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

    xml_files = sorted(args.xml_dir.rglob("*.xml"))
    if not xml_files:
        _LOGGER.warning("No XML files found under %s", args.xml_dir)
        return 0
    _LOGGER.info("Found %d XML file(s) under %s", len(xml_files), args.xml_dir)

    stats = Stats()
    used_outputs: set[Path] = set()
    for xml_path in xml_files:
        process_xml_file(
            xml_path, args.image_dir, args.output_dir, args.anchor,
            args.on_exists, args.dry_run, used_outputs, stats,
        )

    _LOGGER.info(
        "Done%s. Summary:\n%s",
        " (dry-run)" if args.dry_run else "", stats.render(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
