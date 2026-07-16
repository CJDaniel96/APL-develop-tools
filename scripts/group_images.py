#!/usr/bin/env python3
"""Groups cropped AOI images into per-component folders by light source.

Recursively scans an image directory (typically the output of
``crop_images.py``) for image files named::

    {ComponentName}_{PadID}_{Light}.jpg

where ``Light`` is ``SolderLight`` or ``UniformLight``. For every image whose
light source matches the requested one this script:

  1. Parses ``ComponentName`` and the light source out of the file name. A
     trailing ``_1``, ``_2`` ... de-duplication suffix (as produced by
     ``crop_images.py``) is tolerated, and ``ComponentName`` may itself
     contain underscores.
  2. Copies the file to::

         <output-dir>/{ComponentName}/{file}          # one light selected
         <output-dir>/{Light}/{ComponentName}/{file}  # --light all

Source images are copied, never moved. Because the same file name can occur
under several boards or dates, a name that collides inside a component folder
gets a ``_1``, ``_2`` ... suffix rather than overwriting.

Run inside the project's uv environment, e.g.::

    uv run scripts/group_images.py \
        --image-dir  ./OUT \
        --output-dir ./GROUPED \
        --light SolderLight
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass, fields
from pathlib import Path

_LOGGER = logging.getLogger("group_images")

# Light sources, in their canonical spelling. File names are matched against
# these case-insensitively; the canonical form is used for output folders.
_LIGHTS = ("SolderLight", "UniformLight")

# Selects every light source instead of just one.
_ALL = "all"

_DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# File name parsing
# --------------------------------------------------------------------------- #
def parse_stem(stem: str) -> tuple[str, str] | None:
    """Splits a ``{Component}_{PadID}_{Light}`` stem into its parts.

    The light token is matched case-insensitively and searched from the right,
    so a de-duplication suffix appended by ``crop_images.py``
    (``C1_1_SolderLight_1``) still resolves. Everything left of the pad ID is
    treated as the component name, which may therefore contain underscores.

    Args:
        stem: The file name without its extension.

    Returns:
        A ``(component_name, light)`` pair with ``light`` in its canonical
        spelling, or None when the stem carries no known light token or has
        no room for both a component name and a pad ID.
    """
    canonical = {light.lower(): light for light in _LIGHTS}
    tokens = stem.split("_")
    for index in range(len(tokens) - 1, -1, -1):
        light = canonical.get(tokens[index].lower())
        if light is None:
            continue
        if index < 2:
            # Nothing left for {Component}_{PadID}.
            return None
        return "_".join(tokens[:index - 1]), light
    return None


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def _unique_output(
    target: Path, used: set[Path], on_exists: str
) -> Path | None:
    """Picks the final output path, honoring the on_exists policy.

    Within a single run two distinct source files never overwrite each other:
    if the path was already produced this run it always gets a numeric
    suffix. For a path that already exists on disk from a *previous* run the
    on_exists policy applies.

    Args:
        target: The desired output path.
        used: Output paths already produced by this run.
        on_exists: Policy for a path that exists on disk from a previous run:
            ``suffix``, ``skip`` or ``overwrite``.

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


def find_images(
    image_dir: Path, extensions: tuple[str, ...], output_dir: Path
) -> list[Path]:
    """Collects the image files to consider, newest-first ordering aside.

    Files living under the output directory are excluded, so that re-running
    with an output directory nested inside the image directory does not
    re-group its own copies.

    Args:
        image_dir: Root directory searched recursively.
        extensions: Accepted file extensions, lower-case and dot-prefixed.
        output_dir: Directory whose contents are excluded from the scan.

    Returns:
        The matching image paths, sorted.
    """
    resolved_output = output_dir.resolve()
    found: list[Path] = []
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.resolve().is_relative_to(resolved_output):
            continue
        found.append(path)
    return found


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    """Counters describing the outcome of a run.

    Attributes:
        images_seen: Image files scanned.
        copied: Files copied (or that would be copied, when dry-running).
        skipped_existing: Files skipped because the output already existed.
        other_light: Files whose light source was not the requested one.
        unparsed: Files whose name does not follow the naming rule.
        copy_errors: Files that could not be copied.
    """

    images_seen: int = 0
    copied: int = 0
    skipped_existing: int = 0
    other_light: int = 0
    unparsed: int = 0
    copy_errors: int = 0

    def render(self) -> str:
        """Returns the counters as an indented, one-per-line summary."""
        return "\n".join(
            f"  {f.name:<18}: {getattr(self, f.name)}" for f in fields(self)
        )


# --------------------------------------------------------------------------- #
# Core processing
# --------------------------------------------------------------------------- #
def process_image(
    source: Path,
    output_dir: Path,
    light_filter: str,
    on_exists: str,
    dry_run: bool,
    used_outputs: set[Path],
    stats: Stats,
) -> None:
    """Copies one image into its component folder, if its light matches.

    Failures are logged and counted rather than raised, so that one bad file
    does not abort the run.

    Args:
        source: The image file to classify.
        output_dir: Root directory the component folders are created under.
        light_filter: The requested light source, or ``all`` to keep every
            light and group by light first.
        on_exists: Policy for an output path that already exists on disk:
            ``suffix``, ``skip`` or ``overwrite``.
        dry_run: If True, report the copy without performing it.
        used_outputs: Output paths already produced by this run. Mutated in
            place to reserve the path chosen for this file.
        stats: Counters, updated in place.
    """
    stats.images_seen += 1

    parsed = parse_stem(source.stem)
    if parsed is None:
        _LOGGER.warning(
            "Name does not match {Component}_{Pad}_{Light}: %s", source
        )
        stats.unparsed += 1
        return
    component, light = parsed

    if light_filter != _ALL and light != light_filter:
        _LOGGER.debug("Light %s not requested, skipping %s", light, source.name)
        stats.other_light += 1
        return

    parent = output_dir / component
    if light_filter == _ALL:
        parent = output_dir / light / component

    final = _unique_output(parent / source.name, used_outputs, on_exists)
    if final is None:
        _LOGGER.info("Exists, skipping: %s", parent / source.name)
        stats.skipped_existing += 1
        return
    used_outputs.add(final)

    if dry_run:
        _LOGGER.info("[dry-run] %s -> %s", source, final)
        stats.copied += 1
        return

    try:
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, final)
    except OSError as exc:
        _LOGGER.warning("Failed to copy %s: %s", source, exc)
        stats.copy_errors += 1
        return
    _LOGGER.debug("Copied %s -> %s", source, final)
    stats.copied += 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _light_arg(value: str) -> str:
    """Normalizes a ``--light`` value to its canonical spelling.

    Args:
        value: The raw argument, matched case-insensitively.

    Returns:
        The canonical light name, or ``all``.

    Raises:
        argparse.ArgumentTypeError: If the value names no known light source.
    """
    lookup = {light.lower(): light for light in _LIGHTS}
    lookup[_ALL] = _ALL
    try:
        return lookup[value.lower()]
    except KeyError:
        choices = ", ".join((*_LIGHTS, _ALL))
        raise argparse.ArgumentTypeError(
            f"invalid light {value!r} (choose from {choices})"
        ) from None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Group cropped AOI images into per-component folders, filtered "
            "by light source."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--image-dir", required=True, type=Path,
        help="Directory searched recursively for cropped images.",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, type=Path,
        help="Root output directory; one sub-folder per component name.",
    )
    parser.add_argument(
        "-l", "--light", required=True, type=_light_arg,
        metavar="{" + ",".join((*_LIGHTS, _ALL)) + "}",
        help="Light source to keep; 'all' groups by light then component.",
    )
    parser.add_argument(
        "--ext", nargs="+", default=list(_DEFAULT_EXTENSIONS),
        metavar="EXT", help="Image extensions to scan for.",
    )
    parser.add_argument(
        "--on-exists", choices=("suffix", "skip", "overwrite"),
        default="suffix",
        help="What to do when an output file already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be copied without writing.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose (DEBUG) logging."
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Only warnings and errors."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Groups every matching image under the given image directory.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code: 0 on success, 2 if the image directory is
        missing.
    """
    args = parse_args(argv)
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    if not args.image_dir.is_dir():
        _LOGGER.error("Image directory does not exist: %s", args.image_dir)
        return 2

    extensions = tuple(
        e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext
    )
    images = find_images(args.image_dir, extensions, args.output_dir)
    if not images:
        _LOGGER.warning("No images found under %s", args.image_dir)
        return 0
    _LOGGER.info("Found %d image(s) under %s", len(images), args.image_dir)

    stats = Stats()
    used_outputs: set[Path] = set()
    for source in images:
        process_image(
            source, args.output_dir, args.light, args.on_exists, args.dry_run,
            used_outputs, stats,
        )

    _LOGGER.info(
        "Done%s. Summary:\n%s",
        " (dry-run)" if args.dry_run else "", stats.render(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
