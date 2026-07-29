#!/usr/bin/env python3
"""Run Ultralytics YOLO inference and sort annotated images into OK/NG.

An image is OK only when:

* the configured OK class occurs exactly ``--ok-count`` times;
* every OK bounding-box centre is inside the configured image-centre region;
* no non-OK bounding box is fully contained by an OK bounding box.

Non-OK detections outside every OK bounding box are ignored.  If an image has
no OK detection, any non-OK detection makes it NG.  NG images are saved below
the class-name folder of the qualifying NG detection with the smallest class
id.  Rule failures without such a detection use a diagnostic folder.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

_LOGGER = logging.getLogger("yolo_classify")

_DEFAULT_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)
_DIAGNOSTIC_OK_RULE = "_ok_rule"
_DIAGNOSTIC_NO_DETECTION = "_no_detection"


@dataclass(frozen=True)
class Detection:
    """One YOLO detection in absolute image coordinates."""

    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Decision:
    """Classification result and the information used to save it."""

    is_ok: bool
    reason: str
    folder: str
    qualifying_ng: tuple[Detection, ...] = ()


@dataclass
class Stats:
    """Counters for one CLI run."""

    images_seen: int = 0
    ok: int = 0
    ng: int = 0
    saved: int = 0
    skipped_existing: int = 0
    inference_errors: int = 0
    save_errors: int = 0

    def render(self) -> str:
        return "\n".join(
            f"  {item.name:<20}: {getattr(self, item.name)}"
            for item in fields(self)
        )


def bbox_is_in_image_center(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    tolerance: float,
) -> bool:
    """Return whether the bbox centre falls in the central tolerance region.

    ``tolerance`` is measured independently along each image axis.  For
    example, 0.25 accepts bbox centres in the middle 50% of both the image
    width and height.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        return False
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2 / width
    center_y = (y1 + y2) / 2 / height
    return (
        0.5 - tolerance <= center_x <= 0.5 + tolerance
        and 0.5 - tolerance <= center_y <= 0.5 + tolerance
    )


def bbox_is_contained(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    """Return whether all four edges of ``inner`` are inside ``outer``."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    return ox1 <= ix1 and oy1 <= iy1 and ix2 <= ox2 and iy2 <= oy2


def safe_folder_name(label: str) -> str:
    """Make a model label safe to use as one folder name."""
    cleaned = re.sub(r"[^\w.-]+", "_", label, flags=re.UNICODE)
    cleaned = cleaned.strip(" .")
    return cleaned or "_unnamed_label"


def classify_detections(
    detections: Sequence[Detection],
    ok_class_id: int,
    ok_count: int,
    image_size: tuple[int, int],
    center_tolerance: float,
) -> Decision:
    """Apply the requested AOI OK/NG rules to one image."""
    ok_detections = [
        detection
        for detection in detections
        if detection.class_id == ok_class_id
    ]
    other_detections = [
        detection
        for detection in detections
        if detection.class_id != ok_class_id
    ]

    if ok_detections:
        qualifying_ng = [
            ng_detection
            for ng_detection in other_detections
            if any(
                bbox_is_contained(ng_detection.bbox, ok_detection.bbox)
                for ok_detection in ok_detections
            )
        ]
    else:
        # With no OK object as a reference, every detected non-OK class is NG.
        qualifying_ng = list(other_detections)

    if len(ok_detections) != ok_count:
        reason = (
            f"OK count mismatch: expected {ok_count}, "
            f"detected {len(ok_detections)}"
        )
        return _ng_decision(reason, qualifying_ng, detections)

    off_center = [
        detection
        for detection in ok_detections
        if not bbox_is_in_image_center(
            detection.bbox, image_size, center_tolerance
        )
    ]
    if off_center:
        return _ng_decision(
            f"{len(off_center)} OK bbox centre(s) outside image centre region",
            qualifying_ng,
            detections,
        )

    if qualifying_ng:
        labels = ", ".join(
            f"{d.label}(id={d.class_id})" for d in qualifying_ng
        )
        return _ng_decision(
            f"NG bbox contained by OK bbox: {labels}",
            qualifying_ng,
            detections,
        )

    return Decision(
        is_ok=True,
        reason=(
            f"detected {ok_count} centred OK object(s); "
            "no contained NG bbox"
        ),
        folder="OK",
    )


def _ng_decision(
    reason: str,
    qualifying_ng: Sequence[Detection],
    all_detections: Sequence[Detection],
) -> Decision:
    """Build an NG decision using the lowest qualifying class id."""
    ordered = tuple(
        sorted(qualifying_ng, key=lambda item: (item.class_id, item.label))
    )
    if ordered:
        folder = safe_folder_name(ordered[0].label)
    elif all_detections:
        folder = _DIAGNOSTIC_OK_RULE
    else:
        folder = _DIAGNOSTIC_NO_DETECTION
    return Decision(
        is_ok=False,
        reason=reason,
        folder=folder,
        qualifying_ng=ordered,
    )


def normalize_names(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    """Normalize Ultralytics' dict/list class-name representation."""
    if isinstance(names, Mapping):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


def resolve_ok_class_id(names: Mapping[int, str], ok_label: str) -> int:
    """Resolve an exact class name to its id, raising a helpful error."""
    matches = [
        class_id for class_id, name in names.items() if name == ok_label
    ]
    if not matches:
        available = ", ".join(
            f"{class_id}:{name}" for class_id, name in sorted(names.items())
        )
        raise ValueError(
            f"OK label {ok_label!r} is not in the model. "
            f"Available labels: {available}"
        )
    return min(matches)


def detections_from_result(
    result: Any,
    names: Mapping[int, str],
) -> list[Detection]:
    """Convert an Ultralytics Results object's boxes to plain values."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    class_ids = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    coordinates = boxes.xyxy.detach().cpu().tolist()
    return [
        Detection(
            class_id=int(class_id),
            label=names.get(int(class_id), f"class_{int(class_id)}"),
            confidence=float(confidence),
            bbox=tuple(float(value) for value in bbox),
        )
        for class_id, confidence, bbox
        in zip(class_ids, confidences, coordinates, strict=True)
    ]


def find_images(
    source: Path,
    extensions: tuple[str, ...],
    output_dir: Path,
) -> list[Path]:
    """Find one source image or recursively scan a source directory."""
    if source.is_file():
        return [source] if source.suffix.lower() in extensions else []

    resolved_output = output_dir.resolve()
    images: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.resolve().is_relative_to(resolved_output):
            continue
        images.append(path)
    return images


def unique_output(
    target: Path,
    used: set[Path],
    on_exists: str,
) -> Path | None:
    """Choose a non-colliding output path according to the CLI policy."""
    if target not in used:
        if not target.exists() or on_exists == "overwrite":
            return target
        if on_exists == "skip":
            return None

    index = 1
    while True:
        candidate = target.with_name(
            f"{target.stem}_{index}{target.suffix}"
        )
        if candidate not in used and not candidate.exists():
            return candidate
        index += 1


def save_annotated_result(result: Any, target: Path) -> None:
    """Render YOLO boxes/labels and save the annotated image."""
    plotted_bgr = result.plot()
    plotted_rgb = plotted_bgr[..., ::-1]
    image = Image.fromarray(plotted_rgb)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 0.5:
        raise argparse.ArgumentTypeError("must be between 0.0 and 0.5")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Ultralytics YOLO inference and sort annotated images into "
            "OK and NG folders."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Ultralytics model path or model name, e.g. best.pt.",
    )
    parser.add_argument(
        "-s",
        "--source",
        required=True,
        type=Path,
        help="One image or a directory searched recursively.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Root directory for OK/ and NG/<label>/ results.",
    )
    parser.add_argument(
        "--ok-label",
        required=True,
        help="Exact YOLO class name that represents OK.",
    )
    parser.add_argument(
        "--ok-count",
        required=True,
        type=_positive_int,
        help="Exact number of OK detections required.",
    )
    parser.add_argument(
        "--center-tolerance",
        type=_unit_interval,
        default=0.25,
        metavar="RATIO",
        help=(
            "Allowed bbox-centre offset from image centre per axis; 0.25 "
            "means the middle 50%% of image width and height."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="YOLO NMS IoU threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. cpu, 0, or mps.",
    )
    parser.add_argument(
        "--max-det",
        type=_positive_int,
        default=300,
        help="Maximum detections per image.",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=list(_DEFAULT_EXTENSIONS),
        metavar="EXT",
        help="Image extensions to scan.",
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
        help="Run inference and classification without saving images.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only log warnings and errors.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run inference, classification, and annotated-image saving."""
    args = parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    if args.quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    if not args.source.exists():
        _LOGGER.error("Source does not exist: %s", args.source)
        return 2
    if not 0.0 <= args.conf <= 1.0:
        _LOGGER.error("--conf must be between 0.0 and 1.0")
        return 2
    if not 0.0 <= args.iou <= 1.0:
        _LOGGER.error("--iou must be between 0.0 and 1.0")
        return 2
    if args.imgsz < 1:
        _LOGGER.error("--imgsz must be at least 1")
        return 2

    extensions = tuple(
        item.lower() if item.startswith(".") else f".{item.lower()}"
        for item in args.ext
    )
    images = find_images(args.source, extensions, args.output_dir)
    if not images:
        _LOGGER.warning("No matching images found: %s", args.source)
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        _LOGGER.error(
            "Ultralytics is not installed. Run `uv sync` before this command."
        )
        return 2

    try:
        model = YOLO(args.model)
        names = normalize_names(model.names)
        ok_class_id = resolve_ok_class_id(names, args.ok_label)
    except (OSError, RuntimeError, ValueError) as exc:
        _LOGGER.error("Could not load/validate model: %s", exc)
        return 2

    _LOGGER.info(
        "Found %d image(s); OK class is %s (id=%d), expected count=%d",
        len(images),
        args.ok_label,
        ok_class_id,
        args.ok_count,
    )

    stats = Stats()
    used_outputs: set[Path] = set()
    predict_options: dict[str, Any] = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "verbose": False,
    }
    if args.device is not None:
        predict_options["device"] = args.device

    for source in images:
        stats.images_seen += 1
        try:
            results = model.predict(source=str(source), **predict_options)
            if not results:
                raise RuntimeError("model returned no result")
            result = results[0]
            detections = detections_from_result(result, names)
            height, width = (int(value) for value in result.orig_shape)
            decision = classify_detections(
                detections=detections,
                ok_class_id=ok_class_id,
                ok_count=args.ok_count,
                image_size=(width, height),
                center_tolerance=args.center_tolerance,
            )
        except Exception as exc:  # Keep processing the rest of a production run.
            _LOGGER.warning("Inference failed for %s: %s", source, exc)
            stats.inference_errors += 1
            continue

        if decision.is_ok:
            stats.ok += 1
            target = args.output_dir / "OK" / source.name
            verdict = "OK"
        else:
            stats.ng += 1
            target = args.output_dir / "NG" / decision.folder / source.name
            verdict = f"NG/{decision.folder}"
        _LOGGER.info("%s -> %s (%s)", source, verdict, decision.reason)

        final = unique_output(target, used_outputs, args.on_exists)
        if final is None:
            stats.skipped_existing += 1
            continue
        used_outputs.add(final)

        if args.dry_run:
            continue
        try:
            save_annotated_result(result, final)
            stats.saved += 1
        except (OSError, ValueError) as exc:
            _LOGGER.warning("Could not save %s: %s", final, exc)
            stats.save_errors += 1

    _LOGGER.info(
        "Done%s. Summary:\n%s",
        " (dry-run)" if args.dry_run else "",
        stats.render(),
    )
    return 1 if stats.inference_errors or stats.save_errors else 0


if __name__ == "__main__":
    sys.exit(main())
