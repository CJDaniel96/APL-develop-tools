#!/usr/bin/env python3
"""Run Ultralytics YOLO inference and sort review artifacts into OK/NG.

An image is OK only when:

* the configured OK class occurs exactly ``--ok-count`` times (near-identical
  boxes can count as one when only one OK object is required);
* each counted OK object's centre is in the configured image-centre region;
* no non-OK bounding box is fully contained by an OK bounding box.
* enabled OBB angle, centre-distance and pairwise-separation limits pass.

Both regular detection and oriented bounding box (OBB) models are supported.
OBB containment uses the rotated corners, while Pascal VOC annotations use
the axis-aligned enclosing rectangle because VOC cannot retain rotation.

Non-OK detections outside every OK bounding box are ignored.  If an image has
no OK detection, any non-OK detection makes it NG.  NG images are saved below
the class-name folder of the qualifying NG detection with the smallest class
id.  Rule failures without such a detection use a diagnostic folder.

Each final category contains an ``inference`` directory for rendered
detections and an ``original`` directory for source images plus Pascal VOC
XML annotations.  The XML contains every detection returned by the model so
that the image/annotation pairs can be imported into CVAT for review.
OBB runs also save per-image JSON measurements, including all pairwise gaps.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, fields
from itertools import combinations
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
_INFERENCE_DIR = "inference"
_ORIGINAL_DIR = "original"


@dataclass(frozen=True)
class Detection:
    """Absolute image coordinates: enclosing bbox and optional OBB corners."""

    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    polygon: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class ObbRules:
    """Optional limits in degrees and original-image pixels."""

    angle_target_deg: float = 0.0
    angle_tolerance_deg: float | None = None
    max_center_distance_px: float | None = None
    min_edge_distance_px: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.angle_target_deg):
            raise ValueError("--ok-angle-target must be finite")
        tolerance = self.angle_tolerance_deg
        if tolerance is not None and (
            not math.isfinite(tolerance) or not 0 <= tolerance <= 90
        ):
            raise ValueError("--ok-angle-tolerance must be between 0 and 90")
        for option, value in (
            ("--ok-max-center-distance", self.max_center_distance_px),
            ("--ok-min-edge-distance", self.min_edge_distance_px),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{option} must be finite and nonnegative")

    @property
    def enabled(self) -> bool:
        """Return whether any additional NG limit is enabled."""
        return any(value is not None for value in (
            self.angle_tolerance_deg,
            self.max_center_distance_px,
            self.min_edge_distance_px,
        ))


@dataclass(frozen=True)
class OkObjectMeasurement:
    """One counted OK object, indexed from one after duplicate handling."""

    index: int
    confidence: float
    corners_px: tuple[tuple[float, float], ...]
    angle_deg: float
    angle_deviation_deg: float
    center_x_px: float
    center_y_px: float
    offset_x_px: float
    offset_y_px: float
    center_distance_px: float


@dataclass(frozen=True)
class OkPairMeasurement:
    """Shortest separation between two closed OK rectangles."""

    first_index: int
    second_index: int
    edge_distance_px: float


@dataclass(frozen=True)
class ObbMeasurements:
    """Geometry of all counted OK objects, including every pair's gap."""

    image_size: tuple[int, int]
    raw_ok_count: int
    effective_ok_count: int
    rules: ObbRules
    objects: tuple[OkObjectMeasurement, ...]
    pairs: tuple[OkPairMeasurement, ...]
    min_edge_distance_px: float | None


@dataclass(frozen=True)
class Decision:
    """Classification result and the information used to save it."""

    is_ok: bool
    reason: str
    folder: str
    qualifying_ng: tuple[Detection, ...] = ()
    measurements: ObbMeasurements | None = None


@dataclass(frozen=True)
class OutputTargets:
    """Paths for one image's related review artifacts."""

    inference: Path
    original: Path
    annotation: Path
    measurements: Path | None = None

    def all(self) -> tuple[Path, ...]:
        """Return all paths for collision checks and reservations."""
        paths = (self.inference, self.original, self.annotation)
        return paths + ((self.measurements,) if self.measurements else ())


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


def detection_is_contained(inner: Detection, outer: Detection) -> bool:
    """Check full containment using rotated corners when available.

    Args:
        inner: Detection whose entire box must be inside the outer box.
        outer: Detection defining the containing rectangle.

    Returns:
        Whether all inner corners are inside or on the outer boundary.
    """
    if outer.polygon is None:
        return bbox_is_contained(inner.bbox, outer.bbox)

    polygon = outer.polygon
    edges = tuple(zip(polygon, polygon[1:] + polygon[:1]))
    signed_area = sum(
        x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in edges
    )
    if abs(signed_area) <= 1e-12:
        return False
    direction = 1 if signed_area > 0 else -1

    corners = inner.polygon
    if corners is None:
        x1, y1, x2, y2 = inner.bbox
        corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))

    for (x1, y1), (x2, y2) in edges:
        dx, dy = x2 - x1, y2 - y1
        # Allow 0.0001 pixel of roundoff from float32 OBB coordinates.
        tolerance = 1e-4 * math.hypot(dx, dy)
        for x, y in corners:
            cross = dx * (y - y1) - dy * (x - x1)
            if direction * cross < -tolerance:
                return False
    return True


def detection_iou(first: Detection, second: Detection) -> float:
    """Compute intersection over union using the actual detection shapes.

    Args:
        first: Regular or oriented detection in absolute image coordinates.
        second: Detection to compare with the first.

    Returns:
        IoU in [0, 1], or zero if either box has no area.
    """
    if first.polygon is None and second.polygon is None:
        ax1, ay1, ax2, ay2 = first.bbox
        bx1, by1, bx2, by2 = second.bbox
        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        intersection = (
            max(0.0, min(ax2, bx2) - max(ax1, bx1))
            * max(0.0, min(ay2, by2) - max(ay1, by1))
        )
    else:
        # OpenCV and NumPy are included with Ultralytics. Keep regular-box
        # classification independent of these heavier geometry imports.
        import cv2
        import numpy as np

        polygons = []
        for detection in (first, second):
            corners = detection.polygon
            if corners is None:
                x1, y1, x2, y2 = detection.bbox
                corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
            polygons.append(np.asarray(corners, dtype=np.float32))
        first_area = cv2.contourArea(polygons[0])
        second_area = cv2.contourArea(polygons[1])
        if first_area <= 0.0 or second_area <= 0.0:
            return 0.0
        intersection, _ = cv2.intersectConvexConvex(
            polygons[0], polygons[1], handleNested=True,
        )

    if first_area <= 0.0 or second_area <= 0.0:
        return 0.0
    intersection = max(0.0, min(intersection, first_area, second_area))
    union = first_area + second_area - intersection
    return intersection / union


def detection_corners(
    detection: Detection,
) -> tuple[tuple[float, float], ...]:
    """Return ordered OBB vertices or regular rectangle corners."""
    if detection.polygon is not None:
        return detection.polygon
    x1, y1, x2, y2 = detection.bbox
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def obb_angle_degrees(detection: Detection) -> float:
    """Return the long-axis angle clockwise from horizontal in [-90, 90).

    Opposite edges and rotations by 180 degrees describe the same axis.
    Square boxes use the axis closest to horizontal to break the tie.
    """
    if detection.polygon is None:
        raise ValueError("OBB angle measurement requires rotated corners")
    corners = detection.polygon
    candidates = []
    for first, second in zip(corners, corners[1:] + corners[:1]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length = math.hypot(dx, dy)
        angle = (math.degrees(math.atan2(dy, dx)) + 90) % 180 - 90
        candidates.append((length, angle))
    longest = max(length for length, _ in candidates)
    if longest <= 0:
        raise ValueError("Cannot measure an OBB with zero-length edges")
    return min(
        (angle for length, angle in candidates
         if math.isclose(length, longest, rel_tol=1e-6)),
        key=lambda angle: (abs(angle), angle),
    )


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Return the distance to a finite segment, including its endpoints."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    squared_length = dx * dx + dy * dy
    if squared_length == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / squared_length
    projection = max(0.0, min(1.0, projection))
    return math.hypot(
        point[0] - start[0] - projection * dx,
        point[1] - start[1] - projection * dy,
    )


def detection_edge_distance(first: Detection, second: Detection) -> float:
    """Return exact rectangle separation in pixels; overlap/touch is zero.

    Separating-axis projections detect intersection and containment.  For
    disjoint convex rectangles, the nearest points lie on a vertex and an
    opposing segment (including its endpoints).
    """
    first_corners, second_corners = map(detection_corners, (first, second))
    first_edges = tuple(zip(
        first_corners, first_corners[1:] + first_corners[:1]
    ))
    second_edges = tuple(zip(
        second_corners, second_corners[1:] + second_corners[:1]
    ))
    axes = [(1.0, 0.0), (0.0, 1.0)]
    axes.extend(
        (-(end[1] - start[1]), end[0] - start[0])
        for start, end in first_edges + second_edges
    )
    for nx, ny in axes:
        first_projection = [nx * x + ny * y for x, y in first_corners]
        second_projection = [nx * x + ny * y for x, y in second_corners]
        if (
            max(first_projection) < min(second_projection)
            or max(second_projection) < min(first_projection)
        ):
            break
    else:
        return 0.0

    return min(
        _point_segment_distance(point, start, end)
        for points, edges in (
            (first_corners, second_edges), (second_corners, first_edges)
        )
        for point in points
        for start, end in edges
    )


def measure_ok_objects(
    detections: Sequence[Detection],
    image_size: tuple[int, int],
    raw_ok_count: int,
    rules: ObbRules,
) -> ObbMeasurements:
    """Measure counted OK OBBs using original-image pixel coordinates."""
    width, height = image_size
    objects = []
    for index, detection in enumerate(detections, 1):
        corners = detection_corners(detection)
        center_x = sum(x for x, _ in corners) / len(corners)
        center_y = sum(y for _, y in corners) / len(corners)
        dx, dy = center_x - width / 2, center_y - height / 2
        angle = obb_angle_degrees(detection)
        deviation = abs((angle - rules.angle_target_deg + 90) % 180 - 90)
        objects.append(OkObjectMeasurement(
            index=index,
            confidence=detection.confidence,
            corners_px=corners,
            angle_deg=angle,
            angle_deviation_deg=deviation,
            center_x_px=center_x,
            center_y_px=center_y,
            offset_x_px=dx,
            offset_y_px=dy,
            center_distance_px=math.hypot(dx, dy),
        ))
    pairs = tuple(
        OkPairMeasurement(i, j, detection_edge_distance(first, second))
        for (i, first), (j, second)
        in combinations(enumerate(detections, 1), 2)
    )
    return ObbMeasurements(
        image_size=image_size,
        raw_ok_count=raw_ok_count,
        effective_ok_count=len(detections),
        rules=rules,
        objects=tuple(objects),
        pairs=pairs,
        min_edge_distance_px=min(
            (pair.edge_distance_px for pair in pairs), default=None
        ),
    )


def _measurement_summary(measurements: ObbMeasurements) -> str:
    """Format measured values for the image log and dry-run output."""
    parts = [
        f"OK#{item.index}: angle={item.angle_deg:.2f}deg, "
        f"centre_distance={item.center_distance_px:.2f}px "
        f"(dx={item.offset_x_px:.2f}, dy={item.offset_y_px:.2f})"
        for item in measurements.objects
    ]
    if measurements.min_edge_distance_px is not None:
        parts.append(
            f"minimum edge distance={measurements.min_edge_distance_px:.2f}px"
        )
    return "; OBB measurements: " + ("; ".join(parts) or "no OK objects")


def _obb_rule_failures(measurements: ObbMeasurements) -> list[str]:
    """Return every enabled geometry-rule failure, with measured values."""
    rules = measurements.rules
    failures = []
    for item in measurements.objects:
        if (
            rules.angle_tolerance_deg is not None
            and item.angle_deviation_deg > rules.angle_tolerance_deg + 1e-6
        ):
            failures.append(
                f"OK#{item.index} angle {item.angle_deg:.2f}deg deviates "
                f"{item.angle_deviation_deg:.2f}deg from target "
                f"{rules.angle_target_deg:.2f}deg "
                f"(allowed {rules.angle_tolerance_deg:.2f}deg)"
            )
        if (
            rules.max_center_distance_px is not None
            and item.center_distance_px > rules.max_center_distance_px + 1e-6
        ):
            failures.append(
                f"OK#{item.index} centre distance "
                f"{item.center_distance_px:.2f}px exceeds "
                f"{rules.max_center_distance_px:.2f}px"
            )
    if rules.min_edge_distance_px is not None:
        for pair in measurements.pairs:
            if pair.edge_distance_px + 1e-6 < rules.min_edge_distance_px:
                failures.append(
                    f"OK#{pair.first_index}/OK#{pair.second_index} edge "
                    f"distance {pair.edge_distance_px:.2f}px below "
                    f"{rules.min_edge_distance_px:.2f}px"
                )
    return failures


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
    ok_overlap_iou: float = 0.8,
    obb_rules: ObbRules | None = None,
) -> Decision:
    """Apply AOI rules, optionally counting overlapping OK boxes as one.

    When exactly one OK object is required, all OK boxes must overlap each
    other by at least ``ok_overlap_iou`` to count as one.  Its centre is taken
    from the highest-confidence box.  NG containment always uses every
    original OK box, and the input detections are left intact for review.
    Set ``ok_overlap_iou`` to zero to require the original exact count.
    OBB measurements are automatic; optional ``obb_rules`` enable additional
    angle, radial centre-offset and edge-separation limits.
    """
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
    raw_ok_count = len(ok_detections)

    if ok_detections:
        qualifying_ng = [
            ng_detection
            for ng_detection in other_detections
            if any(
                detection_is_contained(ng_detection, ok_detection)
                for ok_detection in ok_detections
            )
        ]
    else:
        # With no OK object as a reference, every detected non-OK class is NG.
        qualifying_ng = list(other_detections)

    overlap_note = ""
    if ok_count == 1 and len(ok_detections) > 1 and ok_overlap_iou > 0:
        minimum_iou = min(
            detection_iou(first, second)
            for first, second in combinations(ok_detections, 2)
        )
        if minimum_iou >= ok_overlap_iou:
            overlap_note = (
                f"; counted {len(ok_detections)} overlapping OK boxes as 1 "
                f"(min IoU={minimum_iou:.3f}, threshold={ok_overlap_iou:.3f})"
            )
            ok_detections = [
                max(ok_detections, key=lambda item: item.confidence)
            ]
        else:
            overlap_note = (
                f"; min OK IoU={minimum_iou:.3f} "
                f"below overlap threshold={ok_overlap_iou:.3f}"
            )

    measurements = None
    if obb_rules is not None or any(d.polygon is not None for d in detections):
        measurements = measure_ok_objects(
            ok_detections, image_size, raw_ok_count, obb_rules or ObbRules()
        )
        overlap_note += _measurement_summary(measurements)

    if len(ok_detections) != ok_count:
        reason = (
            f"OK count mismatch: expected {ok_count}, "
            f"detected {len(ok_detections)}{overlap_note}"
        )
        return _ng_decision(reason, qualifying_ng, detections, measurements)

    off_center = [
        detection
        for detection in ok_detections
        if not bbox_is_in_image_center(
            detection.bbox, image_size, center_tolerance
        )
    ]
    if off_center:
        return _ng_decision(
            f"{len(off_center)} OK bbox centre(s) outside image centre region"
            f"{overlap_note}",
            qualifying_ng,
            detections,
            measurements,
        )

    if measurements is not None:
        failures = _obb_rule_failures(measurements)
        if failures:
            return _ng_decision(
                "; ".join(failures) + overlap_note,
                qualifying_ng,
                detections,
                measurements,
            )

    if qualifying_ng:
        labels = ", ".join(
            f"{d.label}(id={d.class_id})" for d in qualifying_ng
        )
        return _ng_decision(
            f"NG bbox contained by OK bbox: {labels}{overlap_note}",
            qualifying_ng,
            detections,
            measurements,
        )

    return Decision(
        is_ok=True,
        reason=(
            f"detected {ok_count} centred OK object(s); "
            f"no contained NG bbox{overlap_note}"
        ),
        folder="OK",
        measurements=measurements,
    )


def _ng_decision(
    reason: str,
    qualifying_ng: Sequence[Detection],
    all_detections: Sequence[Detection],
    measurements: ObbMeasurements | None = None,
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
        measurements=measurements,
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
    """Convert regular or OBB results to absolute-coordinate detections."""
    obb = getattr(result, "obb", None)
    boxes = obb if obb is not None else result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    boxes = boxes.cpu()
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()
    coordinates = boxes.xyxy.tolist()
    polygons = (
        boxes.xyxyxyxy.tolist() if obb is not None else [None] * len(boxes)
    )
    return [
        Detection(
            class_id=int(class_id),
            label=names.get(int(class_id), f"class_{int(class_id)}"),
            confidence=float(confidence),
            bbox=tuple(float(value) for value in bbox),
            polygon=(
                tuple((float(x), float(y)) for x, y in polygon)
                if polygon is not None else None
            ),
        )
        for class_id, confidence, bbox, polygon
        in zip(class_ids, confidences, coordinates, polygons, strict=True)
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


def output_targets(
    category_dir: Path,
    filename: str,
    include_measurements: bool = False,
) -> OutputTargets:
    """Build the related output paths for one final category."""
    return OutputTargets(
        inference=category_dir / _INFERENCE_DIR / filename,
        original=category_dir / _ORIGINAL_DIR / filename,
        annotation=category_dir / _ORIGINAL_DIR / Path(filename).with_suffix(
            ".xml"
        ).name,
        measurements=(
            category_dir / "measurements" / Path(filename).with_suffix(
                ".json"
            ).name if include_measurements else None
        ),
    )


def unique_output_targets(
    category_dir: Path,
    filename: str,
    used: set[Path],
    on_exists: str,
    include_measurements: bool = False,
) -> OutputTargets | None:
    """Choose synchronized, non-colliding artifact paths."""
    targets = output_targets(category_dir, filename, include_measurements)
    paths = targets.all()
    if not any(path in used for path in paths):
        if not any(path.exists() for path in paths) or on_exists == "overwrite":
            return targets
        if on_exists == "skip":
            return None

    index = 1
    while True:
        source_name = Path(filename)
        candidate_name = (
            f"{source_name.stem}_{index}{source_name.suffix}"
        )
        candidate = output_targets(
            category_dir, candidate_name, include_measurements
        )
        candidate_paths = candidate.all()
        if not any(path in used for path in candidate_paths) and not any(
            path.exists() for path in candidate_paths
        ):
            return candidate
        index += 1


def save_annotated_result(result: Any, target: Path) -> None:
    """Render YOLO boxes/labels and save the annotated image."""
    plotted_bgr = result.plot()
    plotted_rgb = plotted_bgr[..., ::-1]
    image = Image.fromarray(plotted_rgb)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)


def _voc_coordinate_pair(
    lower: float,
    upper: float,
    limit: int,
) -> tuple[int, int]:
    """Convert one floating-point bbox axis to clamped VOC coordinates."""
    low = max(0, min(limit, math.floor(min(lower, upper))))
    high = max(0, min(limit, math.ceil(max(lower, upper))))
    return low, high


def write_pascal_voc(
    target: Path,
    image_path: Path,
    image_size: tuple[int, int],
    depth: int,
    detections: Sequence[Detection],
) -> None:
    """Write VOC boxes, using axis-aligned enclosing rectangles for OBB."""
    width, height = image_size
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = image_path.parent.name
    ET.SubElement(annotation, "filename").text = image_path.name

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "YOLO inference"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(annotation, "segmented").text = "0"

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        xmin, xmax = _voc_coordinate_pair(x1, x2, width)
        ymin, ymax = _voc_coordinate_pair(y1, y2, height)
        is_truncated = (
            min(x1, x2) < 0
            or min(y1, y2) < 0
            or max(x1, x2) > width
            or max(y1, y2) > height
        )

        item = ET.SubElement(annotation, "object")
        ET.SubElement(item, "name").text = detection.label
        ET.SubElement(item, "pose").text = "Unspecified"
        ET.SubElement(item, "truncated").text = "1" if is_truncated else "0"
        ET.SubElement(item, "difficult").text = "0"
        box = ET.SubElement(item, "bndbox")
        ET.SubElement(box, "xmin").text = str(xmin)
        ET.SubElement(box, "ymin").text = str(ymin)
        ET.SubElement(box, "xmax").text = str(xmax)
        ET.SubElement(box, "ymax").text = str(ymax)

    ET.indent(annotation, space="  ")
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(annotation).write(
        target,
        encoding="utf-8",
        xml_declaration=True,
    )


def save_review_artifacts(
    result: Any,
    source: Path,
    targets: OutputTargets,
    detections: Sequence[Detection],
    image_size: tuple[int, int],
    decision: Decision | None = None,
) -> None:
    """Save review images, VOC XML and optional OBB measurements JSON."""
    if targets.measurements is not None and (
        decision is None or decision.measurements is None
    ):
        raise ValueError("OBB measurement output requires a measured decision")
    save_annotated_result(result, targets.inference)
    targets.original.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, targets.original)
    with Image.open(source) as image:
        depth = len(image.getbands())
    write_pascal_voc(
        target=targets.annotation,
        image_path=targets.original,
        image_size=image_size,
        depth=depth,
        detections=detections,
    )
    if targets.measurements is not None and decision is not None:
        payload = {
            "image": targets.original.name,
            "verdict": "OK" if decision.is_ok else "NG",
            "reason": decision.reason,
            "angle_convention": (
                "long axis, clockwise from image horizontal, [-90, 90) deg"
            ),
            "distance_unit": "original-image pixels",
            "measurements": asdict(decision.measurements),
        }
        targets.measurements.parent.mkdir(parents=True, exist_ok=True)
        targets.measurements.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )


def add_obb_arguments(parser: Any) -> None:
    """Add shared optional OBB limits to a parser or argument group."""
    parser.add_argument(
        "--ok-angle-target", type=float, default=0.0, metavar="DEG",
        help="Expected OK OBB long-axis angle from horizontal, in degrees.",
    )
    parser.add_argument(
        "--ok-angle-tolerance", type=float, default=None, metavar="DEG",
        help="Maximum angle deviation modulo 180 degrees (0-90); OBB only.",
    )
    parser.add_argument(
        "--ok-max-center-distance", type=float, default=None, metavar="PX",
        help="Maximum OK OBB centre-to-image-centre distance in pixels.",
    )
    parser.add_argument(
        "--ok-min-edge-distance", type=float, default=None, metavar="PX",
        help="Minimum separation of any two OK OBBs in pixels; closer is NG.",
    )


def obb_rules_from_args(args: argparse.Namespace) -> ObbRules:
    """Build and validate geometry limits shared by the two YOLO CLIs."""
    return ObbRules(
        angle_target_deg=args.ok_angle_target,
        angle_tolerance_deg=args.ok_angle_tolerance,
        max_center_distance_px=args.ok_max_center_distance,
        min_edge_distance_px=args.ok_min_edge_distance,
    )


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
            "Run Ultralytics YOLO detection or OBB inference and sort "
            "review artifacts into OK and NG folders."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Ultralytics detection or OBB model path/name, e.g. best.pt.",
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
        help=(
            "Root for OK/ and NG/<label>/ review artifacts. Each category "
            "gets inference/ and original/; OBB also gets measurements/."
        ),
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
        help="Exact number of OK objects required.",
    )
    parser.add_argument(
        "--ok-overlap-iou",
        type=float,
        default=0.8,
        metavar="RATIO",
        help=(
            "When --ok-count is 1, count multiple OK boxes as one only if "
            "every pair has IoU >= this threshold (0-1); 0 disables it."
        ),
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
    add_obb_arguments(parser)
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
        help="What to do when any related output artifact already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference and classification without saving artifacts.",
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
    """Run inference, classification, and review-artifact saving."""
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
    if not 0.0 <= args.ok_overlap_iou <= 1.0:
        _LOGGER.error("--ok-overlap-iou must be between 0.0 and 1.0")
        return 2
    if args.imgsz < 1:
        _LOGGER.error("--imgsz must be at least 1")
        return 2
    try:
        obb_rules = obb_rules_from_args(args)
    except ValueError as exc:
        _LOGGER.error("%s", exc)
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
        if obb_rules.enabled and model.task != "obb":
            raise ValueError("OBB geometry limits require an OBB model")
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
                ok_overlap_iou=args.ok_overlap_iou,
                obb_rules=(
                    obb_rules if getattr(result, "obb", None) is not None
                    else None
                ),
            )
        except Exception as exc:  # Keep processing the rest of a production run.
            _LOGGER.warning("Inference failed for %s: %s", source, exc)
            stats.inference_errors += 1
            continue

        if decision.is_ok:
            stats.ok += 1
            category_dir = args.output_dir / "OK"
            verdict = "OK"
        else:
            stats.ng += 1
            category_dir = args.output_dir / "NG" / decision.folder
            verdict = f"NG/{decision.folder}"
        _LOGGER.info("%s -> %s (%s)", source, verdict, decision.reason)

        targets = unique_output_targets(
            category_dir,
            source.name,
            used_outputs,
            args.on_exists,
            include_measurements=decision.measurements is not None,
        )
        if targets is None:
            stats.skipped_existing += 1
            continue
        used_outputs.update(targets.all())

        if args.dry_run:
            continue
        try:
            save_review_artifacts(
                result=result,
                source=source,
                targets=targets,
                detections=detections,
                image_size=(width, height),
                decision=decision,
            )
            stats.saved += 1
        except (OSError, ValueError) as exc:
            _LOGGER.warning(
                "Could not save artifacts for %s: %s",
                source,
                exc,
            )
            stats.save_errors += 1

    _LOGGER.info(
        "Done%s. Summary:\n%s",
        " (dry-run)" if args.dry_run else "",
        stats.render(),
    )
    return 1 if stats.inference_errors or stats.save_errors else 0


if __name__ == "__main__":
    sys.exit(main())
