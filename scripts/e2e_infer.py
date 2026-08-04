#!/usr/bin/env python3
"""Runs the whole APL AOI inference chain and reports every decision.

The pipeline has four stages:

1. **Orientation** - one image or a directory is scanned; portrait images are
   rotated to landscape so every downstream model sees the same layout.
2. **YOLO gate** - the trained detector runs on the oriented image and the
   OK/NG rules of :mod:`scripts.yolo_classify` are applied.  NG images are
   written to ``<output-dir>/NG/<label>/`` and leave the pipeline; OK images
   continue.
3. **Class routing** - ``anomaly_dino`` and ``patchcore`` need to know which
   component class an image belongs to, so the trained HOAM/HOAMV2 KNN index
   assigns one class per OK image.  ``dinomaly`` skips this stage because a
   single model covers every class.
4. **Anomaly inference** - the anomalib checkpoint for the routed class (or the
   single ``dinomaly`` checkpoint) scores every OK image.  Image, heat-map
   result and score are written below ``<output-dir>/anomaly/``.

Every image keeps one trace row through all four stages, and the run finishes
by writing ``report.json`` and ``report.csv`` so a validation run can be
reviewed image by image.

Examples::

    # dinomaly: no class routing, one shared checkpoint
    uv run scripts/e2e_infer.py \
        --input ./images --output-dir ./out \
        --yolo-model ./models/yolo/best.pt --ok-label component --ok-count 1 \
        --anomaly-model dinomaly --anomaly-model-dir ./models/dinomaly

    # patchcore: HOAM routes each image to ./models/patchcore/<class>/*.ckpt
    uv run scripts/e2e_infer.py \
        --input ./images --output-dir ./out \
        --yolo-model ./models/yolo/best.pt --ok-label component --ok-count 1 \
        --anomaly-model patchcore --anomaly-model-dir ./models/patchcore \
        --hoam-root ../HOAM --hoam-model-path ./models/hoam/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, UnidentifiedImageError

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow both `uv run scripts/e2e_infer.py` and `import scripts.e2e_infer`.
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.rotate_images import (  # noqa: E402
    _save_rotated as save_rotated,
    find_images,
    orientation_for_size,
)
from scripts.yolo_classify import (  # noqa: E402
    classify_detections,
    detections_from_result,
    normalize_names,
    resolve_ok_class_id,
    safe_folder_name,
    save_annotated_result,
    unique_output,
)

_LOGGER = logging.getLogger("e2e_infer")

_DEFAULT_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

# User-facing anomaly model name -> class exported by ``anomalib.models``.
_ANOMALY_MODELS: dict[str, str] = {
    "patchcore": "Patchcore",
    "anomaly_dino": "AnomalyDINO",
    "dinomaly": "Dinomaly",
}
_ANOMALY_ALIASES: dict[str, str] = {
    "patch_core": "patchcore",
    "patch-core": "patchcore",
    "anomalydino": "anomaly_dino",
    "anomaly-dino": "anomaly_dino",
    "dino": "anomaly_dino",
}
# Only these models need the HOAM class routing stage.
_NEEDS_CLASS_ROUTING = frozenset({"patchcore", "anomaly_dino"})
# Folder used below ``<output-dir>/anomaly/`` when no class routing happened.
_UNROUTED_GROUP = "_all"


@dataclass(eq=False)
class ImageTrace:
    """One input image and every decision taken about it.

    Instances are compared by identity so a trace can be used as a dict key
    while its fields are still being filled in.
    """

    source: str
    relative: str
    orientation: str = ""
    rotated: bool = False
    oriented: str = ""
    stage: str = "input"
    yolo_verdict: str = ""
    yolo_reason: str = ""
    yolo_output: str = ""
    hoam_class: str = ""
    anomaly_checkpoint: str = ""
    anomaly_score: float | None = None
    anomaly_threshold: float | None = None
    anomaly_verdict: str = ""
    anomaly_output: str = ""
    anomaly_result_image: str = ""
    error: str = ""

    def fail(self, stage: str, message: str) -> None:
        """Records an error and stops the trace at ``stage``."""
        self.stage = f"{stage}_error"
        self.error = message


@dataclass
class Stats:
    """Counters for one pipeline run."""

    images_seen: int = 0
    rotated: int = 0
    orient_errors: int = 0
    yolo_ok: int = 0
    yolo_ng: int = 0
    yolo_errors: int = 0
    routed: int = 0
    routing_errors: int = 0
    anomaly_ok: int = 0
    anomaly_ng: int = 0
    anomaly_errors: int = 0

    def render(self) -> str:
        """Returns the counters as an indented summary."""
        return "\n".join(
            f"  {item.name:<18}: {getattr(self, item.name)}"
            for item in fields(self)
        )


@dataclass(frozen=True)
class HoamSettings:
    """Everything needed to run the HOAM KNN routing stage."""

    model_path: Path
    structure: str
    backbone: str | None
    embedding_size: int
    image_size: int
    index_path: Path
    dataset_pkl: Path
    mean_std_file: Path | None
    k: int
    batch_size: int
    device: str | None


# --------------------------------------------------------------------------- #
# Stage 1: orientation
# --------------------------------------------------------------------------- #
def orient_images(
    pairs: Sequence[tuple[Path, Path]],
    work_dir: Path,
    target: str,
    clockwise: bool,
    dry_run: bool,
    stats: Stats,
) -> list[ImageTrace]:
    """Rotates every off-orientation image and returns one trace per image.

    Images that already have the wanted orientation are not copied; their trace
    points at the original file. Rotated images are written below ``work_dir``
    with the input's relative structure preserved.

    Args:
        pairs: ``(source, relative_path)`` pairs from ``find_images``.
        work_dir: Directory that receives rotated copies.
        target: ``landscape`` or ``portrait``.
        clockwise: Whether required rotations turn clockwise.
        dry_run: If True, report rotations without writing files.
        stats: Counters updated in place.

    Returns:
        One :class:`ImageTrace` per input image, in input order.
    """
    traces: list[ImageTrace] = []
    for source, relative in pairs:
        trace = ImageTrace(source=str(source), relative=str(relative))
        traces.append(trace)
        stats.images_seen += 1
        try:
            with Image.open(source) as image:
                image.load()
                current = orientation_for_size(image.width, image.height)
                trace.orientation = current
                if current == "square" or current == target:
                    trace.oriented = str(source)
                    trace.stage = "oriented"
                    continue

                destination = work_dir / relative
                trace.rotated = True
                trace.oriented = str(destination)
                trace.stage = "oriented"
                stats.rotated += 1
                if dry_run:
                    _LOGGER.info("[dry-run] rotate %s -> %s", source, destination)
                    continue
                save_rotated(image, destination, clockwise)
                _LOGGER.debug("Rotated %s -> %s", source, destination)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            _LOGGER.warning("Could not read/rotate %s: %s", source, exc)
            trace.fail("orient", str(exc))
            stats.orient_errors += 1
    return traces


# --------------------------------------------------------------------------- #
# Stage 2: YOLO gate
# --------------------------------------------------------------------------- #
def run_yolo_stage(
    traces: Sequence[ImageTrace],
    args: argparse.Namespace,
    stats: Stats,
) -> list[ImageTrace]:
    """Runs YOLO on every oriented image and files the NG ones.

    Args:
        traces: Traces produced by :func:`orient_images`.
        args: Parsed CLI arguments.
        stats: Counters updated in place.

    Returns:
        The traces judged OK, in input order.
    """
    from ultralytics import YOLO

    model = YOLO(args.yolo_model)
    names = normalize_names(model.names)
    ok_class_id = resolve_ok_class_id(names, args.ok_label)
    _LOGGER.info(
        "YOLO OK class is %s (id=%d), expected count=%d",
        args.ok_label,
        ok_class_id,
        args.ok_count,
    )

    predict_options: dict[str, Any] = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "verbose": False,
    }
    if args.yolo_device is not None:
        predict_options["device"] = args.yolo_device

    used_outputs: set[Path] = set()
    passed: list[ImageTrace] = []
    for trace in traces:
        if trace.stage != "oriented":
            continue
        source = Path(trace.source)
        oriented = Path(trace.oriented)
        if args.dry_run and trace.rotated and not oriented.exists():
            # The rotated file was never written, so score the original.
            oriented = source

        try:
            results = model.predict(source=str(oriented), **predict_options)
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
            _LOGGER.warning("YOLO inference failed for %s: %s", oriented, exc)
            trace.fail("yolo", str(exc))
            stats.yolo_errors += 1
            continue

        trace.yolo_reason = decision.reason
        if decision.is_ok:
            trace.yolo_verdict = "OK"
            trace.stage = "yolo_ok"
            stats.yolo_ok += 1
            passed.append(trace)
            _LOGGER.info("%s -> YOLO OK (%s)", source, decision.reason)
            if args.save_ok_annotated:
                target = args.output_dir / "OK" / source.name
                saved = _save_yolo_image(
                    result, target, used_outputs, args.on_exists, args.dry_run
                )
                if saved is not None:
                    trace.yolo_output = str(saved)
            continue

        trace.yolo_verdict = "NG"
        trace.stage = "yolo_ng"
        stats.yolo_ng += 1
        _LOGGER.info(
            "%s -> YOLO NG/%s (%s)", source, decision.folder, decision.reason
        )
        target = args.output_dir / "NG" / decision.folder / source.name
        saved = _save_yolo_image(
            result, target, used_outputs, args.on_exists, args.dry_run
        )
        if saved is not None:
            trace.yolo_output = str(saved)
    return passed


def _save_yolo_image(
    result: Any,
    target: Path,
    used_outputs: set[Path],
    on_exists: str,
    dry_run: bool,
) -> Path | None:
    """Saves one annotated YOLO image, honouring the collision policy."""
    final = unique_output(target, used_outputs, on_exists)
    if final is None:
        _LOGGER.debug("Exists, skipping: %s", target)
        return None
    used_outputs.add(final)
    if dry_run:
        _LOGGER.info("[dry-run] write %s", final)
        return final
    try:
        save_annotated_result(result, final)
    except (OSError, ValueError) as exc:
        _LOGGER.warning("Could not save %s: %s", final, exc)
        return None
    return final


# --------------------------------------------------------------------------- #
# Stage 3: HOAM class routing
# --------------------------------------------------------------------------- #
def prepare_hoam_imports(hoam_root: Path | None) -> None:
    """Makes the ``hoam`` package importable.

    Args:
        hoam_root: HOAM repository root (or its ``src`` directory).  When None,
            ``hoam`` must already be installed in the environment.

    Raises:
        FileNotFoundError: If ``hoam_root`` contains no ``hoam`` package.
    """
    if hoam_root is None:
        return
    for candidate in (hoam_root / "src", hoam_root):
        if (candidate / "hoam").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise FileNotFoundError(
        f"No 'hoam' package under {hoam_root} or {hoam_root / 'src'}"
    )


def resolve_hoam_settings(args: argparse.Namespace) -> HoamSettings:
    """Fills omitted HOAM options from ``config_used.yaml`` and the model dir.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A fully resolved :class:`HoamSettings`.

    Raises:
        FileNotFoundError: If a required KNN artefact cannot be located.
    """
    model_path = args.hoam_model_path
    model_dir = model_path.parent if model_path.is_file() else model_path

    structure = args.hoam_structure
    backbone = args.hoam_backbone
    embedding_size = args.hoam_embedding_size
    image_size = args.hoam_image_size

    config_path = args.hoam_config or (model_dir / "config_used.yaml")
    if config_path.exists():
        config = _load_hoam_config(config_path)
        if config is not None:
            structure = structure or config.get("structure")
            backbone = backbone or config.get("backbone")
            embedding_size = embedding_size or config.get("embedding_size")
            image_size = image_size or config.get("image_size")
            _LOGGER.debug("Applied HOAM config defaults from %s", config_path)

    index_path = args.hoam_index or (model_dir / "knn.index")
    dataset_pkl = args.hoam_dataset_pkl or (model_dir / "dataset.pkl")
    mean_std_file = args.hoam_mean_std or (model_dir / "mean_std.json")
    for label, path in (("KNN index", index_path), ("dataset pickle", dataset_pkl)):
        if not path.exists():
            raise FileNotFoundError(
                f"HOAM {label} not found: {path}. Build it with "
                "`hoam build-knn` or pass the matching option."
            )
    if not mean_std_file.exists():
        _LOGGER.warning(
            "No mean_std.json at %s; falling back to ImageNet statistics, "
            "which must match how the model was trained.",
            mean_std_file,
        )
        mean_std_file = None

    return HoamSettings(
        model_path=model_path,
        structure=structure or "HOAMV2",
        backbone=backbone,
        embedding_size=embedding_size or 128,
        image_size=image_size or 224,
        index_path=index_path,
        dataset_pkl=dataset_pkl,
        mean_std_file=mean_std_file,
        k=args.hoam_k,
        batch_size=args.hoam_batch_size,
        device=args.hoam_device,
    )


def _load_hoam_config(path: Path) -> dict[str, Any] | None:
    """Reads model/data defaults out of a HOAM ``config_used.yaml``."""
    try:
        from omegaconf import OmegaConf
    except ImportError:
        _LOGGER.debug("omegaconf missing; ignoring %s", path)
        return None
    try:
        config = OmegaConf.load(path)
        return {
            "structure": config.model.structure,
            "backbone": config.model.backbone,
            "embedding_size": config.model.embedding_size,
            "image_size": config.data.image_size,
        }
    except Exception as exc:  # A malformed config must not stop the run.
        _LOGGER.warning("Could not read %s: %s", path, exc)
        return None


def route_with_hoam(
    traces: Sequence[ImageTrace],
    settings: HoamSettings,
    stats: Stats,
) -> None:
    """Assigns a component class to every OK image with the HOAM KNN index.

    Each trace's ``hoam_class`` is set in place.  Neighbours are retrieved with
    ``k = settings.k`` and the class is the majority vote, ties broken by the
    nearest neighbour.

    Args:
        traces: Traces that passed the YOLO gate.
        settings: Resolved HOAM options.
        stats: Counters updated in place.
    """
    import joblib
    import torch
    from pytorch_metric_learning.distances import CosineSimilarity
    from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder

    from hoam.data.statistics import DataStatistics
    from hoam.data.transforms import DEFAULT_MEAN, DEFAULT_STD, build_transforms
    from hoam.utils import load_model

    device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if settings.mean_std_file is not None:
        mean, std = DataStatistics.load_mean_std(settings.mean_std_file)
    else:
        mean, std = DEFAULT_MEAN, DEFAULT_STD

    model = load_model(
        settings.structure,
        str(settings.model_path),
        settings.embedding_size,
        device=device,
        backbone_name=settings.backbone,
    )
    inference_model = InferenceModel(
        model,
        match_finder=MatchFinder(distance=CosineSimilarity(), threshold=0.5),
        data_device=device,
    )
    inference_model.load_knn_func(str(settings.index_path))

    dataset = joblib.load(str(settings.dataset_pkl))
    classes = list(dataset.classes)
    transform = build_transforms("test", settings.image_size, mean, std)
    _LOGGER.info(
        "HOAM routing with %s (%d reference classes, k=%d) on %s",
        settings.structure,
        len(classes),
        settings.k,
        device,
    )

    pending = [trace for trace in traces if trace.stage == "yolo_ok"]
    for chunk in _chunked(pending, settings.batch_size):
        tensors = []
        batch: list[ImageTrace] = []
        for trace in chunk:
            try:
                with Image.open(_scored_path(trace)) as image:
                    tensors.append(transform(image.convert("RGB")))
                batch.append(trace)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                _LOGGER.warning("Could not read %s: %s", trace.oriented, exc)
                trace.fail("routing", str(exc))
                stats.routing_errors += 1
        if not batch:
            continue

        try:
            stacked = torch.stack(tensors).to(device)
            _, indices = inference_model.get_nearest_neighbors(stacked, settings.k)
            neighbours = indices.detach().cpu().tolist()
        except Exception as exc:  # One bad batch must not kill the run.
            _LOGGER.warning("HOAM KNN failed for %d image(s): %s", len(batch), exc)
            for trace in batch:
                trace.fail("routing", str(exc))
                stats.routing_errors += 1
            continue

        for trace, row in zip(batch, neighbours, strict=True):
            class_ids = [_dataset_class_id(dataset, int(index)) for index in row]
            label = classes[_majority(class_ids)]
            trace.hoam_class = label
            trace.stage = "routed"
            stats.routed += 1
            _LOGGER.info("%s -> class %s", trace.source, label)


def _dataset_class_id(dataset: Any, index: int) -> int:
    """Returns the class id of reference sample ``index``."""
    if hasattr(dataset, "targets"):
        return int(dataset.targets[index])
    if hasattr(dataset, "samples"):
        return int(dataset.samples[index][1])
    _, class_id = dataset[index]
    return int(class_id)


def _majority(class_ids: Sequence[int]) -> int:
    """Returns the most frequent class id, ties broken by nearest neighbour."""
    best = class_ids[0]
    best_count = 0
    for candidate in dict.fromkeys(class_ids):
        count = class_ids.count(candidate)
        if count > best_count:
            best, best_count = candidate, count
    return best


def _chunked(items: Sequence[ImageTrace], size: int) -> Iterable[list[ImageTrace]]:
    """Yields ``items`` in lists of at most ``size`` entries."""
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


# --------------------------------------------------------------------------- #
# Stage 4: anomaly inference
# --------------------------------------------------------------------------- #
def normalize_anomaly_model(name: str) -> str:
    """Maps a user supplied anomaly model name to its canonical key."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = _ANOMALY_ALIASES.get(key, key)
    if key not in _ANOMALY_MODELS:
        raise ValueError(
            f"Unknown anomaly model {name!r}. Choose one of: "
            f"{', '.join(_ANOMALY_MODELS)}."
        )
    return key


def resolve_checkpoint(
    model_dir: Path,
    class_name: str | None,
    explicit: Path | None,
) -> Path:
    """Finds the anomalib checkpoint for one class.

    Args:
        model_dir: Root of the anomaly models.  With class routing it holds one
            sub-directory per class; without routing it holds the checkpoint.
        class_name: Routed class, or None when no routing happened.
        explicit: ``--anomaly-ckpt`` override (file or directory).

    Returns:
        Path of the checkpoint to load.

    Raises:
        FileNotFoundError: If no checkpoint matches.
    """
    if explicit is not None:
        return _newest_checkpoint(explicit)
    if class_name is None:
        return _newest_checkpoint(model_dir)
    class_dir = model_dir / class_name
    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"No model directory for class {class_name!r}: {class_dir}"
        )
    return _newest_checkpoint(class_dir)


def _newest_checkpoint(path: Path) -> Path:
    """Returns ``path`` itself when it is a file, else its newest ``*.ckpt``."""
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    candidates = sorted(path.rglob("*.ckpt"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No *.ckpt found under {path}")
    return candidates[-1]


def build_anomalib_model(name: str, image_size: tuple[int, int] | None) -> Any:
    """Instantiates the anomalib module for ``name``."""
    import anomalib.models as models

    model_cls = getattr(models, _ANOMALY_MODELS[name])
    kwargs: dict[str, Any] = {}
    if image_size is not None:
        kwargs["pre_processor"] = model_cls.configure_pre_processor(
            image_size=image_size
        )
    return model_cls(**kwargs)


def image_threshold(model: Any) -> float | None:
    """Reads the image-level threshold stored in the loaded checkpoint."""
    post_processor = getattr(model, "post_processor", None)
    if post_processor is None:
        return None
    for attribute in ("image_threshold", "_image_threshold"):
        value = getattr(post_processor, attribute, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            for inner in ("value", "threshold"):
                nested = getattr(value, inner, None)
                if nested is None:
                    continue
                try:
                    return float(nested)
                except (TypeError, ValueError):
                    continue
    return None


def run_anomaly_stage(
    traces: Sequence[ImageTrace],
    args: argparse.Namespace,
    stats: Stats,
) -> None:
    """Scores every routed image with its class' anomaly model.

    Args:
        traces: Traces that passed the YOLO gate (and routing, when enabled).
        args: Parsed CLI arguments.
        stats: Counters updated in place.
    """
    groups: dict[str | None, list[ImageTrace]] = {}
    for trace in traces:
        if trace.stage not in {"yolo_ok", "routed"}:
            continue
        key = trace.hoam_class or None
        groups.setdefault(key, []).append(trace)
    if not groups:
        _LOGGER.warning("No image reached the anomaly stage")
        return

    for class_name, group in groups.items():
        label = class_name or _UNROUTED_GROUP
        try:
            checkpoint = resolve_checkpoint(
                args.anomaly_model_dir, class_name, args.anomaly_ckpt
            )
        except FileNotFoundError as exc:
            _LOGGER.error("%s", exc)
            for trace in group:
                trace.fail("anomaly", str(exc))
                stats.anomaly_errors += 1
            continue

        _LOGGER.info(
            "Anomaly stage: %d image(s) for class %s using %s",
            len(group),
            label,
            checkpoint,
        )
        for trace in group:
            trace.anomaly_checkpoint = str(checkpoint)
        if args.dry_run:
            for trace in group:
                trace.stage = "anomaly_dry_run"
            continue

        try:
            _predict_group(group, class_name, checkpoint, args, stats)
        except Exception as exc:  # Report and continue with the next class.
            _LOGGER.error("Anomaly inference failed for class %s: %s", label, exc)
            for trace in group:
                if trace.stage in {"yolo_ok", "routed"}:
                    trace.fail("anomaly", str(exc))
                    stats.anomaly_errors += 1


def _predict_group(
    group: Sequence[ImageTrace],
    class_name: str | None,
    checkpoint: Path,
    args: argparse.Namespace,
    stats: Stats,
) -> None:
    """Runs one anomalib checkpoint over all images of a single class."""
    from anomalib.data import PredictDataset
    from anomalib.engine import Engine
    from torch.utils.data import DataLoader

    label = class_name or _UNROUTED_GROUP
    stage_dir = args.work_dir / "anomaly_input" / safe_folder_name(label)
    staged = _stage_images(group, stage_dir)
    if not staged:
        return

    transform = None
    batch_size = args.anomaly_batch_size
    if args.anomaly_image_size is not None:
        from torchvision.transforms.v2 import Resize

        transform = Resize(args.anomaly_image_size, antialias=True)
    elif batch_size > 1:
        # Without a shared size, images of different sizes cannot be collated.
        _LOGGER.debug("No --anomaly-image-size given; forcing batch size 1")
        batch_size = 1

    dataset = PredictDataset(path=stage_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=args.anomaly_num_workers,
    )
    model = build_anomalib_model(args.anomaly_model, args.anomaly_image_size)
    engine = Engine(
        default_root_dir=str(args.work_dir / "anomalib"),
        accelerator=args.accelerator,
        devices=_coerce_devices(args.devices),
        max_epochs=1,
        logger=False,
    )
    predictions = engine.predict(
        model=model, dataloaders=[loader], ckpt_path=str(checkpoint)
    )
    checkpoint_threshold = image_threshold(model)

    by_path = {key: trace for trace, keys in staged.items() for key in keys}
    for record in _collect_predictions(predictions):
        trace = by_path.get(record["path"])
        if trace is None:
            _LOGGER.warning("Unmatched prediction for %s", record["path"])
            continue
        _save_anomaly_result(
            trace, record, class_name, checkpoint_threshold, args, stats
        )

    for trace in group:
        if trace.stage in {"yolo_ok", "routed"}:
            trace.fail("anomaly", "no prediction returned")
            stats.anomaly_errors += 1


def _stage_images(
    group: Sequence[ImageTrace], stage_dir: Path
) -> dict[ImageTrace, tuple[str, str]]:
    """Links every image of one class into a flat directory for anomalib.

    Args:
        group: Traces belonging to one class.
        stage_dir: Directory that receives the links (recreated per run).

    Returns:
        Mapping of trace to the path keys anomalib may report for it.
    """
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    staged: dict[ImageTrace, tuple[str, str]] = {}
    for index, trace in enumerate(group, start=1):
        source = _scored_path(trace)
        name = f"{index:05d}_{safe_folder_name(source.stem)}{source.suffix.lower()}"
        link = stage_dir / name
        try:
            link.symlink_to(source.resolve())
        except (OSError, NotImplementedError):
            shutil.copy2(source, link)
        staged[trace] = (str(link), str(link.resolve()))
    return staged


def _collect_predictions(predictions: Any) -> list[dict[str, Any]]:
    """Flattens anomalib ``ImageBatch`` objects into per-image dicts."""
    import numpy as np

    def to_numpy(value: Any) -> Any:
        if value is None:
            return None
        detach = getattr(value, "detach", None)
        if detach is not None:
            return detach().cpu().float().numpy()
        return np.asarray(value)

    def item(array: Any, index: int) -> float | None:
        if array is None:
            return None
        flat = np.ravel(array)
        return float(flat[index]) if index < flat.size else None

    def plane(array: Any, index: int) -> Any:
        if array is None:
            return None
        return np.squeeze(array[index] if array.ndim >= 3 else array)

    records: list[dict[str, Any]] = []
    for batch in predictions or []:
        scores = to_numpy(getattr(batch, "pred_score", None))
        labels = to_numpy(getattr(batch, "pred_label", None))
        maps = to_numpy(getattr(batch, "anomaly_map", None))
        masks = to_numpy(getattr(batch, "pred_mask", None))
        paths = getattr(batch, "image_path", None)
        if isinstance(paths, str):
            paths = [paths]
        paths = list(paths or [])
        for index, path in enumerate(paths):
            records.append(
                {
                    "path": str(path),
                    "score": item(scores, index),
                    "label": item(labels, index),
                    "anomaly_map": plane(maps, index),
                    "pred_mask": plane(masks, index),
                }
            )
    return records


def _save_anomaly_result(
    trace: ImageTrace,
    record: dict[str, Any],
    class_name: str | None,
    checkpoint_threshold: float | None,
    args: argparse.Namespace,
    stats: Stats,
) -> None:
    """Writes one image and its result visualisation to the output tree.

    ``--anomaly-threshold`` wins when given; otherwise the label anomalib
    derived from the checkpoint's own post-processing decides, and only if that
    is missing is the score compared against the checkpoint threshold.
    """
    score = record["score"]
    label = record["label"]
    used_threshold: float | None = None
    if args.anomaly_threshold is not None and score is not None:
        is_ng = score >= args.anomaly_threshold
        used_threshold = args.anomaly_threshold
    elif label is not None:
        is_ng = bool(round(label))
    elif score is not None and checkpoint_threshold is not None:
        is_ng = score >= checkpoint_threshold
        used_threshold = checkpoint_threshold
    else:
        trace.fail("anomaly", "prediction carried neither label nor threshold")
        stats.anomaly_errors += 1
        return

    verdict = "NG" if is_ng else "OK"
    trace.anomaly_score = score
    trace.anomaly_threshold = used_threshold
    trace.anomaly_verdict = verdict
    trace.stage = f"anomaly_{verdict.lower()}"
    if is_ng:
        stats.anomaly_ng += 1
    else:
        stats.anomaly_ok += 1

    parts = ["anomaly"]
    if class_name:
        parts.append(safe_folder_name(class_name))
    parts.append(verdict)
    source = Path(trace.source)
    target_dir = args.output_dir.joinpath(*parts)
    target = unique_output(target_dir / source.name, set(), args.on_exists)
    if target is None:
        _LOGGER.debug("Exists, skipping: %s", target_dir / source.name)
        return

    scored = _scored_path(trace)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scored, target)
        trace.anomaly_output = str(target)
    except OSError as exc:
        _LOGGER.warning("Could not save %s: %s", target, exc)
        stats.anomaly_errors += 1
        return

    _LOGGER.info(
        "%s -> anomaly %s (score=%s)",
        trace.source,
        verdict,
        "n/a" if score is None else f"{score:.4f}",
    )
    if args.no_result_image:
        return
    result_path = target.with_name(f"{target.stem}_result.png")
    try:
        render_result_image(
            image_path=scored,
            anomaly_map=record["anomaly_map"],
            pred_mask=record["pred_mask"],
            score=score,
            verdict=verdict,
            target=result_path,
        )
        trace.anomaly_result_image = str(result_path)
    except (OSError, ValueError) as exc:
        _LOGGER.warning("Could not render %s: %s", result_path, exc)


def render_result_image(
    image_path: Path,
    anomaly_map: Any,
    pred_mask: Any,
    score: float | None,
    verdict: str,
    target: Path,
) -> None:
    """Writes an ``input | heat-map | predicted mask`` result strip.

    Args:
        image_path: Image that was scored.
        anomaly_map: ``(H, W)`` anomaly map, or None.
        pred_mask: ``(H, W)`` predicted mask, or None.
        score: Image-level anomaly score, or None.
        verdict: ``OK`` or ``NG``.
        target: Destination PNG path.
    """
    import numpy as np
    from PIL import ImageDraw, ImageFont

    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    size = original.size
    panels = [original]

    if anomaly_map is not None:
        heat = _resized_map(np.asarray(anomaly_map, dtype=float), size)
        overlay = Image.fromarray(_jet(heat)).convert("RGB")
        panels.append(Image.blend(original, overlay, 0.5))
    if pred_mask is not None:
        mask = (np.asarray(pred_mask) > 0).astype(float)
        resized = _resized_map(mask, size, normalize=False)
        panels.append(
            Image.fromarray((resized * 255).astype("uint8")).convert("RGB")
        )

    width = size[0] * len(panels)
    strip = Image.new("RGB", (width, size[1]), color=(0, 0, 0))
    for index, panel in enumerate(panels):
        strip.paste(panel, (index * size[0], 0))

    caption = f"{verdict}  score={'n/a' if score is None else f'{score:.4f}'}"
    font = ImageFont.load_default(size=max(12, size[1] // 25))
    draw = ImageDraw.Draw(strip)
    box = draw.textbbox((6, 4), caption, font=font)
    draw.rectangle((0, 0, box[2] + 6, box[3] + 4), fill=(0, 0, 0))
    draw.text(
        (6, 4),
        caption,
        font=font,
        fill=(255, 80, 80) if verdict == "NG" else (80, 255, 80),
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    strip.save(target)


def _resized_map(array: Any, size: tuple[int, int], normalize: bool = True) -> Any:
    """Min-max normalises a 2-D array to [0, 1] and resizes it to ``size``."""
    import numpy as np

    values = np.squeeze(np.asarray(array, dtype=float))
    if normalize:
        span = float(values.max() - values.min())
        values = (values - values.min()) / (span + 1e-8)
    if values.shape[::-1] != size:
        resized = Image.fromarray((values * 255).astype("uint8")).resize(size)
        values = np.asarray(resized, dtype=float) / 255.0
    return values


def _jet(values: Any) -> Any:
    """Maps [0, 1] values to an RGB uint8 jet-style colour map."""
    import numpy as np

    clipped = np.clip(values, 0.0, 1.0)
    red = np.clip(np.minimum(4 * clipped - 1.5, -4 * clipped + 4.5), 0, 1)
    green = np.clip(np.minimum(4 * clipped - 0.5, -4 * clipped + 3.5), 0, 1)
    blue = np.clip(np.minimum(4 * clipped + 0.5, -4 * clipped + 2.5), 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype("uint8")


def _coerce_devices(devices: str) -> Any:
    """Normalises the Lightning ``--devices`` value."""
    value = str(devices).strip()
    if value.lower() == "auto":
        return "auto"
    if "," in value:
        return [int(part) for part in value.split(",") if part.strip()]
    try:
        return int(value)
    except ValueError:
        return value


def _scored_path(trace: ImageTrace) -> Path:
    """Returns the oriented image if it exists, else the original source."""
    oriented = Path(trace.oriented) if trace.oriented else Path(trace.source)
    return oriented if oriented.exists() else Path(trace.source)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_report(
    traces: Sequence[ImageTrace],
    stats: Stats,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    """Writes ``report.json`` and ``report.csv`` below the output directory."""
    rows = [asdict(trace) for trace in traces]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "report.json"
    payload = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "yolo_model": str(args.yolo_model),
        "ok_label": args.ok_label,
        "ok_count": args.ok_count,
        "anomaly_model": args.anomaly_model,
        "anomaly_model_dir": str(args.anomaly_model_dir),
        "dry_run": args.dry_run,
        "summary": asdict(stats),
        "images": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    csv_path = args.output_dir / "report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(ImageTrace)])
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
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


def parse_image_size(text: str | None) -> tuple[int, int] | None:
    """Parses ``256`` or ``256x256`` into a ``(height, width)`` pair."""
    if not text:
        return None
    cleaned = text.lower().replace(" ", "")
    for separator in ("x", ",", "*"):
        if separator in cleaned:
            height, width = cleaned.split(separator, 1)
            return int(height), int(width)
    side = int(cleaned)
    return side, side


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the APL end-to-end AOI pipeline: orientation, YOLO gate, "
            "HOAM class routing and anomaly inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="One image file or a directory searched recursively.",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, type=Path,
        help="Root for OK/, NG/, anomaly/ and the run report.",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="Scratch directory for rotated and staged images. "
             "Default: <output-dir>/_work.",
    )
    parser.add_argument(
        "--ext", nargs="+", default=list(_DEFAULT_EXTENSIONS), metavar="EXT",
        help="Image extensions scanned when the input is a directory.",
    )
    parser.add_argument(
        "--limit", type=_positive_int, default=None,
        help="Process at most this many images (useful for smoke tests).",
    )

    orientation = parser.add_argument_group("stage 1: orientation")
    orientation.add_argument(
        "--target-orientation", choices=("landscape", "portrait"),
        default="landscape",
        help="Orientation every non-square image is rotated to.",
    )
    orientation.add_argument(
        "--rotation", choices=("clockwise", "counterclockwise"),
        default="clockwise",
        help="Direction used for required 90-degree rotations.",
    )

    yolo = parser.add_argument_group("stage 2: YOLO gate")
    yolo.add_argument(
        "--yolo-model", required=True,
        help="Ultralytics model path or name, e.g. best.pt.",
    )
    yolo.add_argument(
        "--ok-label", required=True,
        help="Exact YOLO class name that represents OK.",
    )
    yolo.add_argument(
        "--ok-count", required=True, type=_positive_int,
        help="Exact number of OK detections required.",
    )
    yolo.add_argument(
        "--center-tolerance", type=_unit_interval, default=0.25, metavar="RATIO",
        help="Allowed OK bbox-centre offset from the image centre per axis.",
    )
    yolo.add_argument("--conf", type=float, default=0.25,
                      help="YOLO confidence threshold.")
    yolo.add_argument("--iou", type=float, default=0.7,
                      help="YOLO NMS IoU threshold.")
    yolo.add_argument("--imgsz", type=int, default=640,
                      help="YOLO inference image size.")
    yolo.add_argument("--max-det", type=_positive_int, default=300,
                      help="Maximum YOLO detections per image.")
    yolo.add_argument("--yolo-device", default=None,
                      help="YOLO device, e.g. cpu, 0 or mps.")
    yolo.add_argument(
        "--save-ok-annotated", action="store_true",
        help="Also write the annotated OK images to <output-dir>/OK/.",
    )

    hoam = parser.add_argument_group(
        "stage 3: HOAM routing (anomaly_dino / patchcore only)"
    )
    hoam.add_argument(
        "--hoam-root", type=Path, default=None,
        help="HOAM repository root (or its src/) when hoam is not installed.",
    )
    hoam.add_argument(
        "--hoam-model-path", type=Path, default=None,
        help="Trained HOAM/HOAMV2 state_dict (.pt).",
    )
    hoam.add_argument(
        "--hoam-config", type=Path, default=None,
        help="Training config_used.yaml. Default: next to the model file.",
    )
    hoam.add_argument(
        "--hoam-structure", choices=("HOAM", "HOAMV2"), default=None,
        help="Model architecture. Inferred from the config, else HOAMV2.",
    )
    hoam.add_argument("--hoam-backbone", default=None,
                      help="Backbone used at training. Inferred from the config.")
    hoam.add_argument("--hoam-embedding-size", type=_positive_int, default=None,
                      help="Embedding size. Inferred from the config, else 128.")
    hoam.add_argument("--hoam-image-size", type=_positive_int, default=None,
                      help="Input size. Inferred from the config, else 224.")
    hoam.add_argument("--hoam-index", type=Path, default=None,
                      help="FAISS index. Default: <model-dir>/knn.index.")
    hoam.add_argument("--hoam-dataset-pkl", type=Path, default=None,
                      help="Reference dataset. Default: <model-dir>/dataset.pkl.")
    hoam.add_argument("--hoam-mean-std", type=Path, default=None,
                      help="Training mean_std.json. Default: <model-dir>/mean_std.json.")
    hoam.add_argument("--hoam-k", type=_positive_int, default=1,
                      help="Neighbours retrieved per image; the class is a majority vote.")
    hoam.add_argument("--hoam-batch-size", type=_positive_int, default=16,
                      help="Images embedded per KNN query batch.")
    hoam.add_argument("--hoam-device", default=None,
                      help="HOAM device, e.g. cpu, cuda or mps.")

    anomaly = parser.add_argument_group("stage 4: anomaly inference")
    anomaly.add_argument(
        "--anomaly-model", required=True, metavar="NAME",
        help="Anomaly detector: dinomaly | anomaly_dino | patchcore (aliases "
             "such as anomalydino and patch-core are accepted). dinomaly "
             "skips the HOAM routing stage.",
    )
    anomaly.add_argument(
        "--anomaly-model-dir", required=True, type=Path,
        help="Model root. With routing it holds one sub-directory per class; "
             "with dinomaly it holds the checkpoint.",
    )
    anomaly.add_argument(
        "--anomaly-ckpt", type=Path, default=None,
        help="Explicit checkpoint (file or directory) used for every image.",
    )
    anomaly.add_argument(
        "--anomaly-image-size", default=None, metavar="SIZE",
        help="Anomaly input size, '448' or '448x448'. Required to batch "
             "images of different sizes.",
    )
    anomaly.add_argument("--anomaly-batch-size", type=_positive_int, default=8,
                         help="Images per anomaly inference batch.")
    anomaly.add_argument("--anomaly-num-workers", type=int, default=0,
                         help="DataLoader workers for anomaly inference.")
    anomaly.add_argument(
        "--anomaly-threshold", type=float, default=None,
        help="Compare the predicted score against this value instead of "
             "trusting the checkpoint's own OK/NG label.",
    )
    anomaly.add_argument("--accelerator", default="auto",
                         help="Lightning accelerator: auto | cpu | gpu | mps.")
    anomaly.add_argument("--devices", default="auto",
                         help="Lightning devices: 'auto', an int, or '0,1'.")
    anomaly.add_argument(
        "--no-result-image", action="store_true",
        help="Skip the input|heat-map|mask result strip.",
    )

    parser.add_argument(
        "--on-exists", choices=("suffix", "skip", "overwrite"), default="suffix",
        help="What to do when an output image already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run stages 1-3 without writing images or scoring anomalies.",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose (DEBUG) logging.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only warnings and errors.")
    return parser.parse_args(argv)


def needs_class_routing(args: argparse.Namespace) -> bool:
    """Returns whether the run has to ask HOAM for a per-image class.

    ``dinomaly`` covers every class with one model, and an explicit
    ``--anomaly-ckpt`` pins one model for the whole run, so neither needs the
    routing stage.
    """
    return args.anomaly_model in _NEEDS_CLASS_ROUTING and args.anomaly_ckpt is None


def validate_args(args: argparse.Namespace) -> str | None:
    """Returns an error message when the argument combination is unusable."""
    if not args.input.exists():
        return f"Input path does not exist: {args.input}"
    if args.input.is_dir() and args.input.resolve() == args.output_dir.resolve():
        return "Input and output directories must be different"
    if not 0.0 <= args.conf <= 1.0:
        return "--conf must be between 0.0 and 1.0"
    if not 0.0 <= args.iou <= 1.0:
        return "--iou must be between 0.0 and 1.0"
    if args.imgsz < 1:
        return "--imgsz must be at least 1"
    if not args.anomaly_model_dir.exists():
        return f"--anomaly-model-dir does not exist: {args.anomaly_model_dir}"
    if needs_class_routing(args) and args.hoam_model_path is None:
        return (
            f"--hoam-model-path is required for --anomaly-model "
            f"{args.anomaly_model} (it routes each OK image to a class model). "
            "Pass --anomaly-ckpt instead to score every image with one model."
        )
    if args.hoam_model_path is not None and not args.hoam_model_path.exists():
        return f"--hoam-model-path does not exist: {args.hoam_model_path}"
    return None


def main(argv: list[str] | None = None) -> int:
    """Runs the four pipeline stages and writes the run report."""
    args = parse_args(argv)
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    try:
        args.anomaly_model = normalize_anomaly_model(args.anomaly_model)
        args.anomaly_image_size = parse_image_size(args.anomaly_image_size)
    except ValueError as exc:
        _LOGGER.error("%s", exc)
        return 2
    args.work_dir = args.work_dir or (args.output_dir / "_work")

    error = validate_args(args)
    if error:
        _LOGGER.error("%s", error)
        return 2

    extensions = tuple(
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in args.ext
    )
    pairs = find_images(args.input, args.output_dir, extensions)
    if not pairs:
        _LOGGER.warning("No images found under %s", args.input)
        return 0
    if args.limit is not None:
        pairs = pairs[: args.limit]
    _LOGGER.info("Found %d image(s) from %s", len(pairs), args.input)

    needs_routing = needs_class_routing(args)
    hoam_settings: HoamSettings | None = None
    if needs_routing:
        # Fail before inference starts if the routing artefacts are incomplete.
        try:
            prepare_hoam_imports(args.hoam_root)
            hoam_settings = resolve_hoam_settings(args)
        except (FileNotFoundError, ValueError) as exc:
            _LOGGER.error("%s", exc)
            return 2

    stats = Stats()
    traces = orient_images(
        pairs=pairs,
        work_dir=args.work_dir / "oriented",
        target=args.target_orientation,
        clockwise=args.rotation == "clockwise",
        dry_run=args.dry_run,
        stats=stats,
    )

    try:
        passed = run_yolo_stage(traces, args, stats)
    except ImportError:
        _LOGGER.error(
            "Ultralytics is not installed. Run `uv sync` before this command."
        )
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        _LOGGER.error("Could not load/validate the YOLO model: %s", exc)
        return 2

    if passed and needs_routing and hoam_settings is not None:
        if args.dry_run:
            _LOGGER.info("[dry-run] skipping HOAM routing for %d image(s)", len(passed))
        else:
            try:
                route_with_hoam(passed, hoam_settings, stats)
            except ImportError as exc:
                _LOGGER.error("HOAM dependencies are missing: %s", exc)
                return 2
            except (OSError, RuntimeError, ValueError) as exc:
                _LOGGER.error("HOAM routing failed: %s", exc)
                return 2

    if passed:
        try:
            run_anomaly_stage(passed, args, stats)
        except ImportError as exc:
            _LOGGER.error("anomalib dependencies are missing: %s", exc)
            return 2

    json_path, csv_path = write_report(traces, stats, args)
    _LOGGER.info(
        "Done%s. Summary:\n%s\nReport: %s\n        %s",
        " (dry-run)" if args.dry_run else "",
        stats.render(),
        json_path,
        csv_path,
    )
    errors = (
        stats.orient_errors
        + stats.yolo_errors
        + stats.routing_errors
        + stats.anomaly_errors
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
