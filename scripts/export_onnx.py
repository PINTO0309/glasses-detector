import argparse
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn
from onnxsim import simplify

from glasses_detector import GlassesClassifier, GlassesDetector, GlassesSegmenter


@dataclass(frozen=True)
class ExportTarget:
    task: str
    kind: str
    size: str
    model_cls: type[GlassesClassifier | GlassesDetector | GlassesSegmenter]

    @property
    def filename(self) -> str:
        return f"{self.task}_{self.kind}_{self.size}.onnx"


class SegmentationOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        return output["out"] if isinstance(output, dict) else output


class DetectionOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions = self.model(list(images))
        boxes = torch.stack([prediction["boxes"] for prediction in predictions])
        labels = torch.stack([prediction["labels"] for prediction in predictions])
        scores = torch.stack([prediction["scores"] for prediction in predictions])
        return boxes, labels, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all supported glasses-detector models to simplified ONNX."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("onnx"),
        help="Directory where ONNX files will be written. Defaults to ./onnx.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to use. Defaults to 17.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square input image size used for export. Defaults to 256.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for export. Defaults to 1.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used during export. Defaults to cpu.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["small", "medium"],
        choices=["small", "medium", "large"],
        help=(
            "Model sizes to export. Defaults to small medium because large pretrained "
            "weights are not supported by the package loader."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["classification", "detection", "segmentation"],
        choices=["classification", "detection", "segmentation"],
        help="Tasks to export. Defaults to all tasks.",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=None,
        help="Optional list of kinds to export, e.g. anyglasses worn smart.",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Export initialized model architectures without loading pretrained weights.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Mark the batch dimension dynamic for classification and segmentation exports.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip targets whose output file already exists.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately if any target fails to export.",
    )
    return parser.parse_args()


def iter_targets(
    sizes: Iterable[str],
    tasks: Iterable[str],
    kinds: Iterable[str] | None,
) -> Iterable[ExportTarget]:
    specs: tuple[tuple[str, type[Any], tuple[str, ...]], ...] = (
        (
            "classification",
            GlassesClassifier,
            ("anyglasses", "eyeglasses", "sunglasses", "shadows"),
        ),
        ("detection", GlassesDetector, ("eyes", "solo", "worn")),
        (
            "segmentation",
            GlassesSegmenter,
            ("frames", "full", "legs", "lenses", "shadows", "smart"),
        ),
    )

    selected_tasks = set(tasks)
    selected_kinds = set(kinds) if kinds is not None else None

    for task, model_cls, task_kinds in specs:
        if task not in selected_tasks:
            continue

        for kind in task_kinds:
            if selected_kinds is not None and kind not in selected_kinds:
                continue

            for size in sizes:
                yield ExportTarget(task=task, kind=kind, size=size, model_cls=model_cls)


def wrap_model(task: str, model: nn.Module) -> nn.Module:
    if task == "detection":
        return DetectionOutputWrapper(model)
    if task == "segmentation":
        return SegmentationOutputWrapper(model)
    return model


def output_names(task: str) -> list[str]:
    if task == "detection":
        return ["boxes", "labels", "scores"]
    if task == "segmentation":
        return ["logits"]
    return ["logits"]


def dynamic_axes(task: str, enabled: bool) -> dict[str, dict[int, str]] | None:
    if not enabled or task == "detection":
        return None

    axes = {"images": {0: "batch"}, "logits": {0: "batch"}}
    return axes


def export_target(
    target: ExportTarget,
    output_path: Path,
    opset: int,
    image_size: int,
    batch_size: int,
    device: torch.device,
    weights: bool,
    use_dynamic_batch: bool,
) -> None:
    print(f"Exporting {target.task}:{target.kind}:{target.size} -> {output_path}")

    glasses_model = target.model_cls(
        kind=target.kind,
        size=target.size,
        weights=weights,
        device=device,
    )
    model = wrap_model(target.task, glasses_model.model).to(device).eval()
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.inference_mode():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using the legacy TorchScript-based ONNX export.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*Iterating over a tensor might cause the trace to be incorrect.*",
                category=torch.jit.TracerWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*Using len to get tensor shape might cause the trace to be incorrect.*",
                category=torch.jit.TracerWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Constant folding - Only steps=1 can be constant folded.*",
                category=UserWarning,
            )
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                dynamo=False,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["images"],
                output_names=output_names(target.task),
                dynamic_axes=dynamic_axes(target.task, use_dynamic_batch),
            )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    simplified_model, ok = simplify(onnx_model)
    if not ok:
        raise RuntimeError(f"onnxsim failed to validate {output_path}")

    onnx.checker.check_model(simplified_model)
    onnx.save(simplified_model, output_path)
    print(f"Simplified {output_path}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    weights = not args.no_weights
    failures: list[tuple[ExportTarget, Exception]] = []

    for target in iter_targets(args.sizes, args.tasks, args.kinds):
        output_path = args.output_dir / target.filename
        if args.skip_existing and output_path.exists():
            print(f"Skipping existing {output_path}")
            continue

        try:
            export_target(
                target=target,
                output_path=output_path,
                opset=args.opset,
                image_size=args.image_size,
                batch_size=args.batch_size,
                device=device,
                weights=weights,
                use_dynamic_batch=args.dynamic_batch,
            )
        except Exception as exc:
            if args.strict:
                raise
            print(f"Failed {target.task}:{target.kind}:{target.size}: {exc}")
            failures.append((target, exc))

    if failures:
        print("\nFailures:")
        for target, exc in failures:
            print(f"- {target.task}:{target.kind}:{target.size}: {exc}")
        return 1

    print("All ONNX exports completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
