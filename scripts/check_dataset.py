"""Verify FloorPlanCAD image/label pairing without importing PyTorch."""

import argparse
from collections import Counter
from pathlib import Path

from _bootstrap import ROOT


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/FloorPlanCAD")
    args = parser.parse_args()

    total_images = total_labels = total_instances = 0
    errors: list[str] = []
    class_counts: Counter[int] = Counter()

    for split in ("train", "val", "test"):
        split_instances_before = total_instances
        image_dir = args.root / "images" / split
        label_dir = args.root / "labels" / split
        images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        labels = sorted(label_dir.rglob("*.txt"))
        label_stems = {p.relative_to(label_dir).with_suffix("") for p in labels}
        image_stems = {p.relative_to(image_dir).with_suffix("") for p in images}

        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)
        empty_labels = sum(label.stat().st_size == 0 for label in labels)
        # Background images legitimately have no label; report rather than fail them.
        for label in labels:
            for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    errors.append(f"{label}:{line_no}: expected 5 fields, got {len(fields)}")
                    continue
                try:
                    cls = int(float(fields[0]))
                    coords = [float(value) for value in fields[1:]]
                except ValueError:
                    errors.append(f"{label}:{line_no}: non-numeric YOLO annotation")
                    continue
                if not 0 <= cls < 28 or any(not 0 <= value <= 1 for value in coords):
                    errors.append(f"{label}:{line_no}: value outside valid range")
                class_counts[cls] += 1
                total_instances += 1

        total_images += len(images)
        total_labels += len(labels)
        print(
            f"{split:5s}: images={len(images):5d}, labels={len(labels):5d}, "
            f"empty/background={empty_labels:4d}, missing-labels={len(missing_labels):3d}, "
            f"orphan-labels={len(orphan_labels):3d}, instances={total_instances - split_instances_before:5d}"
        )
        if orphan_labels:
            errors.extend(f"{split}: orphan label {path}" for path in orphan_labels[:20])

    print(f"Total: images={total_images}, labels={total_labels}, instances={total_instances}")
    print(f"Classes found: {len(class_counts)} / 28")
    if errors:
        raise SystemExit("Dataset check failed:\n" + "\n".join(errors[:50]))
    print("Dataset check passed.")


if __name__ == "__main__":
    main()
