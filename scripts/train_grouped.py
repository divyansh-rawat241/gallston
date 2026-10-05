"""Train YOLOv26L and/or RF-DETR-L on the leakage-free grouped split.

Safeguards against repeating the 2026-10-01 audit findings:
  * Refuses to start unless scripts/check_leakage.py passes on the dataset.
  * Starts from public COCO-pretrained weights (`yolo26l.pt`, RF-DETR's default
    download), NEVER from production_models/ or runs/: those were fine-tuned on
    images that now sit in the grouped test set.
  * Trains RF-DETR with the 4-layer Large architecture it will be loaded with.
  * The test split is not touched here; model selection uses `valid` only.

Hyperparameters are the Optuna winners from the old runs. They were selected
on a leaked validation set, so treat them as a starting point, not as tuned.

Usage:
  python scripts/train_grouped.py --model yolo            # full YOLO run
  python scripts/train_grouped.py --model rfdetr          # full RF-DETR run
  python scripts/train_grouped.py --model both --smoke    # 1-epoch pipeline check
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Optuna winners from run_yolo26l_hpo.py (trial 16) and run_rfdetr_large_hpo.py (trial 10)
YOLO_HPARAMS = dict(
    optimizer="AdamW", lr0=0.0003729790675877055, lrf=0.1848577362852768, momentum=0.937,
    weight_decay=0.002179440885696096, warmup_epochs=5, warmup_momentum=0.8, warmup_bias_lr=0.1,
    box=6.31751915780746, cls=0.600366886995146, dfl=2.228632819733598, cos_lr=False,
    close_mosaic=15, amp=True, nbs=64, mixup=0.02553818035946693, copy_paste=0.09425586863956208,
    dropout=0.12209905583667471, mosaic=1.0, fliplr=0.5, scale=0.5, erasing=0.4,
    auto_augment="randaugment",
)
# rfdetr's default augmentation is only resize/crop jitter + horizontal flip.
# Ultrasound-motivated set (docs/research/training_techniques.md #3): gain /
# contrast (TGC proxy), mild blur and noise (speckle proxy), small rotation,
# scale and shift. No colour transforms, no vertical flip (shadows fall
# downward), only transforms the GPU (Kornia) backend supports.
RFDETR_AUG = {
    "ultrasound": {
        "HorizontalFlip": {"p": 0.5},
        "Affine": {"scale": (0.85, 1.15), "translate_percent": (-0.05, 0.05), "rotate": (-12, 12), "p": 0.5},
        "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5},
        "GaussianBlur": {"blur_limit": 3, "p": 0.2},
        "GaussNoise": {"std_range": (0.01, 0.04), "p": 0.3},
    },
}
RFDETR_HPARAMS = dict(
    lr=5.61e-05, lr_encoder=7.97e-05, weight_decay=0.00157, clip_max_norm=0.319,
    batch_size=2, grad_accum_steps=4,
)


def leakage_gate(data: str) -> None:
    # cross-validation folds (scripts/build_cv_folds.py) have no test split
    splits = [s for s in ("train", "valid", "test") if (ROOT / data / s).exists()]
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_leakage.py"), "--data", data, "--splits", *splits],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout.strip())
    if r.returncode != 0:
        sys.exit("Leakage gate failed: fix the split before training (scripts/build_grouped_split.py).")


def seg_variant(data_dir: Path) -> Path:
    """Same split with polygon labels in `labels/`, for YOLO segmentation models.

    Images are hard-linked, not copied. Polygon supervision gives the model the
    stone outline and makes Ultralytics' copy_paste augmentation work (it is a
    no-op on box-only labels).
    """
    import os
    import shutil
    seg = data_dir.with_name(data_dir.name + "_seg")
    if seg.exists():
        shutil.rmtree(seg)
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            (seg / split / sub).mkdir(parents=True)
        for img in (data_dir / split / "images").iterdir():
            os.link(img, seg / split / "images" / img.name)
        for lab in (data_dir / split / "labels_polygon").iterdir():
            shutil.copy2(lab, seg / split / "labels" / lab.name)
    (seg / "data.yaml").write_text((data_dir / "data.yaml").read_text().replace(
        data_dir.as_posix(), seg.as_posix()), encoding="utf-8")
    return seg


def train_yolo(data_dir: Path, out: Path, epochs: int, patience: int, smoke: bool,
               weights: str = "yolo26l.pt", imgsz: int = 640, overrides: dict | None = None) -> Path:
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    SETTINGS["mlflow"] = False
    if "-seg" in weights:
        data_dir = seg_variant(data_dir)
    name = Path(weights).stem
    model = YOLO(weights)  # COCO-pretrained, downloaded by Ultralytics if absent
    model.train(
        data=str(data_dir / "data.yaml"), epochs=1 if smoke else epochs, patience=patience,
        imgsz=imgsz, batch=6 if imgsz <= 640 else max(1, int(6 * (640 / imgsz) ** 2)), workers=2,
        device=0, seed=42, deterministic=True,
        fraction=0.1 if smoke else 1.0, project=str(out), name=name, exist_ok=True,
        plots=not smoke, verbose=False, **{**YOLO_HPARAMS, **(overrides or {})},
    )
    return out / name / "weights" / "best.pt"


def train_rfdetr(data_dir: Path, out: Path, epochs: int, patience: int, smoke: bool,
                 resolution: int | None = None, init: str | None = None, aug: str | None = None,
                 class_names: list[str] | None = None, batch: int | None = None, workers: int = 2) -> Path:
    from rfdetr import RFDETRLarge
    # COCO-pretrained Large; 4 decoder layers. Small stones (AP 0.26 vs 0.56 for
    # large) motivate a resolution above the 704 default; keep the effective
    # batch at 8 by trading batch size for accumulation.
    kw = {"resolution": resolution} if resolution else {}
    if init:  # e.g. a checkpoint pretrained on kidney-stone ultrasound
        kw["pretrain_weights"] = init
    model = RFDETRLarge(**kw)
    hp = dict(RFDETR_HPARAMS)
    if resolution and resolution > 704:
        hp.update(batch_size=1, grad_accum_steps=8)
    if aug:
        hp["aug_config"] = RFDETR_AUG[aug]
    if batch:  # keep the effective batch near 8 (lr was tuned for 2 x 4)
        hp.update(batch_size=batch, grad_accum_steps=max(1, round(8 / batch)))
    run_dir = out / "rfdetr_large"
    model.train(
        dataset_dir=str(data_dir), dataset_file="yolo", class_names=class_names or ["gallstone"],
        epochs=1 if smoke else epochs, early_stopping=not smoke, early_stopping_patience=patience,
        output_dir=str(run_dir), num_workers=workers, eval_interval=1, checkpoint_interval=50,
        lr_scheduler="cosine", wandb=False, mlflow=False, progress_bar="tqdm", **hp,
    )
    return run_dir / "checkpoint_best_total.pth"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["yolo", "rfdetr", "both"], default="both")
    ap.add_argument("--data", default="dataset_grouped")
    ap.add_argument("--out", default="runs/grouped")
    ap.add_argument("--yolo-epochs", type=int, default=300)
    ap.add_argument("--rfdetr-epochs", type=int, default=150)
    ap.add_argument("--yolo-weights", default="yolo26l.pt", help="COCO start; *-seg.pt trains on polygons")
    ap.add_argument("--yolo-imgsz", type=int, default=640)
    ap.add_argument("--yolo-set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override a YOLO hyperparameter, e.g. copy_paste=0.3")
    ap.add_argument("--rfdetr-aug", choices=sorted(RFDETR_AUG), help="augmentation preset (default: rfdetr's flip only)")
    ap.add_argument("--workers", type=int, default=2,
                    help="RF-DETR dataloader workers; each is a full Python+torch process on Windows, "
                         "1 avoids running out of virtual memory at validation")
    ap.add_argument("--rfdetr-batch", type=int, help="real batch size; grad accumulation keeps effective batch ~8")
    ap.add_argument("--class-names", nargs="+", help="RF-DETR classes in label-id order (default: gallstone)")
    ap.add_argument("--rfdetr-init", help="start RF-DETR from this checkpoint instead of COCO")
    ap.add_argument("--no-leak-gate", action="store_true",
                    help="skip check_leakage; ONLY for external pretraining data or the final train+valid model")
    ap.add_argument("--rfdetr-resolution", type=int, help="multiple of 32; default 704")
    ap.add_argument("--patience", type=int, default=50, help="early-stopping patience on the valid split")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, small fraction: checks the pipeline only")
    args = ap.parse_args()

    data_dir, out = ROOT / args.data, ROOT / args.out / ("smoke" if args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    if args.no_leak_gate:
        print("leakage gate skipped (--no-leak-gate): external pretraining data or the final train+valid model "
              "(scripts/build_final_dataset.py; leakage vs test checked on the source dataset)")
    else:
        leakage_gate(args.data)

    log = {"data": args.data, "smoke": args.smoke, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "split_manifest": str(data_dir / "split_manifest.json"), "weights": {}}
    if args.model in ("yolo", "both"):
        t0 = time.time()
        log["weights"]["yolo"] = str(train_yolo(data_dir, out, args.yolo_epochs, args.patience, args.smoke,
                                                  args.yolo_weights, args.yolo_imgsz,
                                                  {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.yolo_set)}))
        log["yolo_minutes"] = round((time.time() - t0) / 60, 1)
    if args.model in ("rfdetr", "both"):
        t0 = time.time()
        log["weights"]["rfdetr"] = str(train_rfdetr(data_dir, out, args.rfdetr_epochs, min(args.patience, 20), args.smoke,
                                                    args.rfdetr_resolution, args.rfdetr_init,
                                                    args.rfdetr_aug, args.class_names, args.rfdetr_batch,
                                                    args.workers))
        log["rfdetr_minutes"] = round((time.time() - t0) / 60, 1)
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))
    print("\nNext: python scripts/eval_grouped.py --yolo <yolo best.pt> --rfdetr <rfdetr .pth>")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
