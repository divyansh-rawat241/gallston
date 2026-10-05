# Leakage and Metric Audit — 2026-10-01

Every number this project has reported so far is inflated: a third of the
test set is byte-identical copies of training images, every test video also
appears in training, and the headline evaluation script reports per-image F1
under the name "mAP". This file records what was tested and what was found,
before any fix. Audit script: kept outside the repo; raw output reproduced
below.

## 1. Claim: "mAP50 was actually average F1" — partly true

| Number reported | Produced by | Is it mAP? |
| --- | --- | --- |
| 0.9598 YOLO, 0.9606 RF-DETR (Optuna best) | Ultralytics / RF-DETR built-in COCO evaluators on `val` | Yes, real mAP50 — but on a leaked `val` |
| 0.9298 baseline, 0.8260 shadow, 0.9166 TTA, 0.4974 combined | `src/evaluation/map_calculator.calculate_map` | **No** — mean over images of per-image F1 at IoU 0.5, at a fixed 0.25 threshold |
| Shadow-threshold, TTA-scale, ensemble-weight tuning results | `calculate_map` | **No** — same function |

Measured on the same predictions (test split, n = 201), COCO mAP vs the
repo's function:

| Model | COCO mAP50 | COCO mAP50:95 | `calculate_map` (per-image F1) |
| --- | ---: | ---: | ---: |
| YOLOv26L | 0.957 | 0.841 | 0.923 |
| RF-DETR, production checkpoint (3-layer weights in 4-layer model) | 0.966 | 0.770 | 0.897 |
| RF-DETR-L, correct 4-layer checkpoint | 0.966 | 0.837 | 0.916 |

The two metrics disagree by 3–7 points and are not comparable. COCO mAP is
the right one to report.

## 2. Leakage found

### 2.1 Exact copies across splits (new, most severe)

| Split | Files byte-identical to a training file |
| --- | ---: |
| test | 125 / 201 (62.2%) |
| val | 250 / 368 (67.9%) |
| test identical to a val file | 32 |

`dataset_final_resplit/` holds 1,593 files but only 1,186 unique images
(407 duplicate pairs, all with identical labels).

**Root cause:** `scripts/resplit_dataset_corrected.py` creates the target
folders with `mkdir(exist_ok=True)` and copies into them without emptying
them first. A re-run on top of an earlier split left the earlier files in
place, so the same filename sits in two splits. Its "remove duplicates" step
deduplicates `Path` objects, not file contents, so it never catches this.

### 2.2 Same video in train and test

1,520 images are frames named `pseudo_<video>__<frame>` from 58 videos.
All 49 test videos and all 52 val videos also have frames in train. The
source folder `dataset_final/` has the same problem (54 of 58 videos span
more than one split), although it has no exact copies.

A test image's most similar training frame has a median correlation of
**1.000**; restricted to training frames from *other* videos, the median
drops to **0.574**, and none exceed 0.95. Videos are therefore a valid
grouping unit: no near-duplicates cross video boundaries, and none of the
73 `pl_*` images matches any video.

### 2.3 Measured effect on the test score

| Model | COCO mAP50:95, test images copied in train (n = 125) | Test images not copied (n = 76) | Gap |
| --- | ---: | ---: | ---: |
| YOLOv26L | 0.900 | 0.760 | −0.140 |
| RF-DETR production | 0.810 | 0.712 | −0.098 |
| RF-DETR-L 4-layer | 0.876 | 0.781 | −0.095 |

The "not copied" subset is still optimistic, because its videos are in
training too. Honest performance is expected to be lower than 0.76–0.78
mAP50:95; it can only be measured on a video-grouped split.

### 2.4 Everything fitted on `val` inherits the leak

Optuna hyperparameter selection, temperature scaling, isotonic fusion,
selective thresholds, conformal quantiles, the CRC threshold, shadow
threshold, TTA scales and ensemble weights were all fitted on a `val` split
that is 68% copies of training images.

## 3. Other defects found

- **No negative images.** All 1,593 images contain at least one stone. False
  positives on normal gallbladders cannot be measured. Adding the Mississippi
  "normal GB" images as negatives would create a shortcut (they are GE
  Voluson, sepia-coloured; all positives are greyscale), so that needs care.
- **Deployed RF-DETR mismatch.** `production_models/rfdetr_best.pth` has 3
  decoder layers but is loaded as `RFDETRLarge` (4 layers): layer 3 is random.
  The correct 4-layer checkpoint scores the same mAP50 but +0.067 mAP50:95.
- **Manual review exclusions.** 49 images were flagged "bad" during review
  and removed from all splits (`review_state_*.json`). Applied to every split,
  so not a test-selection bias; they stay excluded.
- **Preprocessing.** Training and the API both use raw images and leave
  resizing to the models, so train/inference are consistent. No dataset-level
  statistics are fitted, so preprocessing itself leaks nothing.

## 4. What changes (next section of work)

1. New split builder: deduplicate by content, drop review-flagged images,
   assign whole videos to one split, wipe the target folder first.
2. Leakage checker that fails on any shared file, video or near-duplicate
   across splits; run before every training.
3. COCO mAP evaluator replacing `calculate_map` in all scripts.
4. Training scripts on the new split, with the correct RF-DETR-L model.
