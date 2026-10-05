# Improvement plan from the 2026-10-03 research sweep

Combines [training_techniques.md](training_techniques.md),
[pretrained_models.md](pretrained_models.md), [literature.md](literature.md)
and [datasets.md](datasets.md). Experiments are logged in
[../EXPERIMENTS.md](../EXPERIMENTS.md).

## Tier 1 - do next (high value, low/medium effort)

| # | Action | Why | Cost |
|---|---|---|---|
| 1 | **Grouped 5-fold CV + paired video-bootstrap of differences** in `eval_grouped.py` / a CV driver | valid CI (+/-0.15) is wider than most gains; folds already in the manifest | script; folds reuse training time below |
| 2 | **RF-DETR `aug_config`** (gain/gamma, speckle, blur, rotation +/-10-15 deg, scale/translate; start conservative) | rfdetr default aug is only resize + hflip; US augmentation benchmark +5% | 1 RF run (~2.5 h) |
| 3 | **More hard negatives**: normal GB from MSU Abdominal US (CC BY 4.0) + non-stone UIdataGB classes (polyps, wall thickening, CC BY 4.0) | negatives already cut FP 62% -> 0%; 136 negatives is the weakest part of the data | download (needs approval) + rebuild split |
| 4 | **Label audit with out-of-fold predictions** (cleanlab OD scores), one convention (box per separable stone; gravel = ignore tag) | inconsistent conventions corrupt DETR matching and high-IoU mAP | 5 YOLO folds (~8-12 h) + human review |
| 5 | **Fold ensemble (WBF) of RF-DETR** | same-architecture fold ensembles are the standard winner | free once folds exist |

## Tier 2 - bigger levers

| # | Action | Why | Cost |
|---|---|---|---|
| 6 | **New gallstone images from UIdataGB** (1,326 stone images / 221 pts) + its Roboflow boxed "parts"; pHash-dedupe augmented copies; pseudo-box + review; train only | only sizeable open new-gallstone source; multi-vendor | download + review time |
| 7 | **Continued DINO/iBOT SSL of RF-DETR-L's own ViT-S/16 encoder** on unlabelled US (dataset_new, MSU, kidney), written back into the checkpoint | domain SSL beat generic and US foundation models (UltraDINO) | ~6-12 h + engineering |
| 8 | **Gallbladder-ROI-first detection** (find GB, detect stones on an up-sampled crop) | GBCNet / FocusMAE / OGLMFI; helps small stones, removes off-organ FPs | medium engineering |
| 9 | **Self-training on unlabelled frames** (after #4) | +1-3 AP typical; risk of copying label noise | 1-2 retrains |
| 10 | **Second-stage stone pretraining on URI-CAD expert boxes** (CC BY-NC, gated) and the Punjabi kidney set | independent vendor/centre stone boxes | download + 1 run |

## Paper / patent actions

- Lead with the leak-free video-grouped benchmark (no gallstone paper reports
  COCO mAP50:95 on a patient/video split; Kumar 2025 got 0.32 on a random
  split). Add a deliberately leaky random-split run to quantify inflation.
- Re-scope claims: shadow-aware *learned multi-detector fusion* (not "shadow
  detects stones"); conformal guarantees made *video-/study-level*.
- **Freedom-to-operate check on US9597103B2** (bright spot + distal shadow,
  active to ~2031) before filing.
- GBCU (+ GBUSV v2 in the same request) as the external test set.

## Avoid

Kaggle `aneerbansaha/gallbladder-cancer` (probable GBCU re-upload), Roboflow
workspace `gallstone-kbw07` and its copies (likely our own source), paid
kidney resale, SAHI/tiling, NWD loss, ViT-B foundation backbone swaps.
