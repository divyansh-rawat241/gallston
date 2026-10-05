# Final model and one-time test evaluation (pre-registered 2026-10-05)

Written **before** the test split is used for anything. The test split of
`dataset_grouped_neg4` (287 images: 116 stone images after excluding video
`a6`, 171 stone-free) has not been used for training, model selection,
threshold choice or any decision so far (see `docs/EXPERIMENTS.md`).

## Data

`dataset_grouped_neg4`: video/patient-grouped split, convention A labels
(faint companion echoes are ignore regions), `pl_*` web images and two bad
boxes removed, video `a6` held out pending per-stone relabelling, burned-in
text inpainted. Leak check clean (train/valid/test).

## Primary model (decided now)

RF-DETR-L, 704 px, COCO-pretrained, trained on **train + valid** of
`dataset_grouped_neg4` (`dataset_final_neg4`), same hyperparameters as the
CV runs (lr 5.61e-5, lr_encoder 7.97e-5, wd 0.00157, batch 3, effective ~9,
cosine schedule, flip-only augmentation), **20 epochs, fixed** - chosen from
the CV folds' best epochs (27, 7, 15, 23, 14; median 15, rounded up because
the final model sees ~25% more data). No early stopping on test. Weights:
`checkpoint_best_ema.pth` (rfdetr's EMA checkpoint; its monitoring split is part of the
training data, so this effectively tracks the latest EMA weights).

## Secondary model (reported, not selected)

Ensemble of the five CV fold models (`runs/cv4/rf704/fold*`; `checkpoint_best_total.pth`,
fold 0 `checkpoint_best_ema.pth` because it was stopped manually), fused with
weighted boxes fusion (IoU 0.55, `conf_type="avg"`, score floor 0.001).
Reported next to the primary model; the primary stays primary whatever the
test numbers show.

## Metrics (all on test, once)

- COCO mAP50:95 and mAP50 with 95% CI from a video-grouped bootstrap
  (200 resamples), ignore regions as COCO iscrowd.
- Operating threshold 0.25 (fixed in advance, never tuned): stone recall,
  share of stone images with >= 1 hit, false positives per image.
- Image level: AUROC stone vs. stone-free, separately vs. polyp and vs.
  normal gallbladder; share of stone-free images with a false positive.
- Known limitation, stated in advance: normal-gallbladder negatives come from
  a different scanner, so results on them may be optimistic.

## After the test run

Numbers are recorded in `docs/EXPERIMENTS.md` and `docs/BENCHMARKS.md`
unchanged. Any further model change after this point needs a new, untouched
test set (e.g. GBCU) before it can claim a test result.
