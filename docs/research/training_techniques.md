# Training techniques for our setup (research notes, 2026-10-03)

Source: web research agent, cross-checked against our logs. GPU-hour figures
are extrapolations from our own run times, not measurements.

## Key constraints found

- `valid` has 17 video groups: CI about +/-0.15 mAP, wider than most reported
  gains. Measurement must improve first.
- rfdetr default augmentation is weak (resize/crop jitter + 50% hflip only,
  unless `aug_config` is set): https://rfdetr.roboflow.com/latest/learn/train/augmentations/ ,
  issue https://github.com/roboflow/rf-detr/issues/81
- rfdetr has no training-time ignore regions (PR #1532 only ignores crowd
  boxes in eval): https://github.com/roboflow/rf-detr/pull/1532 ; no backbone
  freeze option (#783).

## Ranked experiments

1. **Paired grouped evaluation**: 5-fold grouped CV on train+valid (folds
   already in `split_manifest.json`) + paired video-cluster bootstrap of the
   *difference* between candidate and baseline. Low effort.
   Refs: https://arxiv.org/html/2609.14703v1 , https://arxiv.org/pdf/2307.00260
2. **Label audit + one convention** (one box per separable stone; gravel /
   sludge as a separate tag treated as ignore in eval, Open Images "group-of"
   style). Out-of-fold predictions + cleanlab object-detection label-quality
   scores; re-review pseudo-labelled videos. Box noise mostly hurts high-IoU
   mAP. Refs: https://cleanlab.ai/blog/learn/object-detection/ ,
   https://arxiv.org/pdf/1811.00982 , https://arxiv.org/abs/2207.09697 ,
   https://arxiv.org/pdf/2104.12218 . ~8-12 GPU h (5 YOLO folds) + review time.
   Never relabel test with a model that has seen test; version the labels.
3. **RF-DETR `aug_config`**: brightness/contrast/gamma (gain/TGC proxy), mild
   blur, speckle noise, rotation +/-10-15 deg, scale/translate, USAugment
   depth-attenuation / Gaussian shadow; start with `AUG_CONSERVATIVE`.
   Ultrasound augmentation benchmark (14 tasks): rotate/zoom/crop best
   (+5.4-5.5% avg). Refs: https://arxiv.org/abs/2501.13193 ,
   https://github.com/adamtupper/ultrasound-augmentation . ~2-3 GPU h.
4. **Fold ensemble of RF-DETR with WBF** (same architecture, tune WBF on
   out-of-fold preds). Ref: https://arxiv.org/abs/1910.13302 . Re-check
   negative-image false positives.
5. **Self-training on unlabelled frames** (ensemble pseudo-labels, temporally
   confirmed, strong aug, lower weight; new patients preferred). Semi-DETR is
   mmdet/multi-GPU, impractical. Refs: https://arxiv.org/pdf/2006.06882 ,
   https://arxiv.org/abs/2102.09480 , https://arxiv.org/abs/2302.07577 ,
   https://arxiv.org/pdf/2309.01391 . Do after label cleanup.
6. **Stone copy-paste with synthetic posterior shadow** inside the GB lumen
   (positives or negatives). Copy-paste: https://arxiv.org/abs/2012.07177 ;
   ultrasound-shadow evidence weak.
7. **YOLO P2 head** (small objects). Moderate/weak transfer evidence:
   https://github.com/orgs/ultralytics/discussions/8227
8. **Small RF-DETR HP sweep on 2-3 folds**: lower `lr_encoder`, drop_path 0.1,
   warmup 1-3, 704 vs 896 with negatives; leave `num_queries`; FAQ reports
   real batch 16 >> batch 4 x accum 4 -> try RF-DETR-M at larger real batch.
   https://rfdetr.roboflow.com/latest/faq/
9. **Checkpoint averaging (SWA)** after training: ~+1 AP on COCO.
   https://arxiv.org/abs/2012.12645
10. **Temporal context** (only for video deployment): IoU-tracking to suppress
    transient FPs (https://arxiv.org/abs/2305.18060), or 3-frame channel
    stacking.

Deprioritised: SAHI training (our E5 tiles tripled FPs), NWD loss, DQ-DETR.

Suggested order: 1 -> 2 -> 3 (on folds) -> 4 -> 5, then 6-9.
