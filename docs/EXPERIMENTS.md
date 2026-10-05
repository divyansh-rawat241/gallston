# Improvement loop: experiment log

Started 2026-10-02 after the leakage audit ([LEAKAGE_AUDIT.md](LEAKAGE_AUDIT.md)).
Loop: find a method -> apply -> evaluate on `valid` -> keep or drop -> repeat.

## Protocol

- Split: `dataset_grouped` (whole videos per split, leak-checked). From E3 on,
  `dataset_grouped_neg` = the same positive split (verified identical, image by
  image, including folds) plus 136 stone-free gallbladder images as hard
  negatives (94 train / 21 valid / 21 test).
- Selection metric: COCO mAP50:95 on `valid` with video-grouped bootstrap CI,
  plus image-level AUROC and false positives per stone-free image
  (`scripts/eval_grouped.py --split valid`).
- `valid` has only 17 video groups: CIs are about +/-0.15 wide. A change is only
  adopted on a large gain, or after confirmation by grouped cross-validation
  (folds are stored in `split_manifest.json`).
- `test` is never used for a decision. It is evaluated once, at the end, on the
  final model.

## Baselines (first overlay-removal version, `runs/grouped/`)

| Model | valid mAP50:95 | valid mAP50 | Notes |
|---|---:|---:|---|
| YOLOv26L 640 | 0.394 [0.17, 0.54] | 0.596 | early stop epoch 74 |
| RF-DETR-L 704 | 0.498 [0.30, 0.65] | 0.731 | early stop epoch 42 |
| WBF of both | 0.444 [0.23, 0.59] | | worse than RF-DETR alone |

## Experiments

| ID | Change | Status | Result |
|---|---|---|---|
| E1 | Textured fill for overlay inpainting (texture ratio 0.55 -> 1.00) | data | adopted, rebuilds dataset |
| E2 | RF-DETR-L at 896 px (small stones: AP_small 0.26 vs AP_large 0.56), trained on `dataset_grouped` | done (190 min, best epoch 20) | on `dataset_grouped_neg` valid: mAP50:95 0.448 [0.27, 0.58], mAP50 0.675; stone recall 0.73, 96% of stone images hit at conf 0.25; but 62% of stone-free GB images get a false positive (AUROC 0.895). + hflip TTA: 0.449, no gain -> TTA dropped |
| E3 | Hard negatives: normal GB from `dataset_new` (tinted, "GB" text inpainted), YOLOv26L 640 | done (best epoch 91; run interrupted by a restart at 113, resumed, stopped at 170 because resume reset early stopping) | valid mAP50:95 0.379 [0.17, 0.51], mAP50 0.578; **0 of 21 stone-free images with a false positive, image AUROC 0.989** (vs 62% / 0.895 for E2, trained without negatives); stone recall 0.51, FP/img 0.28. Negatives adopted for all further runs |
| E4 | YOLOv26L-seg on polygons + copy_paste 0.3 (copy-paste is a no-op on box labels) | queued | |
| E5 | Test-time augmentation, old YOLO on `dataset_grouped_neg` valid (no TTA: 0.280) | done | hflip + max-fusion 0.303 (+0.02, within noise); hflip + avg-fusion 0.244; 2x2 tiles 0.207-0.230 (stone recall up, but FP/img 0.39 -> 1.35). Tiles dropped; hflip re-tested on final models |
| E6 | Kidney-stone ultrasound pretraining (Roboflow `kidney-stone-ultrasound`, CC BY 4.0, 5,171 train images / 14,688 stone boxes, no near-duplicates of our data, max similarity 0.77), then gallstone fine-tune | YOLO done (pretrain 15 ep / 86 min; fine-tune 191 min, best epoch 127); RF-DETR moved behind E8 | YOLO: valid mAP50:95 0.388 [0.16, 0.53] vs 0.379 without (E3); mAP50 0.557 vs 0.578; stone recall 0.50 vs 0.51; AUROC 0.964 vs 0.989; 1/21 vs 0/21 negatives with FP. **No measurable gain for YOLO** |
| E8 | RF-DETR-L 704 + ultrasound `aug_config` (flip, affine +/-12 deg / 0.85-1.15 / 5% shift, brightness-contrast 0.2, blur, noise) on `dataset_grouped_neg` | done (184 min, best EMA epoch 34) | valid mAP50:95 0.429 [0.25, 0.58] vs 0.443 (E7); mAP50 0.670 vs 0.683; stone recall 0.68 vs 0.71; FP/img 0.64 vs 0.84; AUROC 1.00, 0/21 negatives with FP. **No gain; not adopted** (slower to converge, fewer FPs but fewer stones) |
| E7 | RF-DETR-L 704 on `dataset_grouped_neg` (RF baseline with negatives) | done (103 min, best EMA epoch 16) | valid mAP50:95 0.443 [0.25, 0.58], mAP50 0.683 [0.44, 0.83]; **image AUROC 1.000, 0/21 negatives with FP**; stone recall 0.71, 94% of stone images hit; FP/img 0.84 (all on stone images). Current best model |
| E9 | UIdataGB check (Mendeley r6h24d2d3y, CC BY 4.0) | done | gallstone class = **our `a*` videos** (30 of its 32 patients; 694 of our images have a near-copy; heavy augmented copies per patient) -> no new stone data |
| E10 | Polyp hard negatives: UIdataGB "Polyps and cholesterol crystals", 1,020 images / 22 patients (grouped by patient), no overlap with our data -> `dataset_grouped_neg2` (positives and earlier negatives unchanged, leak check clean) | done: `rf704_neg2` (163 min, best EMA epoch 7) | **E7 on neg2 valid: image AUROC 1.00 -> 0.58, 86% of polyp/negative images get a false positive (2.4 per image), mAP50:95 0.443 -> 0.266.** The model cannot tell polyps from stones. **Trained with polyps (`rf704_neg2`), on neg2 valid:** polyp images with FP 97.5% -> 24%, AUROC stone vs polyp 0.53 -> 0.85, mAP50:95 0.266 -> 0.384 [0.13, 0.54]; normal GB still 0% FP. **Cost on stone images** (original valid): mAP50:95 0.443 -> 0.409, stone recall 0.71 -> 0.66, stone images hit 94% -> 82%. Trade-off: far better specificity, some sensitivity lost. Adopted as the new default (polyps are the realistic clinical confounder); recall to be recovered next |
| E11 | RF-DETR kidney pretrain (8 ep, 129 min, kidney valid 0.136) -> gallstone fine-tune on neg2 | **stopped by decision** at epoch 5 | learned slower than COCO start (0.289 vs 0.372 at epoch 4); replaced by joint training (E12) |
| E12 | **Joint training** `dataset_joint`: 2 classes (gallstone, kidney_stone); gallstone positives x3, gallstone negatives x2 (exact repeats, online flip/resize), all 5,431 kidney images in train; valid/test = gallstone `neg2` split unchanged (leak check clean); RF-DETR-L 704, 25 epochs, patience 6, batch 3; scored with `--keep-class 0` | **stopped at epoch 7** (plateau; best EMA epoch 6, 0.351) | neg2 valid (EMA ckpt): mAP50:95 0.344 vs 0.384 gallstone-only (E10); stone recall 0.64 vs 0.66; polyp images with FP 43% vs 24%; AUROC vs polyp 0.79 vs 0.85; normal GB 0% FP. **Worse on every axis; not adopted.** Likely the ~5.4k kidney images per epoch pull the model toward kidney appearance |
| E13 | **Grouped 5-fold CV** of the E10 default (`dataset_cv`, built from train+valid of `dataset_grouped_neg2`, test excluded; near-duplicate groups joined across folds; each fold leak-checked) | done (fold 1 crashed twice and fold 3 once from low virtual memory / full disk: Windows grew the page file; fixed with `--workers 1`; fold 3 stopped at epoch 27 by a full disk, scored from its best EMA checkpoint, so slightly underestimated) | **Out-of-fold, 2,024 images / 228 groups: mAP50:95 0.421 [0.359, 0.496], mAP50 0.622 [0.552, 0.719]**; stone recall 0.70, 88% of stone images hit; normal GB 0% FP (AUROC 0.99); polyp images with FP 36% (AUROC vs polyp 0.86); FP/img 0.69. CI half-width ~0.07 vs ~0.15 on the fixed valid split. Per-fold best EMA: 0.420 / 0.455 / 0.400 / 0.513 / (fold 4 in pooled). This is the reference number for the default |
| E14 | Label audit from E13 out-of-fold predictions (`scripts/label_audit.py`, output `runs/cv/rf704/label_audit/`) | done | 964/2,024 images with some disagreement (375 with missed labels, 340 with confident extra boxes, 110 cluster/instance mismatches, 168 negatives with confident FP). **Video `a32` (train, 112 images) holds 437 boxes = 27% of all training boxes, and 202 of its labels are missed**: it labels every faint echogenic dot in the lumen (often no shadow; microliths vs. sludge/polyp ambiguous) while the model only fires on the brightest. a32 also mixes several patients/scanners under one ID. Convention A chosen by the user (2026-10-04) |
| E15 | **Convention A** (`scripts/apply_label_rule.py`, model-independent): a box is turned into an ignore region if it is not its image's brightest and has peak < 110, contrast < 30 or < 0.45 x the image's max contrast (features: `scripts/stone_box_features.py`). Applied to all splits -> `dataset_grouped_neg3`: train 1,340 kept / 288 ignored (a32: 180 of 437), valid 270/34, test 412/27. Ignore regions are COCO iscrowd in `coco_map` (unit-tested) and excluded from recall/FP counts. Visual check: removals are mostly faint satellites and wall echoes; ~1 in 4 sampled removals debatable (dim stones in clusters), hence ignore rather than negative in evaluation | E10 model re-scored; training `rf704_neg3` running | E10 model on neg3 valid (no retraining): mAP50:95 0.384 -> 0.426, mAP50 0.562 -> 0.624, stone recall 0.66 -> 0.73 (measurement change only) |
| E16 | **Data clean-up after teammate review** (verified by rendering every affected image and an OCR audit, `scripts/text_audit.py`): (1) removed all 54 `pl_*` images (web-scraped: Elsevier/alamy/Getty watermarks, answer text such as "Cholelithiasis"/"Stone"/"GALLBLADDER", a patient-name overlay, some not gallstones); (2) removed 2 full-frame bad boxes (`pseudo_c29__28` liver only, `pseudo_c26__57` mass); (3) held out test video `a6` (whole-gallbladder boxes) until relabelled; (4) OCR found scanner text on 106/136 normal-GB negatives ("Voluson E6") vs. almost no positives -> **text shortcut**; all burned-in text is now inpainted in every image (`src/preprocessing/text_removal.py`, labelled stones protected). Exclusions documented in `docs/data_exclusions.json`; build `scripts/curate_dataset.py` -> `dataset_grouped_neg4` | neg4 built (kept 1,598 train / 380 valid / 287 test; excluded 54 pl_*, 2 bad boxes, 19 a6; text inpainted in 233 images), leak check clean, `dataset_cv4` folds built. CV running: fold 0 stopped at epoch 32 (best EMA epoch 27, 0.449) to avoid exhausting virtual memory (system commit free 1.2 GB, page file 23 GB). Correction: not a leak - the RF-DETR training process holds ~8.5 GB committed memory from the start, each dataloader worker ~1.7 GB more, and other software (vmmem 4 GB, Claude app, Chrome) the rest of the ~32 GB commit limit. Folds 1-4 rerun with `--workers 0` (frees ~3.5 GB, epochs ~40% slower) and patience 10 | Note: the earlier "0% FP on normal GB / AUROC 0.99" is likely inflated by the scanner-text cue; normal-GB negatives also differ in scanner/sector style, which text removal cannot fix -> report normal-GB results as a limitation, polyp negatives (same source family as positives) are the more trustworthy specificity test |
| E17 | **5-fold grouped CV on `dataset_grouped_neg4`** (convention A + clean-up + text removal), RF-DETR-L 704, batch 3; fold 0 workers 1 / stopped at epoch 32, folds 1-4 workers 0, patience 10 | done (per-fold best EMA 0.449 / 0.464 / 0.400 / 0.513 / 0.439) | **OOF, 1,978 images / 184 groups: mAP50:95 0.454 [0.390, 0.512], mAP50 0.658 [0.575, 0.729]**; stone recall 0.73, 87% of stone images hit; polyp images with FP 28% (AUROC vs polyp 0.87); normal GB 0% FP (AUROC 0.98, still a likely scanner-style cue); FP/img 0.53. vs E13 (neg2 data): 0.421, recall 0.70, polyp FP 36%, FP/img 0.69 - not strictly comparable (labels and ignore regions differ), but the specificity gains hold on the same polyp images. **New reference for the default** |
| E18 | **Final model + one-time test** (protocol pre-registered in `docs/FINAL_EVAL_PROTOCOL.md`) | done | Primary (RF-DETR-L 704, train+valid of neg4, 20 epochs, 162 min): **test mAP50:95 0.369 [0.230, 0.563], mAP50 0.583 [0.321, 0.721]**, recall 0.65 @0.25, 96% stone images hit, FP/img 0.71, polyp images with FP 40%, AUROC 0.90 (vs polyp 0.88). Secondary (5-fold WBF ensemble): mAP50:95 0.400, mAP50 0.665, AUROC 0.95 (vs polyp 0.94); its threshold metrics are invalid (WBF avg lowers fused scores below 0.25). Test has only 6 stone videos -> wide CIs. Test is now used; further changes need a new test set |

Note on E5: the old models score lower on the new valid images (0.280 vs 0.394)
because they were trained on the first overlay-removal version (colour kept,
smooth fill). Every model must be retrained on the current preprocessing.

## Dead ends

- `data/processed/` holds no new labelled frames: 80 stems not in the current
  dataset, 49 of them review-rejected, the rest frames of videos already in
  the split.
- `dataset_new/Patient_Wise`: only 3 cholelithiasis patients, unlabelled.
