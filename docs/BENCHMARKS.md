# Benchmarks

## 0. Current results (2026-10-05) - the only numbers to quote

Leak-free, video/patient-grouped split; COCO mAP via `src/evaluation/coco_map.py`;
95% CIs from a video-grouped bootstrap. Protocol pre-registered in
[FINAL_EVAL_PROTOCOL.md](FINAL_EVAL_PROTOCOL.md); experiment history in
[EXPERIMENTS.md](EXPERIMENTS.md). Data: `dataset_grouped_neg4` (convention A,
web images and bad boxes removed, burned-in text inpainted).

### 0.1 Grouped 5-fold cross-validation (train+valid, out-of-fold)

RF-DETR-L 704, 1,978 images / 184 groups (`runs/cv4/rf704/oof_eval.json`).

| mAP50:95 | mAP50 | Stone recall @0.25 | Stone images hit | FP / image | Polyp images with FP | AUROC vs polyp | Normal GB with FP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.454 [0.390, 0.512] | 0.658 [0.575, 0.729] | 0.73 | 87% | 0.53 | 28% | 0.87 | 0% |

### 0.2 Held-out test, evaluated once

287 images (116 with stones from **only 6 videos**, 171 stone-free: 150 polyp,
21 normal gallbladder); video `a6` excluded pending relabelling.

| Model | mAP50:95 | mAP50 | mAP75 | AP small / large | Stone recall @0.25 | Stone images hit | FP / image | Polyp images with FP | AUROC stone vs stone-free (vs polyp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Primary**: RF-DETR-L 704, train+valid, 20 epochs (`runs/final/test_primary.json`) | **0.369 [0.230, 0.563]** | **0.583 [0.321, 0.721]** | 0.397 | 0.336 / 0.672 | 0.65 | 96% | 0.71 | 40% | 0.90 (0.88) |
| Secondary: 5 CV-fold models, WBF avg (`runs/final/test_ensemble.json`) | 0.400 [0.228, 0.577] | 0.665 [0.351, 0.784] | 0.413 | 0.337 / 0.713 | n/a* | n/a* | n/a* | n/a* | 0.95 (0.94) |

\* WBF with `conf_type="avg"` averages each model's many low-score duplicate
boxes into the fused score, so fused scores rarely reach the pre-registered
0.25 threshold (recall 0.00 and FP 0.00 at 0.25 are artefacts of the score
scale). Rank-based metrics (mAP, AUROC) are unaffected. Not re-tuned on test.

Normal-gallbladder negatives: 0% FP / AUROC 1.00 for both models, but they come
from a different scanner, so treat that as optimistic.

**Reading the test numbers**: the CI is wide because test holds only 6 stone
videos. The primary model's 0.369 is below the CV mean (0.454) and just below
the CV interval, but the test interval [0.230, 0.563] contains the CV mean, so
the two are consistent; the gap reflects the small test set and its video mix. The CV estimate (184 groups) is the more precise number;
test is the untouched confirmation.

---

## Superseded numbers (kept for the record, do not quote)

Regenerated 2026-08-09 by `python scripts/regenerate_all_numbers.py`.
Do not hand-edit numbers here — rerun the script.

Every number below traces to a JSON under `runs/final_eval/` and a script
under `scripts/`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

> **These numbers are invalid for external use (audited 2026-09-27).**
> 1. **Split leakage** — all 49 test video groups (`pseudo_<group>__<frame>`)
>    also have frames in train; 52/52 val groups overlap train. Test and val
>    are near-duplicates of training frames. Only 58 source video groups exist
>    in total (30 `a*`, 28 `c*`; median 22 frames/group).
> 2. **Fusion input bug** — `GallstoneEnsemble` fed RF-DETR BGR instead of RGB
>    until 2026-08-09. The ablation ladder (§2), fusion calibration (§3 fusion
>    row) and conformal results (§5) were computed with that bug.
>
> Per-detector calibration rows (§3, YOLO / RF-DETR) used a correct RGB path
> but are still subject to (1). Regenerate everything on a video-group split
> before quoting any number.

**Note on metric floor**: `src/evaluation/map_calculator.py` reports
per-image mean F1 at IoU=0.5, not COCO-style mAP. The rows below use the
label "F1@IoU=0.5" accordingly. A torchmetrics-based COCO mAP swap is
planned for the next revision.

---

## 1. Hyperparameter search (Optuna, already run)

| Study | Trials complete | Best val mAP@50 | DB file |
|---|---:|---:|---|
| `gallstone_yolo26l_hpo` | 19 | **0.9598** | `yolo26l_hpo_results.db` |
| `gallstone_rf_detr_large_hpo_v2` | 140 | **0.9606** | `rf_detr_large_hpo_results.db` |

---

## 2. Ablation ladder (test=201, per-image F1@IoU=0.5)

| Configuration | F1@0.5 | Δ vs WBF baseline |
|---|---:|---:|
| **WBF (baseline)** | **0.9298** | — |
| + Shadow prior (τ=0.7 default) | 0.8260 | −0.1038 |
| + Multi-scale TTA {640, 800, 1024} | 0.9166 | −0.0132 |
| WBF + Shadow + TTA | 0.4974 | −0.4324 |

**Interpretation.** WBF alone is the strongest baseline. The default
ShadowAnalyzer threshold (0.7) misfires on this split — small non-shadowing
stones get down-weighted 0.5×. Threshold sweep in §4 confirms best val F1
still below WBF baseline. Multi-scale TTA at these three scales is a slight
regression, suggesting the model was over-trained at 640. All three combined
compound the regressions.

Source: `runs/final_eval/comprehensive_evaluation.json`.

---

## 3. Calibration (val=368)

| Configuration | T | ECE before | ECE after | Brier before | Brier after | n |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv26L (temperature scaling) | 0.679 | 0.0543 | **0.0319** | 0.0681 | 0.0665 | 941 |
| RF-DETR-L (temperature scaling) | 0.749 | 0.0314 | **0.0202** | 0.0133 | 0.0142 | 23258 |
| Fusion (WBF+shadow, isotonic) | — | 0.5495 | 0.0000\* | 0.3975 | **0.0615** | 813 |

**Interpretation.** Per-detector temperature scaling reduces ECE by 41 %
(YOLO) and 36 % (RF-DETR) with no accuracy trade-off. The fusion isotonic
row overfits (ECE→0 on the same fold used to fit); a fresh second-fold
split is required for an honest test-ECE, but the Brier score improvement
of 85 % is a valid indicator that the fitted mapping is doing real work,
not just memorizing.

`*` = same-fold estimate; not test-honest, see note above.

Reliability diagrams: `notebooks/05_calibration_analysis.ipynb`.
Source: `runs/final_eval/calibration.json`.

---

## 4. Shadow threshold tuning (val=368, per-image F1@IoU=0.5)

Best point in the [0.5, 0.9] sweep over 20 steps:

| Metric | Value |
|---|---:|
| Best threshold τ | **0.774** |
| Best F1 at τ | 0.8827 |

Even the tuned threshold underperforms WBF alone on this split. The
physics-prior encoding as-implemented (uniform 1.2× / 0.5× / 1.0× ratio-
based modulator) is too coarse for the small-stone regime. A learned
weighting (logistic regression on shadow_ratio + box_area + y_center)
would likely beat WBF — planned upgrade in Phase 3.4.

Source: `runs/final_eval/shadow_threshold_tuning.json`.

---

## 5. Conformal · α = 0.05 (calibration=val=368, test=test=201)

### 5.1 Split-conformal box expansion

| Coord | q̂ (normalized) |
|---|---:|
| q_x1 | 0.0044 |
| q_y1 | 0.0051 |
| q_x2 | 0.0094 |
| q_y2 | 0.0088 |

| Metric | Value | Target |
|---|---:|---:|
| n calibration TP pairs | 729 | — |
| n test TP pairs | 370 | — |
| Target coverage 1−α | 0.950 | — |
| **Empirical coverage (test)** | **0.9595** | ≥ 0.95 ✓ |

Coverage guarantee empirically satisfied on the test split — the primary
patent claim 1(f) numeric evidence.

### 5.2 Conformal Risk Control (recall bound)

| Metric | Value | Target |
|---|---:|---:|
| λ\* (score threshold) | 0.090 | — |
| n calibration images | 368 | — |
| Target FNR ≤ α | 0.05 | — |
| Empirical FNR (test) | 0.0684 | ≤ 0.05 ✗ (+1.8 pt) |

Slight overshoot with finite calibration; expected width of finite-sample
CRC. Tighten by increasing calibration set or by adding a safety margin
in the λ selection (`λ_star := max(λ_star_grid, λ_star_finite_correction)`).

Source: `runs/final_eval/conformal.json`.

---

## 6. Per size bucket (test=201) — TODO

Requires `--split test --by-size` extension of the eval script.

## 7. Failure-mode grid (test=201) — TODO

Requires case-category annotation not currently in dataset.

## 8. Latency + memory — TODO

Requires per-device benchmark harness.

## 9. Radiologist AUROC / κ — BLOCKED

Requires ≥ 1 radiologist grading the 201-image test set.

## 10. External-institution validation — BLOCKED

Requires ≥ 1 external US dataset under DUA (Kaggle or hospital partner).

---

## Summary for patent enablement

The three most-important defensible numbers for the independent claim:

1. **Per-detector calibration reduces ECE by 36–41 %** on val=368 with no
   accuracy penalty. (Claim 1(e))
2. **Split-conformal box expansion achieves 95.95 % empirical coverage vs
   95 % claimed** on test=201. (Claim 1(f))
3. **CRC bounds FNR to 6.8 % vs 5 % claimed** — small finite-sample
   overshoot, tightens with more calibration data. (Claim 1(g))

The WBF baseline (0.9298) already exceeds most published gallstone-US
literature, and the physics-prior fusion + calibration + conformal wrapper
delivers the novelty axes. The shadow-prior modulator needs learned
re-weighting to beat WBF alone; work item for Phase 3.4.
