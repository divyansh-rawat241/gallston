# Literature survey: gallstone / gallbladder ultrasound detection (2026-10-03)

Source: web research agent. Several publisher pages blocked access; items
marked "unverified" could not be checked.

## Headline

- No published gallstone ultrasound detector reports COCO mAP50:95 on a
  patient/video-level split. The only gallstone paper with mAP50:95 (Kumar
  2025, DETR) used a **random image split** and got 0.32 (mAP50 0.43).
  Our leak-checked video split is a genuine differentiator: lead the paper
  with it and report a random-split number alongside to show leakage inflation.
- Closest comparable: OGLMFI (2026, Ultrasound Med Biol; liver/GB/kidney
  lesions, video test set, YOLOv11 multi-frame): mAP50 0.703, mAP50:95 0.510.
  Split level unverified; not gallstone-specific.
- No public gallstone detection benchmark exists (stated by Kumar 2025).

## Comparison table

| # | Paper | Year/venue | Data | Split | Metric | Number | Leak risk / comparable |
|---|---|---|---|---|---|---|---|
| 1 | Kumar et al., DETR for cholelithiasis | 2025 CSBJ, PMC12595354 | 1,210 imgs / 263 pts, private | random image-level | COCO mAP50, mAP50:95 | DETR-R50 0.43 / 0.32; YOLOv8 0.38/0.24; RT-DETR 0.40/0.30 | high; same metric |
| 2 | Kumar et al., explainable CNN | 2025 Sci Rep, PMC11794719 | same + 710 DCGAN synthetic in all splits | image-level | accuracy/AUC | 84%, AUC 0.87 | high; classification |
| 3 | Yu et al., POCUS lightweight DNNs | 2021 CMPB | 89k imgs / 2,386 pts | unverified | AUC, "AP 86.95%" | AUC 0.92 | unknown; VOC-style AP |
| 4 | Wang et al., OGLMFI | 2026 UMB, doi 10.1016/j.ultrasmedbio.2026.07.002 | 205 test videos | unverified | mAP50, mAP50:95 | 0.703 / 0.510; +8.1 over YOLOv11-L | closest comparable |
| 5 | Ge et al., single-view cines | 2025 Acad Radiol | 186 pts | unverified | accuracy | 91% | classification |
| 6 | Obaid et al., UIdataGB 9-class | 2023 Diagnostics | 10,692 imgs / 1,782 pts, public, no boxes | patient-level | accuracy | 98.35% | classification |
| 7 | Veena et al. | 2022 | 60 imgs | 48/12 | mAP | n/a | tiny |
| 9 | Kim & Kim, rule-based stone + shadow | 2026 J Imaging, PMC13117089 | 66 GB + 52 kidney | n/a | accuracy | GB 0.30 -> 0.60 with shadow term | prior art (shadow) |
| 10 | GB-YOLOv7 (cancer) | 2026 EAAI | unverified | unverified | mAP | ~94% (unverified) | cancer task |
| 11 | Mayya & Alkayem, YOLOv10+v11 ensemble, kidney | 2026 JIIM | 3,889 US imgs | unverified | mAP50 | +1.34 ensemble | two-YOLO ensemble already published |
| 12 | Hajihashemi, systematic review | 2026 Abdom Radiol | 13 studies | — | — | US acc 71.5-99.6% | "lack of robust external validation" |

## IIT Delhi gallbladder work (Basu, Gupta, Arora)

- GBCNet + GBCU (CVPR 2022, arXiv 2204.11433): 1,255 imgs / 218 pts,
  patient-level split; GB-ROI-first then classify. **GBCU has stone boxes** ->
  possible external test set (licence: permanent faculty/staff only).
- RadFormer (MedIA 2023, arXiv 2211.04793): global ROI + local attention.
- Video contrastive learning with hard negatives (MICCAI 2022, arXiv 2207.13148).
- Weakly supervised DETR for GB cancer (MICCAI 2023, arXiv 2309.05261): AP 0.628
  with image-level labels.
- LQ-Adapter (arXiv 2412.00374): ViT-Adapter + learnable queries.
- FocusMAE (CVPR 2024, arXiv 2403.08848): 91 videos / 73 pts, patient-wise
  5-fold CV; RPN-guided masking.

## Prior art for our novelty claims

| Claim | Risk | Closest prior art | Framing |
|---|---|---|---|
| Shadow-aware fusion (WBF modulated by posterior-shadow prior) | medium | Kim & Kim 2026 (shadow term in rule score); **US9597103B2** (U. Washington, bright spot + distal shadow adjacency, active to ~2031); Meng shadow confidence maps (arXiv 1811.08164); SABR-Net | claim learned multi-detector fusion with a shadow likelihood measured below each fused box (learned/calibrated), not "shadow to detect stones". **Freedom-to-operate check on US9597103 before filing** |
| Two-detector WBF ensemble | high | WBF 2019; kidney YOLO ensemble 2026 | base only |
| Per-detector calibration | high | Kuppers CVPRW 2020 | supporting |
| Conformal box / recall guarantees | medium-high | Andeol 2023 (PMLR v204); ECCV 2024 arXiv 2403.07263; SeqCRC arXiv 2505.24038; Bonferroni box CP arXiv 2605.07549; **Hulsman COPA 2025 arXiv 2412.20167 (medical miss-rate control)** | novelty from ultrasound **video**: video-grouped calibration / exchangeability, study-level guarantees, interaction with shadow-fused scores |
| Cross-architecture XAI agreement | medium | COSE arXiv 2309.10989 | analysis, not core claim |
| Leak-free video-grouped benchmark, COCO mAP50:95 | low (strength) | none for gallstones | lead with it |

No paper found that uses a shadow prior to rescore a fusion of learned
detectors, nor conformal detection on ultrasound video with video-level
calibration.

## Reported small-lesion gains in ultrasound

- Temporal/multi-frame: OGLMFI +8.1 mAP50:95; FA-DETR +3.8 AP (MIDL 2025);
  STNet (arXiv 2309.04702); CVA-Net + public 188-video breast set (arXiv 2207.00141).
- Organ-ROI priors: GBCNet, FocusMAE, OGLMFI organ-guided filtering.
- Domain pretraining: UltraDINO (arXiv 2506.19552); US foundation models beat
  DINOv3 on fetal planes (classification only, arXiv 2605.27796).
- Augmentation: Tupper & Gagne TMLR 2025 (arXiv 2501.13193), cls/seg only.

## Methods worth trying (gain / effort)

1. Temporal score aggregation at inference (Seq-NMS-style linking); no retraining.
2. Gallbladder-ROI-first detection (crop at higher effective resolution).
3. External test on GBCU stone boxes (needs faculty sponsor).
4. TrivialAugment-style augmentation sweep.
5. Higher resolution for RF-DETR (ultrasound evidence unverified).
6. Continued DINOv2 SSL on unlabelled frames (+ temporal hard negatives).
7. Weak supervision from image-level stone labels (only if many such studies).

Unverified: split levels for Yu 2021, OGLMFI, GB-YOLOv7, kidney ensemble;
GB-YOLOv7 numbers; full text of the review and Lancet SE Asia study; SAHI,
Karamalis, RF-DETR paper (cited from memory).
