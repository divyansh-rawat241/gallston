# Ultrasound pretrained / foundation models (research notes, 2026-10-03)

Source: web research agent plus the installed `rfdetr` 1.7.1 source. No weights
downloaded. Runtime estimates are not measured.

## Bottom line

No ultrasound foundation model drops into RF-DETR-L. Best options:
1. continued self-supervised pretraining of RF-DETR-L's *own* encoder on our
   ~13k unlabelled ultrasound images, written back into the checkpoint;
2. pseudo-labelling + hard-negative mining on those images with our detectors.
Expect a few mAP points at most -> confirm with grouped CV.

## RF-DETR-L backbone facts (from rfdetr source)

- `RFDETRLargeConfig`: `dinov2_windowed_small` = ViT-S, 384 dim, 12 layers,
  6 heads, **patch 16**, 704 px, learned absolute pos-emb (44x44), 2 windows,
  out layers [3,6,9,12], no registers; weights `rf-detr-large-2026.pth`.
- Backbone weights come only from the RF-DETR checkpoint (patch 16 disables
  loading HF `facebook/dinov2-*`).
- Custom weights route: `models/weights.py:load_pretrain_weights` accepts
  `{"model": state_dict}`; missing keys warn, shape mismatches fail. Encoder
  keys follow HF DINOv2 naming; inferred prefix `backbone.0.encoder.encoder.`
  (verify by printing keys).
- Compatible encoder must be ViT-S/16, 384 dim, abs pos-emb, LayerScale, no
  registers/RoPE. Swapping backbones loses the COCO encoder-decoder pairing
  (DINOv3 swap reported worse, rf-detr issue #498).

## Models

| Model | Data | Arch | Weights / licence | Fits RF-DETR-L | Use |
|---|---|---|---|---|---|
| USFM (github.com/openmedlab/USFM) | ~2M US images, 12 organs incl. GB | BEiT ViT-B/16, rel-pos | open, CC-BY-NC 4.0 | no | distillation teacher; freq-masking idea |
| TinyUSFM | 200k coreset distil | 5.5M params | Apache-2.0 | no | low |
| UltraDINO (github.com/jakobamb/UltraDINO) | 2M fetal US | DINOv2 ViT-S/16 & B/16 | **no weights found** | would fit | evidence that domain DINOv2 SSL beats USFM |
| URFM (iScience 2025) | US, MIM guided by BiomedCLIP | ViT-B/16 | repo URL not found | no | teacher at most |
| UltraFedFM | federated US MAE | ViT (likely B/16) | Apache-2.0 | no | low |
| EchoCare (github.com/CAIR-HKISI/EchoCare) | 4.5M images, 52 organs | Swin | SharePoint, licence unclear | no | teacher, high effort |
| OpenUS | 308k images | VMamba-S | CC-BY-NC 4.0 | no | low |
| UltraSam (github.com/CAMMA-public/UltraSam) | 280k masks, 43 datasets incl. AbdomenUS | SAM, mmdet, box-refine config | open download, licence unclear | n/a | box->mask, box tightening |
| SAMUS | US30K (no GB) | SAM ViT-B + CNN | MIT | n/a | low |
| MedSAM2 (hf wanglab/MedSAM2) | 3D + video incl. liver/heart US | SAM 2.1 | CC-BY-SA research | n/a | box->mask, video propagation |
| UltraSAM3 (hf zhuqh19/UltraSAM3) | 37 datasets, 13 categories | SAM3 text prompts | Apache-2.0 | n/a | GB-region gating; GB/stone concept unverified |
| BiomedParse | many; GB only CT/MRI | text-prompt seg | gated, NC | n/a | low |
| FreqDINO/++ | US seg adapters | DINOv3 + wavelets | code | no | idea only |
| DINOv3 ViT-S/16 | natural | RoPE + registers | gated | no | skip |
| RAD-DINO | chest X-ray | ViT-B/14 | MIT | no | skip |

## Ranked options

1. **Continued DINO/iBOT SSL of RF-DETR-L's ViT-S/16 encoder** on 13k
   unlabelled + kidney images + gallstone *train* images (exclude every valid/
   test video). US-appropriate aug (grey, crops 224/96, gamma/gain, TGC ramp,
   speckle, small rotation, hflip; no colour jitter). bf16 + grad checkpointing;
   AdamW lr ~5e-5, layer decay 0.8, 30-50 epochs (est. 6-12 h), feature-
   retention L2 to the original encoder to limit decoder drift; write encoder
   back into a copy of `rf-detr-large-2026.pth`; then kidney (E6) -> gallstone.
2. **Teacher pseudo-labelling + hard-negative mining** on the 13k images
   (leak-filtered); thresholds chosen on valid; optional GB-region gate via
   UltraSAM3 / MedSAM2; negatives are the main value.
3. **GBCU stone boxes** (licence route already documented).
4. **UltraSam / MedSAM2 box->mask / box tightening** for label QA.
5. Feature distillation from USFM / EchoCare (high effort).
6. ViT-B foundation backbone swap (likely negative with 830 images, 6 GB).

## Not verified

URFM repo URL; UltraDINO weights; UltraSam licence/backbone size; UltraSAM3
base size and GB/stone concepts; SonoSAM/SonoBase weights; EchoCare licence;
exact checkpoint key prefix; all runtimes.
