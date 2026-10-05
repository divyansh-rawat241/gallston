# Public data sources (research notes, 2026-10-03)

Source: web research agent (pages read 2026-10-02/03). Nothing downloaded.

## Overlap warning

Roboflow workspace `gallstone-kbw07` ("GallStone") has `gallstones_pseudo`
(1.26k), `gallstone_v3` (593; hosted RF-DETR-small / YOLO26n), `gallstone (v1)
using YOLOV 11` (826), `Gallstone-Detection` (462), `gallbladder-anomalies`
(66). The "pseudo" naming matches our `pseudo_<video>__<frame>` files: very
likely our own source -> **not external data**. `amlesh-thakur.../gallstone-v1-
using-yolov-11` (826) and `sushiltamang/gallstone` (827) look like copies.

## Candidates

| # | Name | URL | Content | Labels | Licence | Access | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **UIdataGB** | data.mendeley.com/datasets/r6h24d2d3y/1 | 10,692 GB US / 1,782 pts (Iraq, multi-vendor); gallstones 1,326 imgs / 221 pts; + polyps, cholecystitis, carcinoma ... | class only | CC BY 4.0 | direct | yes - augmented copies inside: pHash-dedupe, group by cluster, pseudo-box + review |
| 2 | Roboflow UIdataGB "parts" | universe.roboflow.com/gallbladder-2nd-part/2nd-part-gallbladder (+3rd/4th/5th) | 1,413 (gallstones, perforation, polyps); 990; 994; 1.17k | boxes, unknown annotator | CC BY 4.0 | fork | yes, check box quality; polyp / wall boxes = hard negatives |
| 3 | small Roboflow GB sets | zoha/gallbladder-stone-new (404), lifeguard-50/ai-gallbladder-project (365), gallstones/gallstone-special-problem (137), datasets-i3hnb/find-gallbladder (544, PD), my-datasets-k2sei/gallbladder-stones (38) | ~1.5k | boxes/masks | CC BY / PD | fork | maybe: near-dup check vs our videos and #1 first |
| 4 | gallstone-ultrasound-classifier (GitHub) | github.com/gitcommit-computerphile/gallstone-ultrasound-classifier | 159 frames, 136 pts | class | none stated, PHI noted | repo | low |
| 5 | GBUSV v1/v2 | gbc-iitd.github.io/data/gbusv | 64 / 91 GB videos | video-level | signed agreement | faculty request (with GBCU) | SSL / negatives |
| 6 | AURORA-GB | pmc.ncbi.nlm.nih.gov/articles/PMC12992070 , forms.gle/xQaqUbGMEzaCr2Br7 | 8,048 GB imgs / 674 pts | patient-level | not stated | request form | pretraining / negatives |
| 7 | Kidney US stone/normal (Punjabi Univ.) | data.mendeley.com/datasets/h6jc4xm4py/1 | 9,416 (5,002 stone) | class | CC BY 4.0 | direct | pretraining after pseudo-boxing; dedupe vs kidney-ktlmt |
| 8 | URI-CAD | doi.org/10.21950/0E0CTF | 1,985 renal US | kidney mask + **expert lesion boxes incl. lithiasis** | CC BY-NC 4.0 | gated file | yes for 2nd-stage stone pretraining (non-commercial) |
| 9 | Roboflow kidney clones | jimma-university-1, selamina, sel, fyp-npqts/ultrasound-u40t9 ... | — | boxes | CC BY | fork | mostly copies of kidney-ktlmt; many "kidney stone" sets are CT |
| 10 | Abdominal US Image Dataset (MSU) | scholarsjunction.msstate.edu/research-data/5/ | 5,468 imgs / 563 pts, 10 organs incl. GB | organ + normal/abnormal | CC BY 4.0 | direct | **normal-GB negatives** + SSL |
| 11 | AbdomenUS | kaggle.com/datasets/ignaciorlando/ussimandsegm | 617 real + 926 synthetic | organ masks | CC-type | direct | small |
| 12 | AUL liver | zenodo.org/records/7272660 | 735 | liver/mass masks | CC BY 4.0 | direct | small SSL |
| 13 | Open Kidney US | rsingla.ca/kidneyUS | 514 | kidney polygons | CC BY-NC-SA | registration | SSL only |
| 14 | Regensburg appendicitis | zenodo.org/records/7669442 | ~2.1k | patient labels | CC BY-NC 4.0 | direct | SSL filler |
| 15 | Butterfly MITGrandHack2018 | github.com/ButterflyNetwork/MITGrandHack2018 | 41k imgs / 31 people | view class | repo terms | direct | SSL only |
| 16 | AHU database | figshare 26889334 | 1,833 items, 224 px | class | CC BY-NC-ND | direct | low |
| 17 | Pretraining corpora | OpenUS-46, OpenUS, SonoCorpus, UltraBench | 308k-457k | mixed | per-source | assemble | prefer released checkpoints |
| 18 | POCUS Atlas hepatobiliary | thepocusatlas.com/hepatobiliary | tens of clips | captions | CC BY-NC | website | demo / external cases |

Avoid: Kaggle `aneerbansaha/gallbladder-cancer` (probably a GBCU re-upload
under "MIT" - would bypass the GBCU licence; inferred, not confirmed);
lonlonago paid kidney set (resale of Roboflow data). Not relevant: CT and
tabular gallstone sets.

Not found: bladder/salivary stone US datasets, any gallstone US challenge.

## Top 10

1. UIdataGB gallstone class (#1)
2. Roboflow UIdataGB boxed parts (#2) - boxes + hard negatives (polyps, walls)
3. Normal-GB negatives: MSU (#10) + non-stone UIdataGB classes
4. URI-CAD expert stone boxes (#8)
5. Punjabi kidney stone set (#7)
6. GBUSV v2 + GBCU in one faculty request (#5)
7. AURORA-GB (#6)
8. Non-workspace Roboflow GB sets (#3), after dedupe
9. Public US foundation checkpoints (#17)
10. Small external sets (#4, #18, #12)
