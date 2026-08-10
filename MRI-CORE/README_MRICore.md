# MRI-CORE — Feature Extraction for FeTS2022/2024

MRI-CORE is a **2D foundation model** built on SAM (Segment Anything Model) ViT-B,
pretrained on large-scale multi-domain MRI data using DINO self-supervised learning.

- **Paper:** [MRI-CORE: A Foundation Model for Magnetic Resonance Imaging](https://arxiv.org/abs/2404.09957)
- **Model repo:** https://github.com/mazurowski-lab/mri_foundation

Because MRI-CORE is a 2D model, each 3D volume is represented by a single
**peak-tumor axial slice** — the slice where the segmentation mask has the most
non-zero voxels (necrosis + enhancing tumor + edema combined).

---

## 1. Setup

### 1.1 Clone MRI-CORE model code

```bash
git clone https://github.com/mazurowski-lab/mri_foundation.git mri_foundation
```

The extraction script expects the cloned repo to be in a folder named `mri_foundation/`
in the same directory as `extract_mricore_fets24.py`.

```
Radiology_features/
├── mri_foundation/          ← cloned repo goes here
│   ├── models/
│   ├── cfg.py
│   ├── requirements.txt
│   └── pretrained_weights/
│       └── mri_foundation.pth
├── extract_mricore_fets24.py
└── ...
```

### 1.2 Create conda environment

```bash
conda create -n mricore python=3.12 -y
conda activate mricore
```

### 1.3 Install PyTorch

```bash
# CUDA 12.4 — adjust the index URL for your CUDA version
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

### 1.4 Install remaining dependencies

```bash
pip install -r mri_foundation/requirements.txt
pip install nibabel pandas tqdm
```

### 1.5 Download pretrained checkpoint

Download `mri_foundation.pth` from the
[Google Drive link in the MRI-CORE README](https://github.com/mazurowski-lab/MRI-CORE#download-pre-trained-models-step-1)
and place it at:

```
mri_foundation/
└── pretrained_weights/
    └── mri_foundation.pth
```

---

## 2. FeTS2022/2024 Data Structure

```
MICCAI_FeTS2022_TrainingData/
├── FeTS2022_00000/
│   ├── FeTS2022_00000_flair.nii.gz
│   ├── FeTS2022_00000_t1.nii.gz
│   ├── FeTS2022_00000_t1ce.nii.gz
│   ├── FeTS2022_00000_t2.nii.gz
│   └── FeTS2022_00000_seg.nii.gz   ← used for slice selection (training only)
├── FeTS2022_00002/
│   └── ...
└── ...   (1251 training patients, 219 validation patients)
```

Data is available on [Synapse](https://www.synapse.org/Synapse:syn54079892).
Volumes are co-registered to the SRI24 atlas, skull-stripped, 1 mm isotropic.
Intensities are **raw scanner values** (not normalized) — the script handles
normalization internally.

---

## 3. Feature Extraction

### 3.1 Training set

Segmentation files are present → peak-tumor slice selected per patient.

```bash
conda activate mricore

python extract_mricore_fets24.py \
    --checkpoint  ./mri_foundation/pretrained_weights/mri_foundation.pth \
    --data_dir    /path/to/MICCAI_FeTS2022_TrainingData/MICCAI_FeTS2022_TrainingData \
    --output_dir  ./features/train \
    --missing_ok
```

### 3.2 Validation set

No segmentation files → falls back to the middle axial slice per patient.

```bash
python extract_mricore_fets24.py \
    --checkpoint  ./mri_foundation/pretrained_weights/mri_foundation.pth \
    --data_dir    /path/to/MICCAI_FeTS2022_ValidationData/MICCAI_FeTS2022_ValidationData \
    --output_dir  ./features/validation \
    --missing_ok
```

### 3.3 Optional: supply labels

```bash
--label_csv ./labels.csv   # columns: pat_id, label
```

If omitted, `GroundTruthClassLabel` defaults to `0`.

---

## 4. Output

Four CSVs written to `--output_dir`, one per modality:

| File | Shape |
|---|---|
| `mricore_t1_features.csv` | (N, 257) |
| `mricore_t2_features.csv` | (N, 257) |
| `mricore_flair_features.csv` | (N, 257) |
| `mricore_t1ce_features.csv` | (N, 257) |

Columns: `Feature_0, Feature_1, …, Feature_255, GroundTruthClassLabel`  
Index: `pat_id`

---

## 5. Arguments Reference

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `mri_foundation.pth` |
| `--data_dir` | required | Root folder — one sub-folder per patient |
| `--output_dir` | `./mricore_features/train_peak` | Where to save the 4 output CSVs |
| `--label_csv` | None | Optional CSV with `pat_id, label` columns |
| `--image_size` | 1024 | Input resolution (default matches SAM ViT-B training) |
| `--missing_ok` | False | Skip patients with missing modalities |

---

## 6. How it works

**Slice selection (per patient):**
- Load `_seg.nii.gz` → count non-zero voxels per axial slice → pick the slice
  with the maximum count (`np.argmax`).
- All 4 modalities share the same slice index (volumes are co-registered).
- Fallback when no seg is found: middle slice (`depth // 2`).

**Normalization:**
- FeTS data has raw intensities (0 – ~12 000). Each slice is normalized to **[0, 1]**
  using the 1st–99th percentile of its non-zero voxels before being fed to the model.

**Encoding:**
- The selected slice is upsampled from 240×240 → **1024×1024** (bilinear) and
  replicated to 3 channels.
- `image_encoder(slice)` → [1, 256, 64, 64] → **global average pool** → **256-dim** embedding.
- Only the image encoder is used; prompt encoder and mask decoder are not loaded.
