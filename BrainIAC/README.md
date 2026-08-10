# BrainIAC — Feature Extraction for FeTS2022/2024

BrainIAC is a **3D Vision Transformer (ViT-B)** foundation model pretrained on large-scale
structural brain MRI data for tasks such as brain age prediction, IDH mutation classification,
and overall survival prediction.

- **Paper:** [A generalizable foundation model for analysis of human brain MRI](https://www.nature.com/articles/s41593-026-02202-6) (Nature Neuroscience, 2026)
- **Model repo:** https://github.com/AIM-KannLab/BrainIAC

---

## 1. Setup

### 1.1 Clone BrainIAC

```bash
git clone https://github.com/AIM-KannLab/BrainIAC.git
```

### 1.2 Create conda environment

```bash
conda create -n brainiac python=3.9 -y
conda activate brainiac
```

### 1.3 Install PyTorch

```bash
# CUDA 11.8 — adjust the index URL for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 1.4 Install remaining dependencies

```bash
pip install -r BrainIAC/requirements.txt
```

### 1.5 Download pretrained checkpoint

Download `BrainIAC.ckpt` from the
[Dropbox link in the BrainIAC README](https://github.com/AIM-KannLab/BrainIAC#model-checkpoints)
and place it at:

```
BrainIAC/
└── checkpoints/
    └── BrainIAC.ckpt
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
│   └── FeTS2022_00000_seg.nii.gz   ← automatically ignored
├── FeTS2022_00002/
│   └── ...
└── ...   (1251 training patients, 219 validation patients)
```

Data is available on [Synapse](https://www.synapse.org/Synapse:syn54079892).
Volumes are already co-registered to the SRI24 atlas, skull-stripped, and at
1 mm isotropic resolution — no additional preprocessing required.

---

## 3. Feature Extraction

### 3.1 Training set

```bash
conda activate brainiac

PYTHONPATH=/path/to/BrainIAC/src \
python extract_brainiac_fets24.py \
    --checkpoint  ./BrainIAC/checkpoints/BrainIAC.ckpt \
    --data_dir    /path/to/MICCAI_FeTS2022_TrainingData/MICCAI_FeTS2022_TrainingData \
    --output_dir  ./features/train \
    --missing_ok
```

### 3.2 Validation set

```bash
PYTHONPATH=/path/to/BrainIAC/src \
python extract_brainiac_fets24.py \
    --checkpoint  ./BrainIAC/checkpoints/BrainIAC.ckpt \
    --data_dir    /path/to/MICCAI_FeTS2022_ValidationData/MICCAI_FeTS2022_ValidationData \
    --output_dir  ./features/validation \
    --missing_ok
```

> Replace `/path/to/BrainIAC/src` with the actual path to the `src/` directory
> inside the cloned BrainIAC repo.

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
| `brainiac_t1_features.csv` | (N, 769) |
| `brainiac_t2_features.csv` | (N, 769) |
| `brainiac_flair_features.csv` | (N, 769) |
| `brainiac_t1ce_features.csv` | (N, 769) |

Columns: `Feature_0, Feature_1, …, Feature_767, GroundTruthClassLabel`  
Index: `pat_id`

---

## 5. Arguments Reference

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `BrainIAC.ckpt` |
| `--data_dir` | required | Root folder — one sub-folder per patient |
| `--output_dir` | `./inference/features` | Where to save the 4 output CSVs |
| `--label_csv` | None | Optional CSV with `pat_id, label` columns |
| `--batch_size` | 1 | DataLoader batch size |
| `--num_workers` | 1 | DataLoader workers |
| `--missing_ok` | False | Skip patients with missing modalities |

---

## 6. How it works

- Each volume (240×240×155) is **resampled to 96×96×96** via trilinear interpolation.
- Intensities are **z-score normalised** over non-zero voxels, channel-wise.
- The **768-dim CLS token** from the ViT-B backbone is used as the feature vector.
- All 4 modalities are processed independently; each gets its own CSV.
