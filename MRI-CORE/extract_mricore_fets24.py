#!/usr/bin/env python3
"""
MRI-CORE — Peak-Slice Feature Extractor for FeTS2022/2024
==========================================================
Extracts a 256-dim MRI-CORE (SAM ViT-B) embedding per modality using the
single axial slice that contains the most tumor voxels.

Slice selection (per subject):
  - With seg: pick the axial index where (seg != 0).sum(axis=(0,1)) is maximum.
  - Without seg (val set): use the middle slice (depth // 2).
  All 4 modalities share the same index (co-registered volumes).

Normalization: FeTS is raw intensity → percentile-normalize each slice to [0,1]
  using the 1st–99th percentile of non-zero voxels.

Output — 4 CSVs, one per modality:
  mricore_t1_features.csv
  mricore_t2_features.csv
  mricore_flair_features.csv
  mricore_t1ce_features.csv

  Index: pat_id
  Cols : Feature_0 … Feature_255, GroundTruthClassLabel

Run
───
  conda activate mricore

  # Training set (has seg → peak-tumor slice)
  python extract_mricore_fets24_peak.py \\
      --checkpoint  ./mri_foundation/pretrained_weights/mri_foundation.pth \\
      --data_dir    /mnt/e/fets24/MICCAI_FeTS2022_TrainingData/MICCAI_FeTS2022_TrainingData \\
      --output_dir  ./mricore_features/train_peak \\
      --missing_ok

  # Validation set (no seg → middle-slice fallback)
  python extract_mricore_fets24_peak.py \\
      --checkpoint  ./mri_foundation/pretrained_weights/mri_foundation.pth \\
      --data_dir    /mnt/e/fets24/MICCAI_FeTS2022_ValidationData/MICCAI_FeTS2022_ValidationData \\
      --output_dir  ./mricore_features/val_peak \\
      --missing_ok
"""

import os
import sys
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── GPU ───────────────────────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODALITY_ORDER = ["t1", "t2", "flair", "t1ce"]

#MODALITY_ORDER = ["flair", "t1ce"]

# t1ce before t1 — prevents "_t1_" matching t1ce filenames
DEFAULT_PATTERNS = {
    "t1ce":  ["_t1ce_", "_t1ce.", "t1ce.nii", "_t1c_", "_t1gd_",
              "postcontrast", "t1gd", "t1+c"],
    "flair": ["_flair_", "_flair.", "flair.nii", "flair"],
    "t2":    ["_t2_",    "_t2.",   "t2.nii",    "t2w",  "t2"],
    "t1":    ["_t1_",    "_t1.",   "t1.nii",    "t1w",  "t1"],
}

SEG_PATTERNS = ["_seg.", "_seg_", "seg.nii", "segmentation"]


# =============================================================================
# Model loading
# =============================================================================

def load_mricore(checkpoint: str, image_size: int = 1024):
    mri_foundation_dir = Path(__file__).parent / "mri_foundation"
    sys.path.insert(0, str(mri_foundation_dir))

    import cfg
    from models.sam import sam_model_registry

    _saved_argv, sys.argv = sys.argv, ["extract"]
    args = cfg.parse_args()
    sys.argv = _saved_argv

    args.if_encoder_adapter      = False
    args.if_mask_decoder_adapter = False
    args.if_encoder_lora_layer   = False
    args.if_decoder_lora_layer   = False
    args.image_size = image_size
    args.num_cls    = 1

    model = sam_model_registry['vit_b'](
        args,
        checkpoint=checkpoint,
        num_classes=1,
        image_size=image_size,
        pretrained_sam=False,
    )
    model.eval().to(device)
    log.info(f"MRI-CORE loaded  (image_size={image_size})")
    return model


# =============================================================================
# File discovery
# =============================================================================

def discover_patient_files(patient_dir: Path) -> dict:
    niftis = sorted(patient_dir.glob("**/*.nii.gz"))
    if not niftis:
        niftis = sorted(patient_dir.glob("**/*.nii"))
    if not niftis:
        raise FileNotFoundError(f"No NIfTI files found under {patient_dir}")

    found = {}

    for nii in niftis:
        if any(p in nii.name.lower() for p in SEG_PATTERNS):
            found["seg"] = nii
            break

    for mod, patterns in DEFAULT_PATTERNS.items():
        for nii in niftis:
            if mod in found:
                break
            if any(p in nii.name.lower() for p in patterns):
                found[mod] = nii

    return found


# =============================================================================
# Peak-slice selection
# =============================================================================

def select_peak_slice(seg_path, depth: int) -> tuple[int, str]:
    """
    Return (slice_index, description).

    With seg: the axial slice whose non-zero voxel count is maximum.
    Without seg: the middle slice (depth // 2).
    """
    if seg_path is not None:
        seg_vol         = nib.load(str(seg_path)).get_fdata()   # [H, W, D]
        tumor_per_slice = (seg_vol != 0).sum(axis=(0, 1))        # [D]

        if tumor_per_slice.max() > 0:
            idx = int(np.argmax(tumor_per_slice))
            return idx, f"peak-tumor  slice={idx}  voxels={int(tumor_per_slice[idx])}"

        log.warning("  seg found but no non-zero voxels — using middle slice")

    idx = depth // 2
    return idx, f"fallback(middle)  slice={idx}"


# =============================================================================
# Slice → tensor
# =============================================================================

def slice_to_tensor(vol: np.ndarray, idx: int, image_size: int) -> torch.Tensor:
    """
    Extract one axial slice, percentile-normalize to [0,1], resize, 3-channel.
    Returns: FloatTensor [1, 3, image_size, image_size]
    """
    s = vol[:, :, idx].astype(np.float32)

    nz = s[s > 0]
    if nz.size > 0:
        p1, p99 = np.percentile(nz, 1), np.percentile(nz, 99)
        s = np.clip((s - p1) / (p99 - p1), 0.0, 1.0) if p99 > p1 else np.zeros_like(s)
    else:
        s = np.zeros_like(s)

    t = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)              # [1,1,H,W]
    t = F.interpolate(t, size=(image_size, image_size),
                      mode="bilinear", align_corners=False)
    t = t.squeeze(0).repeat(3, 1, 1).unsqueeze(0)                  # [1,3,H,W]
    return t



# =============================================================================
# Per-volume embedding
# =============================================================================

@torch.no_grad()
def extract_volume_embedding(
    model,
    nii_path:   Path,
    slice_idx:  int,
    image_size: int,
) -> np.ndarray:
    """256-dim embedding from the peak-tumor axial slice."""
    vol   = nib.load(str(nii_path)).get_fdata()            # [H, W, D]
    batch = slice_to_tensor(vol, slice_idx, image_size).to(device)
    feat  = model.image_encoder(batch)                     # [1, 256, 64, 64]
    emb   = feat.mean(dim=[-2, -1])                        # [1, 256]
    return emb.squeeze(0).cpu().numpy()                    # [256]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract MRI-CORE features from FeTS2022/2024 using the single "
            "axial slice with the most tumor voxels (peak-tumor slice). "
            "Falls back to the middle slice when no segmentation is available."
        )
    )
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to mri_foundation.pth.")
    parser.add_argument("--data_dir",    required=True,
                        help="Root folder — one sub-folder per patient.")
    parser.add_argument("--output_dir",  default="./mricore_features/train_peak",
                        help="Where to save the 4 output CSVs.")
    parser.add_argument("--label_csv",   default=None,
                        help="Optional CSV with pat_id, label columns.")
    parser.add_argument("--image_size",  type=int, default=1024,
                        help="Input resolution for MRI-CORE (default: 1024).")
    parser.add_argument("--missing_ok",  action="store_true",
                        help="Skip patients with missing/failing modalities.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Labels ────────────────────────────────────────────────────────────────
    label_map = {}
    if args.label_csv:
        ldf     = pd.read_csv(args.label_csv)
        id_col  = "pat_id" if "pat_id" in ldf.columns else ldf.columns[0]
        lab_col = "label"  if "label"  in ldf.columns else ldf.columns[1]
        label_map = dict(zip(ldf[id_col].astype(str), ldf[lab_col]))
        log.info(f"Loaded {len(label_map)} labels from {args.label_csv}")

    # ── Discover patients ─────────────────────────────────────────────────────
    data_dir = Path(args.data_dir).resolve()
    pat_dirs  = sorted(d for d in data_dir.iterdir() if d.is_dir())
    log.info(f"Found {len(pat_dirs)} patient directories in {data_dir}")

    all_paths        = {mod: {} for mod in MODALITY_ORDER}
    seg_paths        = {}
    discovery_errors = {}
    n_with_seg       = 0

    for pat_dir in tqdm(pat_dirs, desc="Discovering files"):
        pat_id = pat_dir.name
        try:
            found             = discover_patient_files(pat_dir)
            seg_paths[pat_id] = found.get("seg")
            if seg_paths[pat_id] is not None:
                n_with_seg += 1

            missing = [m for m in MODALITY_ORDER if m not in found]
            if missing:
                msg = f"Missing modalities: {missing}"
                if not args.missing_ok:
                    raise FileNotFoundError(msg)
                log.warning(f"  {pat_id}: {msg}")

            for mod in MODALITY_ORDER:
                if mod in found:
                    all_paths[mod][pat_id] = found[mod]

        except Exception as e:
            log.error(f"  FAILED discovery {pat_id}: {e}")
            discovery_errors[pat_id] = str(e)
            if not args.missing_ok:
                raise

    all_pat_ids = sorted({pid for m in MODALITY_ORDER for pid in all_paths[m]})
    log.info(
        f"  → {len(all_pat_ids)} patients  "
        f"| {n_with_seg} with seg  "
        f"| {len(all_pat_ids) - n_with_seg} without seg (middle-slice fallback)  "
        f"| {len(discovery_errors)} errors"
    )

    # ── Pre-compute peak slice index per patient ──────────────────────────────
    log.info("Computing peak-tumor slice indices...")
    peak_slice = {}
    for pat_id in tqdm(all_pat_ids, desc="Slice selection", unit="pat"):
        ref_mod  = next(m for m in MODALITY_ORDER if pat_id in all_paths[m])
        ref_path = all_paths[ref_mod][pat_id]
        depth    = nib.load(str(ref_path)).shape[2]

        idx, desc = select_peak_slice(seg_paths.get(pat_id), depth)
        peak_slice[pat_id] = idx
        log.debug(f"  {pat_id}: {desc}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_mricore(args.checkpoint, image_size=args.image_size)

    # ── Extract per modality ──────────────────────────────────────────────────
    for modality in MODALITY_ORDER:
        mod_path_map = all_paths[modality]
        if not mod_path_map:
            log.warning(f"\n[SKIP] {modality.upper()} — no files found.")
            continue

        log.info(f"\n{'─'*60}")
        log.info(f"  Modality : {modality.upper()}")
        log.info(f"  Patients : {len(mod_path_map)}")
        log.info(f"{'─'*60}")

        rows = []
        for pat_id, nii_path in tqdm(
            mod_path_map.items(), desc=f"{modality.upper()}", unit="vol"
        ):
            try:
                emb = extract_volume_embedding(
                    model,
                    nii_path,
                    slice_idx  = peak_slice[pat_id],
                    image_size = args.image_size,
                )
                label = label_map.get(pat_id, 0)
                rows.append([pat_id] + emb.tolist() + [label])
            except Exception as e:
                log.error(f"  FAILED {pat_id} [{modality}]: {e}")
                if not args.missing_ok:
                    raise

        if not rows:
            log.warning(f"  No features extracted for {modality.upper()}.")
            continue

        n_feat    = len(rows[0]) - 2
        feat_cols = [f"Feature_{i}" for i in range(n_feat)]
        df        = pd.DataFrame(rows, columns=["pat_id"] + feat_cols + ["GroundTruthClassLabel"])
        df        = df.set_index("pat_id")

        out_csv = os.path.join(args.output_dir, f"mricore_{modality}_features.csv")
        df.to_csv(out_csv)
        log.info(f"  Saved → {out_csv}  shape={df.shape}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(f"Output files in: {args.output_dir}")
    for mod in MODALITY_ORDER:
        out_csv = os.path.join(args.output_dir, f"mricore_{mod}_features.csv")
        if Path(out_csv).exists():
            n = len(pd.read_csv(out_csv))
            log.info(f"  mricore_{mod}_features.csv  →  {n} patients")
        else:
            log.warning(f"  mricore_{mod}_features.csv  →  NOT CREATED")
    if discovery_errors:
        log.warning(f"\n{len(discovery_errors)} patients had discovery errors.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
