#!/usr/bin/env python3
"""
BrainIAC — Per-Modality Feature Extractor for FeTS2022
========================================================
Extracts BrainIAC ViT-B features separately for each of the 4 MRI sequences
and writes one CSV per modality (matching the original get_brainiac_features.py
output format exactly).

Output — 4 separate CSVs, one per modality:
    brainiac_t1_features.csv       768 Feature columns + GroundTruthClassLabel
    brainiac_t2_features.csv
    brainiac_flair_features.csv
    brainiac_t1ce_features.csv

FeTS2022 folder structure
─────────────────────────
  data_dir/
    FeTS2022_00000/
        FeTS2022_00000_flair.nii.gz
        FeTS2022_00000_seg.nii.gz      ← ignored
        FeTS2022_00000_t1.nii.gz
        FeTS2022_00000_t1ce.nii.gz
        FeTS2022_00000_t2.nii.gz
    FeTS2022_00002/
        ...
  (1254 patients, flat layout — no nested baseline subfolder)

Run
───
  conda activate brainiac
  export PYTHONPATH=/path/to/BrainIAC/src:$PYTHONPATH

  # Training set
  python extract_brainiac_fets2022.py \\
      --checkpoint  ./BrainIAC/checkpoints/BrainIAC.ckpt \\
      --data_dir    /mnt/e/fets2024/MICCAI_FeTS2022_TrainingData/MICCAI_FeTS2022_TrainingData \\
      --output_dir  ./inference/features/train \\
      --missing_ok

  # Validation set (same command, different paths)
  python extract_brainiac_fets2022.py \\
      --checkpoint  ./BrainIAC/checkpoints/BrainIAC.ckpt \\
      --data_dir    /mnt/e/fets2024/MICCAI_FeTS2022_ValidationData/MICCAI_FeTS2022_ValidationData \\
      --output_dir  ./inference/features/val \\
      --missing_ok

  Optional — supply a CSV of labels (pat_id, label):
      --label_csv ./data/csvs/labels.csv
  If omitted, GroundTruthClassLabel defaults to 0 for all patients.

Output per modality
───────────────────
  Columns : Feature_0, Feature_1, … Feature_767, GroundTruthClassLabel
  Rows    : one per patient (same order for all 4 CSVs)
  Shape   : (N_patients, 769)   — 768 features + 1 label

  This is identical to the output of the original get_brainiac_features.py
  so any downstream code that reads that CSV works unchanged.
"""

import torch
import numpy as np
import pandas as pd
import random
import os
import sys
import tempfile
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# ── Original BrainIAC imports (unchanged) ─────────────────────────────────────
from dataset import BrainAgeDataset, get_validation_transform
from load_brainiac import load_brainiac

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Seed — verbatim from original ────────────────────────────────────────────
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ── GPU — verbatim from original ──────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

MODALITY_ORDER = ["t1", "t2", "flair", "t1ce"]

# t1ce MUST appear before t1 — prevents "_t1_" matching a t1ce filename.
# Tested against FeTS2022 naming (FeTS2022_00000_t1ce.nii.gz etc.) — all 4
# modalities match correctly, seg file is correctly ignored.
DEFAULT_PATTERNS = {
    "t1ce":  ["_t1ce_", "_t1ce.", "t1ce.nii", "_t1c_", "_t1gd_",
              "postcontrast", "t1gd", "t1+c"],
    "flair": ["_flair_", "_flair.", "flair.nii", "flair"],
    "t2":    ["_t2_",    "_t2.",   "t2.nii",    "t2w",  "t2"],
    "t1":    ["_t1_",    "_t1.",   "t1.nii",    "t1w",  "t1"],
}


# =============================================================================
# Folder discovery
# Uses recursive glob → handles FeTS2022 flat layout AND TCGA nested layout
# =============================================================================

def discover_modalities(patient_dir: Path) -> dict:
    """
    Find one NIfTI per modality under patient_dir.

    FeTS2022 flat layout (primary target):
        FeTS2022_00000/FeTS2022_00000_t1.nii.gz   etc.

    Also works for TCGA nested:
        TCGA-02-0003/TCGA-02-0003_baseline/*_t1_*.nii.gz
    """
    niftis = sorted(patient_dir.glob("**/*.nii.gz"))
    if not niftis:
        niftis = sorted(patient_dir.glob("**/*.nii"))
    if not niftis:
        raise FileNotFoundError(f"No NIfTI files found under {patient_dir}")

    found = {}
    for mod, patterns in DEFAULT_PATTERNS.items():
        for nii in niftis:
            if mod in found:
                break
            if any(p in nii.name.lower() for p in patterns):
                found[mod] = nii
    return found


# =============================================================================
# infer() — verbatim from original get_brainiac_features.py
# =============================================================================

def infer(model, test_loader):
    features_df = None
    model.eval()

    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Extracting ViT features", unit="batch"):
            inputs       = sample['image'].to(device)
            class_labels = sample['label'].float().to(device)

            # Get features from the ViT backbone model
            features       = model(inputs)
            features_numpy = features.cpu().numpy()

            # Expand features into separate columns
            feature_columns = [f'Feature_{i}' for i in range(features_numpy.shape[1])]
            batch_features  = pd.DataFrame(features_numpy, columns=feature_columns)
            batch_features['GroundTruthClassLabel'] = (
                class_labels.cpu().numpy().flatten()
            )

            if features_df is None:
                features_df = batch_features
            else:
                features_df = pd.concat(
                    [features_df, batch_features], ignore_index=True
                )

    return features_df


# =============================================================================
# Bridge: discovered paths → temp CSV → BrainAgeDataset
#
# BrainAgeDataset reads csv_path (pat_id, label) + root_dir.
# We set pat_id = full absolute path, root_dir = ""
# so os.path.join("", abs_path) == abs_path.
#
# If your BrainAgeDataset appends ".nii.gz" internally, change:
#     str(path.resolve())  →  str(path.resolve()).replace(".nii.gz", "")
# =============================================================================

def build_temp_csv(
    mod_path_map: dict,   # {pat_id: Path}  for one modality
    label_map:    dict,   # {pat_id: label}
    tmp_dir:      str,
    modality:     str,
) -> tuple[str, str]:
    """Write temp CSV and return (csv_path, root_dir) for BrainAgeDataset."""
    # BrainAgeDataset appends ".nii.gz" at load time, so strip it from the path.
    rows = [
        {"pat_id": str(path.resolve()).removesuffix(".nii.gz").removesuffix(".nii"),
         "label": label_map.get(pat_id, 0)}
        for pat_id, path in mod_path_map.items()
    ]
    csv_path = os.path.join(tmp_dir, f"tmp_{modality}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, ""   # root_dir="" → pat_id IS the full path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract BrainIAC features separately for each MRI modality "
            "from FeTS2022. Writes one CSV per modality."
        )
    )
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to BrainIAC .ckpt checkpoint.")
    parser.add_argument("--data_dir",    required=True,
                        help="Root folder — each sub-folder is one patient.")
    parser.add_argument("--output_dir",
                        default="./inference/features",
                        help="Directory to save the 4 output CSVs. "
                             "Default: ./inference/features")
    parser.add_argument("--label_csv",   default=None,
                        help="Optional CSV with columns pat_id, label. "
                             "GroundTruthClassLabel defaults to 0 if omitted.")
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--missing_ok",  action="store_true",
                        help="Skip modalities that are missing for a patient "
                             "instead of crashing. That patient is absent from "
                             "that modality's CSV.")
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

    # ── Step 1: discover all patients and their per-modality paths ────────────
    data_dir = Path(args.data_dir).resolve()
    pat_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    log.info(f"Found {len(pat_dirs)} patient directories in {data_dir}")

    # all_paths[modality][pat_id] = Path
    all_paths        = {mod: {} for mod in MODALITY_ORDER}
    discovery_errors = {}

    for pat_dir in tqdm(pat_dirs, desc="Discovering files"):
        pat_id = pat_dir.name
        try:
            found   = discover_modalities(pat_dir)
            missing = [m for m in MODALITY_ORDER if m not in found]
            if missing:
                msg = f"Missing modalities: {missing}"
                if not args.missing_ok:
                    raise FileNotFoundError(msg)
                log.warning(f"  {pat_id}: {msg}")
            for mod, path in found.items():
                all_paths[mod][pat_id] = path
        except Exception as e:
            log.error(f"  FAILED discovery {pat_id}: {e}")
            discovery_errors[pat_id] = str(e)
            if not args.missing_ok:
                raise

    all_pat_ids = sorted({pid for m in MODALITY_ORDER for pid in all_paths[m]})
    log.info(f"  → {len(all_pat_ids)} patients discoverable "
             f"| {len(discovery_errors)} discovery errors")

    # ── Step 2: load model once ───────────────────────────────────────────────
    print(args.checkpoint)
    model = load_brainiac(args.checkpoint, device)

    # ── Step 3: run infer() once per modality → save separate CSV ────────────
    #
    # Output format per CSV (identical to original get_brainiac_features.py):
    #   Feature_0, Feature_1, … Feature_767, GroundTruthClassLabel
    #   (N_patients, 769)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for modality in MODALITY_ORDER:
            mod_path_map = all_paths[modality]   # {pat_id: Path}

            if not mod_path_map:
                log.warning(
                    f"\n[SKIP] {modality.upper()} — no files found for any patient."
                )
                continue

            log.info(f"\n{'─'*60}")
            log.info(f"  Modality : {modality.upper()}")
            log.info(f"  Patients : {len(mod_path_map)}")
            log.info(f"{'─'*60}")

            csv_path, root_dir = build_temp_csv(
                mod_path_map, label_map, tmp_dir, modality
            )

            # ── identical to original: dataset → loader → infer() ────────────
            test_dataset = BrainAgeDataset(
                csv_path=csv_path,
                root_dir=root_dir,
                transform=get_validation_transform()
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            df = infer(model, test_loader)
            # ─────────────────────────────────────────────────────────────────

            # Assign patient IDs as index so the CSV is self-contained
            df.index      = list(mod_path_map.keys())
            df.index.name = "pat_id"

            # Save — one CSV per modality
            out_csv = os.path.join(
                args.output_dir, f"brainiac_{modality}_features.csv"
            )
            df.to_csv(out_csv)

            n_feat = df.shape[1] - 1   # subtract GroundTruthClassLabel
            log.info(f"  Saved → {out_csv}")
            log.info(f"  Shape : {df.shape}  ({n_feat} feature dims + label)")
            print(f"\nViT BrainIAC {modality.upper()} features saved to {out_csv}")
            print(f"Feature shape: {df.shape}")
            print(f"Number of feature dimensions: {n_feat}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("All modalities done.")
    log.info(f"Output files in: {args.output_dir}")
    for mod in MODALITY_ORDER:
        out_csv = os.path.join(args.output_dir, f"brainiac_{mod}_features.csv")
        if Path(out_csv).exists():
            n = len(pd.read_csv(out_csv))
            log.info(f"  brainiac_{mod}_features.csv  →  {n} patients")
        else:
            log.warning(f"  brainiac_{mod}_features.csv  →  NOT CREATED (no files)")
    if discovery_errors:
        log.warning(f"\n{len(discovery_errors)} patients had discovery errors:")
        for pid, msg in discovery_errors.items():
            log.warning(f"  {pid}: {msg}")
    log.info("=" * 60)
    log.info(
        "\nDownstream use:\n"
        "  import pandas as pd\n"
        "  t1    = pd.read_csv('brainiac_t1_features.csv',   index_col='pat_id')\n"
        "  t2    = pd.read_csv('brainiac_t2_features.csv',   index_col='pat_id')\n"
        "  flair = pd.read_csv('brainiac_flair_features.csv',index_col='pat_id')\n"
        "  t1ce  = pd.read_csv('brainiac_t1ce_features.csv', index_col='pat_id')\n"
        "\n"
        "  # Use any single modality:\n"
        "  X = t1.drop(columns='GroundTruthClassLabel').values   # (N, 768)\n"
        "  y = t1['GroundTruthClassLabel'].values\n"
        "\n"
        "  # Or concatenate all 4 later:\n"
        "  import numpy as np\n"
        "  X_all = np.hstack([\n"
        "      t1.drop(columns='GroundTruthClassLabel').values,\n"
        "      t2.drop(columns='GroundTruthClassLabel').values,\n"
        "      flair.drop(columns='GroundTruthClassLabel').values,\n"
        "      t1ce.drop(columns='GroundTruthClassLabel').values,\n"
        "  ])  # (N, 3072)"
    )


if __name__ == "__main__":
    main()