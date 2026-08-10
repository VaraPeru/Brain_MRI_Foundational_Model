import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# MONAI for perturbations
from monai.transforms import AdjustContrast, RandBiasField, RandGibbsNoise

# Import from simclrvit_finetuning project files
from test_inference_finetune import (
    DATASETS, DATASETS_TO_RUN, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, DEVICE, SIMCLR_CKPT_PATH,
    load_model, create_test_dataset
)

# =============================
# SELECT WHICH DATASETS TO RUN
# =============================
# Uncomment and edit this list to override the datasets to run for perturbation analysis
DATASETS_TO_RUN = [
    #"brain_age_100",
    "mci_classification_80",
    #"idh_classification_20",
    #"sequence_multiclass_100",
    #"stroke_100",
    #"os_survival_100int",
]

# =============================================================================
# PERTURBATION CONFIGURATION - MODIFY THESE SETTINGS
# =============================================================================

PERTURBATION_CONFIG = {
    # Master switch to enable or disable all perturbations
    "apply_perturbations": True,
    "types": {
        "contrast": {
            "enabled": True,
            # List of gamma values for contrast adjustment
            "params": list(np.round(np.arange(0.5, 2.0, 0.1), 2))
        },
        "bias_field": {
            "enabled": False,
            # List of coefficient ranges for bias field simulation
            "params": list(np.round(np.arange(0.0, 0.5, 0.05), 2))
        },
        "gibbs_noise": {
            "enabled": False,
            # List of alpha values for Gibbs noise simulation
            "params": list(np.round(np.arange(0.0, 0.5, 0.05), 2))
        }
    }
}

# =============================================================================

def apply_perturbation(image_tensor, p_type, p_param):
    """Applies a single MONAI perturbation to a tensor."""
    image_tensor_b = image_tensor.unsqueeze(0) # Add batch dimension
    if p_type == "contrast":
        perturbed_tensor = AdjustContrast(gamma=p_param)(image_tensor_b)
    elif p_type == "bias_field":
        image_tensor_no_channel = image_tensor_b.squeeze(1) # Shape  [B, H, W, D]
        transform = RandBiasField(prob=1.0, coeff_range=(p_param, p_param))
        perturbed_tensor_no_channel = transform(image_tensor_no_channel)
        perturbed_tensor = perturbed_tensor_no_channel.unsqueeze(1) # Add channel dim back
    elif p_type == "gibbs_noise":
        transform = RandGibbsNoise(prob=1.0, alpha=(p_param, p_param))
        perturbed_tensor = transform(image_tensor_b)
    else:
        raise ValueError(f"Unknown perturbation type: {p_type}")
    return perturbed_tensor.squeeze(0) # Remove batch dimension


def run_perturbation_analysis(model, dataloader, dataset_config, p_config):
    results_dfs = {p_type: [] for p_type in p_config["types"] if p_config["types"][p_type]["enabled"]}
    task_type = dataset_config['task_type']
    image_type = dataset_config['image_type']
    dataset = dataloader.dataset
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Analyzing {dataset_config['name']}")):
            batch_size = 1
            # Extract batch size and patient ids
            if image_type == "single":
                images = batch['image']
                labels = batch['label']
                pat_ids = batch.get('pat_id', [str(batch_idx)])
            elif image_type in ["dual", "quad"]:
                images = batch[0]  
                labels = batch[1]
               
                try:
                    pat_ids = [str(dataset.dataframe.iloc[batch_idx]['pat_id'])]
                except Exception:
                    pat_ids = [str(batch_idx)]
            else:
                raise TypeError(f"Unsupported image_type: {image_type}")

            # For each sample in batch (BATCH_SIZE=1 for now)
            for i in range(images.shape[0]):
                label = labels[i].item() if hasattr(labels[i], 'item') else labels[i]
                pat_id = pat_ids[i] if i < len(pat_ids) else str(batch_idx)
                # --- Baseline (unperturbed) ---
                if image_type == "single":
                    model_input_gpu = images[i].unsqueeze(0).to(DEVICE)
                else:
                    model_input_gpu = images[i].unsqueeze(0).to(DEVICE)
                output = model(model_input_gpu)
                if task_type == 'regression':
                    base_prediction = output.item()
                    base_model_output = output.item()
                elif task_type == 'classification':
                    base_model_output = output.item()
                    base_prediction = torch.sigmoid(output).item()
                elif task_type == 'multiclass':
                    base_model_output = output.cpu().numpy().squeeze()
                    base_prediction = torch.softmax(output, dim=1).cpu().numpy().squeeze()
                else:
                    raise ValueError(f"Unknown task_type: {task_type}")
                base_result = {
                    "patient_id": pat_id,
                    "true_label": label,
                    "perturbation_type": "original",
                    "perturbation_value": 0.0,
                    "slot": None,
                    "model_output": base_model_output.tolist() if isinstance(base_model_output, np.ndarray) else base_model_output,
                    "prediction": base_prediction.tolist() if isinstance(base_prediction, np.ndarray) else base_prediction
                }
                # Store baseline in each enabled perturbation type's results for consistency
                for p_type in results_dfs:
                    results_dfs[p_type].append(base_result.copy())
                if not p_config["apply_perturbations"]:
                    continue
                # --- Perturbations ---
                for p_type, p_settings in p_config["types"].items():
                    if not p_settings["enabled"]:
                        continue
                   
                    n_slots = 1 if image_type == "single" else images.shape[1]
                    for slot in range(n_slots):
                        for p_param in tqdm(p_settings["params"], desc=f"  - {p_type} (slot {slot})", leave=False):
                         
                            if image_type == "single":
                                perturbed_image = apply_perturbation(images[i].cpu(), p_type, p_param)
                                model_input_gpu = perturbed_image.unsqueeze(0).to(DEVICE)
                            else:
                                perturbed_images = images[i].clone().cpu()
                                perturbed_images[slot] = apply_perturbation(perturbed_images[slot], p_type, p_param)
                                model_input_gpu = perturbed_images.unsqueeze(0).to(DEVICE)
                            output = model(model_input_gpu)
                            if task_type == 'regression':
                                prediction = output.item()
                                model_output = output.item()
                            elif task_type == 'classification':
                                model_output = output.item()
                                prediction = torch.sigmoid(output).item()
                            elif task_type == 'multiclass':
                                model_output = output.cpu().numpy().squeeze()
                                prediction = torch.softmax(output, dim=1).cpu().numpy().squeeze()
                            else:
                                raise ValueError(f"Unknown task_type: {task_type}")
                            results_dfs[p_type].append({
                                "patient_id": pat_id,
                                "true_label": label,
                                "perturbation_type": p_type,
                                "perturbation_value": p_param,
                                "slot": slot if n_slots > 1 else None,
                                "model_output": model_output.tolist() if isinstance(model_output, np.ndarray) else model_output,
                                "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
                            })
    for p_type, results_list in results_dfs.items():
        results_dfs[p_type] = pd.DataFrame(results_list)
    return results_dfs


def main():
    print("=== MODEL PERTURBATION ANALYSIS PIPELINE ===")
    print(f"Device: {DEVICE}")
    print(f"Datasets to run: {DATASETS_TO_RUN}")
    print("="*50)
    for dataset_key in DATASETS_TO_RUN:
        if dataset_key not in DATASETS:
            print(f"Error: Dataset '{dataset_key}' not found in config!")
            continue
       
        config = DATASETS[dataset_key]
        config['name'] = dataset_key
        try:
            model = load_model(
                checkpoint_path=config["checkpoint_path"],
                simclr_ckpt_path=SIMCLR_CKPT_PATH,
                task_type=config["task_type"],
                image_type=config["image_type"],
                num_classes=config["num_classes"]
            )
            test_dataset, collate_fn = create_test_dataset(
                csv_path=config["test_csv_path"],
                root_dir=config["root_dir"],
                image_type=config["image_type"],
                image_size=IMAGE_SIZE,
                dataset_name=dataset_key
            )
            test_loader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, collate_fn=collate_fn
            )
            perturbation_results = run_perturbation_analysis(model, test_loader, config, PERTURBATION_CONFIG)
            output_dir = os.path.dirname(config['output_csv_path'])
            os.makedirs(output_dir, exist_ok=True)
            for p_type, df in perturbation_results.items():
                output_filename = f"perturbation_analysis_{dataset_key}_{p_type}.csv"
                output_path = os.path.join(output_dir, output_filename)
                df.to_csv(output_path, index=False)
                print(f"complete...")
        except Exception as e:
            print(f"Error:{e}")
            import traceback
            traceback.print_exc()
            continue
    

if __name__ == "__main__":
    main() 