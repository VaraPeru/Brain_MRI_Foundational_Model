import os
import json
import yaml
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset
import csv

# Import from segmentation model
from dataset_segmentation import get_segmentation_dataloader
from segmentation_model import ViTUNETRSegmentationModel

def load_model(config, checkpoint_path):
    """
    Loads a ViTUNETRSegmentationModel and populates it with weights from a
    PyTorch Lightning checkpoint.
    """
   #load the model
    model = ViTUNETRSegmentationModel(
        simclr_ckpt_path=config['pretrain']['simclr_checkpoint_path'],
        img_size=tuple(config['model']['img_size']),
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    )

    # 2load the state_dic
    state_dict = torch.load(checkpoint_path)['state_dict']
    
    # remove model. for lightning compatibiltiy 
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            k = k[len('model.'):]
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    return model.eval().cuda()

def get_test_dataloader(config, test_csv):
    """
    Creates a DataLoader for the test set.
    """
   
    test_ds = get_segmentation_dataloader(
        csv_file=test_csv,
        img_size=tuple(config['model']['img_size']),
        batch_size=1, 
        num_workers=1,
        is_train=False
    )
    # spinup dataloader
    return DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

def evaluate(model, test_loader, config):
    """
    Runs the evaluation loop, calculates metrics, and returns them.
    Also records per-case Dice scores.
    """
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    jaccard_metric = MeanIoU(include_background=False, reduction="mean")
    cm_metric = ConfusionMatrixMetric(metric_name=["precision", "recall"], include_background=False)

    post_pred = AsDiscrete(threshold=0.5)
    post_label = AsDiscrete(threshold=0.5)

    per_case_dice = {}
    per_case_ids = []
    per_case_scores = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        image = batch['image'].cuda()
        label = batch['label'].cuda()

        pred = sliding_window_inference(
            inputs=image,
            roi_size=tuple(config['model']['img_size']),
            sw_batch_size=config['training']['sw_batch_size'],
            predictor=model,
            overlap=0.5
        )

        pred = torch.sigmoid(pred)
        pred = post_pred(pred)
        label = post_label(label)

        # Per-case Dice (single case per batch)
        dice_case = DiceMetric(include_background=False, reduction="mean")
        dice_case(pred, label)
        dice_value = dice_case.aggregate().item()
        # Get image path or pat_id
        if 'image_meta_dict' in batch and 'filename_or_obj' in batch['image_meta_dict']:
            image_id = batch['image_meta_dict']['filename_or_obj'][0]  # batch size 1
        else:
            image_id = f"case_{len(per_case_dice)}"
        per_case_dice[image_id] = dice_value

        dice_metric(pred, label)
        jaccard_metric(pred, label)
        cm_metric(pred, label)

    # Aggregate metrics
    dice = dice_metric.aggregate().item()
    jaccard = jaccard_metric.aggregate().item()
    precision_per_class = cm_metric.aggregate("precision")
    recall_per_class = cm_metric.aggregate("recall")
    precision = torch.stack(precision_per_class).mean().item()
    recall = torch.stack(recall_per_class).mean().item()

    metrics = {
        "dice": dice,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "per_case_dice": per_case_dice
    }
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--config', type=str, required=False, default="config_finetune_segmentation.yml", help="Path to config YAML")
    parser.add_argument('--output_json', type=str, required=False, default="./inference/model_outputs/segmentation.json", help="Path to save combined metrics JSON")
    parser.add_argument('--test_csv', type=str, required=True, help="Path to test CSV file")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to checkpoint file")
    parser.add_argument('--experiment_name', type=str, required=False, default="segmentation_task", help="Name for the experiment")
    parser.add_argument('--csv_output_dir', type=str, required=False, default="./inference/per_case_results", help="Directory to save per-case CSV files")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']['visible_device']

    # --- Checkpoints to evaluate ---
    checkpoints_to_evaluate = {
        args.experiment_name: args.checkpoint_path,
    }
    # -----------------------------------------

    if not checkpoints_to_evaluate or not args.checkpoint_path:
        print("No checkpoint path provided. Please specify --checkpoint_path")
        exit()
        
    
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(args.csv_output_dir, exist_ok=True)

    all_metrics = {}
    test_loader = get_test_dataloader(config, args.test_csv)

    # store results 
    mean_dice_scores = {}

    for experiment_name, ckpt_path in checkpoints_to_evaluate.items():
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for '{experiment_name}'")
            continue

        print(f" Evaluating : '{experiment_name}' ")
        model = load_model(config, ckpt_path)
        metrics = evaluate(model, test_loader, config)
        
        all_metrics[experiment_name] = metrics
        print(f"Metrics for '{experiment_name}': {metrics}")

        # --- Save per-case Dice to CSV ---
        csv_path = os.path.join(args.csv_output_dir, f"{experiment_name}_per_case_dice.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["dice"])
            for dice in metrics["per_case_dice"].values():
                writer.writerow([dice])
        
        
        # Print mean Dice 
        mean_dice = sum(metrics["per_case_dice"].values()) / len(metrics["per_case_dice"])
        mean_dice_scores[experiment_name] = mean_dice
        print(f"Mean Dice for '{experiment_name}': {mean_dice:.4f}")

    # Print summary 
   
    print("SUMMARY :")
    
    for experiment_name, mean_dice in mean_dice_scores.items():
        print(f"{experiment_name}: {mean_dice:.4f}")
    
    #save results 
    with open(args.output_json, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    