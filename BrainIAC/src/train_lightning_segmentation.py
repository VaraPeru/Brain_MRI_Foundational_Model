import os
import yaml
import torch
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from segmentation_model import ViTUNETRSegmentationModel
from dataset_segmentation import get_segmentation_dataloader

class SegmentationLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model = ViTUNETRSegmentationModel(
            simclr_ckpt_path=config['pretrain']['simclr_checkpoint_path'],
            img_size=tuple(config['model']['img_size']),
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels']
        )
        self.dice_loss = DiceLoss(sigmoid=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.train_metric = DiceMetric(include_background=False, reduction='mean')
        self.val_metric = DiceMetric(include_background=False, reduction='mean')
        
        # Freeze backbone if specified
        if str(config['training'].get('freeze', 'no')).lower() == "yes":
            for param in self.model.vit.parameters():
                param.requires_grad = False
            print("ViT backbone weights frozen!!")
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        img, lbl = batch['image'], batch['label']
        seg = self(img).float()
        
        dice_loss = self.dice_loss(seg, lbl)
        bce_loss = self.bce_loss(seg, lbl)
        loss = dice_loss + bce_loss
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_dice_loss', dice_loss, on_step=True, on_epoch=True)
        self.log('train_bce_loss', bce_loss, on_step=True, on_epoch=True)

        preds = torch.sigmoid(seg) > 0.5
        self.train_metric(y_pred=preds, y=lbl)
        return loss
    def on_train_epoch_end(self):
        train_dice = self.train_metric.aggregate()
        self.log('train_dice', train_dice, prog_bar=True, sync_dist=True)
        self.train_metric.reset()
    def validation_step(self, batch, batch_idx):
        img, lbl = batch['image'], batch['label']
        seg = sliding_window_inference(
            img,
            roi_size=tuple(self.config['model']['img_size']),
            sw_batch_size=self.config['training']['sw_batch_size'],
            predictor=self
        ).float()
        
        dice_loss = self.dice_loss(seg, lbl)
        bce_loss = self.bce_loss(seg, lbl)
        loss = dice_loss + bce_loss

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_dice_loss', dice_loss, sync_dist=True)
        self.log('val_bce_loss', bce_loss, sync_dist=True)

        preds = torch.sigmoid(seg) > 0.5
        self.val_metric(y_pred=preds, y=lbl)
        return loss
    def on_validation_epoch_end(self):
        val_dice = self.val_metric.aggregate()
        self.log('val_dice', val_dice, prog_bar=True, sync_dist=True)
        self.val_metric.reset()
    def configure_optimizers(self):
        t = self.config['training']
       
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.AdamW(trainable_params, lr=float(t['lr']), weight_decay=float(t['weight_decay']))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t['max_epochs'], eta_min=1e-6)
        return [opt], [sch]

def get_dataloaders(config):
    train_ds = get_segmentation_dataloader(
        csv_file=config['data']['train_csv'],
        img_size=tuple(config['model']['img_size']),
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        is_train=True
    )
    val_ds = get_segmentation_dataloader(
        csv_file=config['data']['val_csv'],
        img_size=tuple(config['model']['img_size']),
        batch_size=1,
        num_workers=1,
        is_train=False
    )
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, val_loader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_finetune_segmentation.yml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']['visible_device']
    wandb_logger = WandbLogger(
        project=config['logger']['project_name'],
        name=config['logger']['run_name'],
        config=config
    )
    train_loader, val_loader = get_dataloaders(config)
    model = SegmentationLightningModule(config)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['output']['output_dir'],
        filename=config['logger']['save_name'],
        monitor='val_dice',
        mode='max',
        save_top_k=5
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator='gpu',
        devices=1,
        strategy='auto',
        precision="16-mixed",
        gradient_clip_val=1.0
    )
    trainer.fit(model, train_loader, val_loader) 