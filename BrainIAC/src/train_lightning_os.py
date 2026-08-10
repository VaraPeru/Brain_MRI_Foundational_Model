import os
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
import torchmetrics


from model import ViTBackboneNet, Classifier, SingleScanModelQuad
from dataset import QuadImageDataset, get_default_transform_quad, get_validation_transform_quad, quad_image_collate_fn

class QuadInputBinaryClassificationLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
    
        self.backbone = ViTBackboneNet(
            #vit_config_path=config['simclrvit']['config_path'],
            simclr_ckpt_path=config['simclrvit']['ckpt_path']
        )
        
        # Classifier uses embed_dim features (from mean pooling of four embed_dim features)
        self.classifier = Classifier(d_model=768, num_classes=1)
        self.model = SingleScanModelQuad(self.backbone, self.classifier)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_val_auroc = float('-inf')
        self.validation_step_outputs = []

       
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.train_precision = torchmetrics.Precision(task="binary")
        self.train_recall = torchmetrics.Recall(task="binary")
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.train_auroc = torchmetrics.AUROC(task="binary")

        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.val_auroc = torchmetrics.AUROC(task="binary")

        # Freeze backbone
        if str(config['train']['freeze']).lower() == "yes":
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print("Backbone weights frozen!!")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y = y.float().unsqueeze(1)  # Ensure shape (batch, 1)
        y_hat_logits = self(x)
        loss = self.criterion(y_hat_logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        preds = torch.sigmoid(y_hat_logits)
        squeezed_preds = preds.squeeze(1)
        target = y.int().squeeze(1)

        self.train_accuracy.update(squeezed_preds, target)
        self.train_precision.update(squeezed_preds, target)
        self.train_recall.update(squeezed_preds, target)
        self.train_f1.update(squeezed_preds, target)
        self.train_auroc.update(squeezed_preds, target)
        
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y = y.float().unsqueeze(1)  # Ensure shape (batch, 1)
        y_hat_logits = self(x)
        loss = self.criterion(y_hat_logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        preds = torch.sigmoid(y_hat_logits)
        output = {'val_loss': loss.detach(), 'y_true': y.detach().int().squeeze(1), 'y_pred_probs': preds.detach().squeeze(1)}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_y_true = torch.cat([o['y_true'] for o in self.validation_step_outputs])
        all_y_pred_probs = torch.cat([o['y_pred_probs'] for o in self.validation_step_outputs])

        if all_y_pred_probs.ndim > 1 and all_y_pred_probs.shape[1] == 1:
            all_y_pred_probs_squeezed = all_y_pred_probs.squeeze(1)
        else:
            all_y_pred_probs_squeezed = all_y_pred_probs

        self.val_accuracy.update(all_y_pred_probs_squeezed, all_y_true)
        self.val_precision.update(all_y_pred_probs_squeezed, all_y_true)
        self.val_recall.update(all_y_pred_probs_squeezed, all_y_true)
        self.val_f1.update(all_y_pred_probs_squeezed, all_y_true)
        self.val_auroc.update(all_y_pred_probs_squeezed, all_y_true)

        val_acc = self.val_accuracy.compute()
        val_prec = self.val_precision.compute()
        val_rec = self.val_recall.compute()
        val_f1 = self.val_f1.compute()
        val_auroc_score = self.val_auroc.compute()

        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_precision', val_prec, prog_bar=True)
        self.log('val_recall', val_rec, prog_bar=True)
        self.log('val_f1', val_f1, prog_bar=True)
        self.log('val_auroc', val_auroc_score, prog_bar=True)

        if val_auroc_score > self.best_val_auroc:
            self.best_val_auroc = val_auroc_score
            
        self.validation_step_outputs.clear()
        
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()

    def configure_optimizers(self):
       
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.config['optim']['lr'], weight_decay=self.config['optim']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2) # Consider config for T_0, T_mult
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_auroc'}

class QuadInputBinaryClassificationDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        image_size = tuple(self.config['data']['size'])
        # Use quad transforms for quad image dataset
        self.train_dataset = QuadImageDataset(
            csv_path=self.config['data']['csv_file'],
            root_dir=self.config['data']['root_dir'],
            transform=get_default_transform_quad(image_size=image_size)
        )
        self.val_dataset = QuadImageDataset(
            csv_path=self.config['data']['val_csv'],
            root_dir=self.config['data']['root_dir'],
            transform=get_validation_transform_quad(image_size=image_size)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['data']['batch_size'], shuffle=True, num_workers=self.config['data']['num_workers'], collate_fn=quad_image_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=quad_image_collate_fn)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default="config_finetune.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']['visible_device']

    wandb_logger = WandbLogger(
        project=config['logger']['project_name'],
        name=config['logger']['run_name'],
        config=config
    )

    data_module = QuadInputBinaryClassificationDataModule(config)
    model = QuadInputBinaryClassificationLightningModule(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logger']['save_dir'],
        filename=config['logger']['save_name'],  
        monitor='val_auroc',
        mode='max',
        save_top_k=5
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=config['model']['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator=config['train'].get('accelerator', 'gpu'),
        devices=config['train'].get('devices', 1),
        precision=config['train'].get('precision', "16-mixed")
    )

    trainer.fit(model, datamodule=data_module) 