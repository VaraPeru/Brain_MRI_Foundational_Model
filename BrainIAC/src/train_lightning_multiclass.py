import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb

from model import ViTBackboneNet, Classifier, SingleScanModel
from dataset import SequenceDataset, get_default_transform, get_validation_transform

class MultiClassSequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.backbone = ViTBackboneNet(
            simclr_ckpt_path=config['simclrvit']['ckpt_path']
        )
        self.classifier = Classifier(d_model=768, num_classes=4)  # 768 for ViT-B, 4 for multiclass
        self.model = SingleScanModel(self.backbone, self.classifier)
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.validation_step_outputs = []

        if str(config['train']['freeze']).lower() == "yes":
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print("INFO: Backbone weights FROZEN based on config['train']['freeze'] == 'yes'.")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label'].long()
        y_hat_logits = self(x)
        loss = self.criterion(y_hat_logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label'].long()
        y_hat_logits = self(x)
        loss = self.criterion(y_hat_logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        y_pred = torch.argmax(y_hat_logits, dim=1)
        y_probs = torch.softmax(y_hat_logits, dim=1)  # For multiclass AUC
        output = {'val_loss': loss.detach(), 'y_true': y.detach().cpu(), 'y_pred': y_pred.detach().cpu(), 'y_probs': y_probs.detach().cpu()}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        y_true = torch.cat([o['y_true'] for o in self.validation_step_outputs]).numpy().flatten()
        y_pred = torch.cat([o['y_pred'] for o in self.validation_step_outputs]).numpy().flatten()
        y_probs = torch.cat([o['y_probs'] for o in self.validation_step_outputs]).numpy()
        acc = accuracy_score(y_true, y_pred)
        self.log('val_accuracy', acc, prog_bar=True)
        if acc > self.best_val_acc:
            self.best_val_acc = acc
        # Multiclass AUC (One-vs-Rest)
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
        except Exception as e:
            auc = 0.0
            print(f"AUC calculation failed: {e}")
        self.log('val_auc', auc, prog_bar=True)
        if auc > self.best_val_auc:
            self.best_val_auc = auc
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Pass selected parameters to the optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.config['optim']['lr'], weight_decay=self.config['optim']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

class MultiClassDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        image_size = tuple(self.config['data']['size'])
        self.train_dataset = SequenceDataset(
            csv_path=self.config['data']['csv_file'],
            root_dir=self.config['data']['root_dir'],
            transform=get_default_transform(image_size=image_size)
        )
        self.val_dataset = SequenceDataset(
            csv_path=self.config['data']['val_csv'],
            root_dir=self.config['data']['root_dir'],
            transform=get_validation_transform(image_size=image_size)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['data']['batch_size'], shuffle=True, num_workers=self.config['data']['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['data'].get('val_batch_size', 1), shuffle=False, num_workers=1)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config_finetune.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.get('gpu', {}).get('visible_device', '0')

    wandb_logger = WandbLogger(
        project=config['logger'].get('project_name_multiclass', 'multiclass_classification_project'),
        name=config['logger'].get('run_name_multiclass', 'multiclass_finetune_run'),
        config=config
    )

    data_module = MultiClassDataModule(config)
    model = MultiClassSequenceLightningModule(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logger']['save_dir'],
        filename=config['logger']['save_name'],
        monitor='val_auc',
        mode='max',
        save_top_k=config['logger'].get('save_top_k', 5)
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=config['model']['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator='gpu',
        devices=config.get('gpu', {}).get('devices', 1),
        strategy=config.get('trainer', {}).get('strategy', 'ddp_find_unused_parameters_true'),
        precision=config.get('trainer', {}).get('precision', "16-mixed")
    )

    trainer.fit(model, datamodule=data_module) 