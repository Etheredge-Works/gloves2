
from pickletools import optimize
from turtle import back
import torchvision

import torch
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights, convnext_tiny, resnet50
from torchvision.transforms import AutoAugment
from torchvision import transforms as T
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from siamese_datasets import TripletDataset, PairDataset
from nway import NwayDataset, NwayCallback
import torchmetrics



from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, StochasticWeightAveraging, LearningRateMonitor)
from pytorch_lightning.loggers import WandbLogger

class SiameseNet(LightningModule):
    def __init__(self, backbone, latent_size, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.save_hyperparameters()
        head = [
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(hidden_dim, latent_size)
        ]
        if backbone == "resnet18":
            # self.backbone = resnet18(weights=None)
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = torch.nn.Sequential(
                torch.nn.Linear(512, latent_size),
                # torch.nn.LeakyReLU(),
                *head
            )
        elif backbone == "resnet50":
            self.backbone = resnet50(weights="IMAGENET1K_V2")
            self.backbone.fc = torch.nn.Sequential(
                torch.nn.Linear(2048, latent_size),
                *head
            )
        elif backbone == "convnext_tiny":
            # self.backbone = convnext_tiny(weights="DEFAULT")
            self.backbone = convnext_tiny(weights=None)
            self.backbone.classifier = torch.nn.Sequential(
                torch.nn.Flatten(), 
                torch.nn.Linear(768, latent_size),
                *head
            )
        else:
            raise ValueError("backbone not supported")
        print("backbone", self.backbone)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_size*2, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1)
        )
        
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # self.loss = torch.nn.BCEWithLogitsLoss()
        # self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.TripletMarginWithDistanceLoss()
        # self.dist = torch.nn.CosineSimilarity()
        # self.loss = torch.nn.CosineEmbeddingLoss()
        # self.loss = torch.nn.TripletMarginLoss()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x1, x2):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.classifier(x)
        x = x.view(-1)
        return x


    def training_step(self, batch, batch_idx):
        (anchor, other), label = batch
        # (anchor, pos, neg), _ = batch

        # pos = self.backbone(pos)
        # neg = self.backbone(neg)
        # dist = self.dist(anchor, other)
        # target = torch.ones_like(dist) if is_pos else torch.zeros_like(dist)
        # loss = self.loss(anchor, other, is_pos)
        # loss = self.loss(anchor, pos, neg)
        # loss = self.loss(anchor, pos, neg)
        # loss = self.loss(dist, is_pos)
        x = self(anchor, other)

        # loss = torch.nn.functional.binary_cross_entropy_with_logits(x, label)
        loss = self.loss(x, label.float())
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/acc", self.train_acc(torch.sigmoid(x), label), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)    
        return loss
    
    def validation_step(self, batch, batch_idx):
        # (anchor, pos, neg), _ = batch
        (anchor, other), label = batch
        x = self(anchor, other)

        # pos = self.backbone(pos)
        # neg = self.backbone(neg)
        # loss = self.loss(anchor, pos, neg)
        # loss = self.loss(anchor, pos, neg)
        loss = self.loss(x, label.float())
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(x, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/acc", self.val_acc(torch.sigmoid(x), label), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # return (
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40, factor=0.5, verbose=True)
            # torch.optim.lr_scheduler.CosineAnnealingLR(self.trainer.optimizers[0], T_max=10)
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss"
        }

if __name__ == "__main__":
    print("Loading data")
    ds = torchvision.datasets.OxfordIIITPet(
        root='/nfs/datasets/torch', 
        download=False,
        transform=T.Compose([
            # torchvision.transforms.Resize((224, 224)),
            # TODO TUNE DOWN MAGNITUDE
            T.RandAugment(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
    )

    # train_2_ds = torchvision.datasets.OxfordIIITPet(
    #     root='/nfs/datasets/torch', 
    #     download=False,
    #     transform=T.Compose([
    #         T.RandAugment(),
    #         T.Resize((224, 224)),
    #         # torchvision.transforms.Resize((224, 224)),
    #         T.RandomHorizontalFlip(),
    #         # T.RandomVerticalFlip(),
    #         # T.RandomRotation(30),
    #         # T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    #         T.ToTensor(),
    #         T.Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225]
    #         )
    #     ]),
    #     split="test"
    # )
    # ds = torch.utils.data.ConcatDataset([ds, train_2_ds])
    personal = torchvision.datasets.ImageFolder(
        root='/nfs/datasets/torch/my-pets/',
        target_transform=lambda x: x+37, # offset for oxfordiiitpet
        )
    print(personal[1])
    # ds = torch.utils.data.ConcatDataset([ds, personal])
    
    val_ds = torchvision.datasets.OxfordIIITPet(
        root='/nfs/datasets/torch',
        download=False,
        transform=T.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        split="test"
    )
    # nways = 129
    # train_nway_ds = NwayDataset(ds, nways, False, ratio=0.1)
    # train_nway_dl = torch.utils.data.DataLoader(train_nway_ds, batch_size=nways, num_workers=32, shuffle=False)
    # train_nway_callback = NwayCallback(train_nway_dl, nways=nways, log_prefix="train/", eval_every=25)

    # val_nway_ds = NwayDataset(val_ds, nways, False, ratio=0.2)
    # val_nway_dl = torch.utils.data.DataLoader(val_nway_ds, batch_size=nways, num_workers=32, shuffle=False)
    # val_nway_callback = NwayCallback(val_nway_dl, nways=nways, log_prefix="val/", eval_every=25)

    print("Making trainer")
    trainer = Trainer(
        accelerator="auto", 
        devices=1, 
        # strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
        max_epochs=-1,
        gradient_clip_val=1.0,
        precision=16,
        callbacks=[
            ModelCheckpoint(
                monitor="train/loss",
                save_top_k=1,
                mode="min",
                dirpath="checkpoints",
                filename="best_model"
            ),
            EarlyStopping(
                monitor="train/loss",
                patience=100,
                mode="min"
            ),
            LearningRateMonitor(),
            # train_nway_callback,
            # val_nway_callback,
            # pl.callbacks.Lear
            # pl.callbacks.LearningRateMonitor(logging_interval="step"),
            # StochasticWeightAveraging(1e-4),
        ],
        logger=WandbLogger(
            project="siamese-v2",
        )
    )
    ds = PairDataset(ds)
    val_ds =PairDataset(val_ds)
    print("Fit")
    model = SiameseNet("resnet18", 128, 512)
    # model = SiameseNet("convnext_tiny", 128, 512)
    dl = torch.utils.data.DataLoader(
        ds, 
        batch_size=64, 
        num_workers=16, 
        pin_memory=True, 
        persistent_workers=True,
        shuffle=True,
        drop_last=True)
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=64, 
        num_workers=16, 
        pin_memory=True, 
        persistent_workers=True,
        shuffle=False)
    trainer.fit(model, dl, val_dataloaders=val_dl)