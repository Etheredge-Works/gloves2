import torch
import torchmetrics
from torchvision.models import resnet18, ResNet18_Weights, convnext_tiny, resnet50
from pytorch_lightning import LightningModule
from icecream import ic


class SiameseNet(LightningModule):
    def __init__(
        self, 
        backbone, 
        latent_size, 
        hidden_dim=512, 
        dropout=0.3,
        simple_head=True,

    ):
        super().__init__()
        self.save_hyperparameters()
        if backbone == "resnet18":
            # self.backbone = resnet18(weights=None)
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = torch.nn.Sequential(
                torch.nn.Linear(512, latent_size),
            )
        elif backbone == "resnet50":
            self.backbone = resnet50(weights="IMAGENET1K_V2")
            self.backbone.fc = torch.nn.Sequential(
                torch.nn.Linear(2048, latent_size),
            )
        elif backbone == "convnext_tiny":
            self.backbone = convnext_tiny(weights="DEFAULT")
            # self.backbone = convnext_tiny(weights=None)
            self.backbone.classifier = torch.nn.Sequential(
                torch.nn.Flatten(), 
                torch.nn.Linear(768, latent_size),
            )
        else:
            raise ValueError("backbone not supported")
        print("backbone", self.backbone)

        if simple_head:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(latent_size, 1),
                # torch.nn.Sigmoid()
            )
        else:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(latent_size*2, hidden_dim),
                # torch.nn.LeakyReLU(),
                # torch.nn.Dropout(dropout),
                # torch.nn.Linear(hidden_dim, hidden_dim),
                # torch.nn.LeakyReLU(),
                # torch.nn.Dropout(dropout),
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
        if self.hparams.simple_head:
            x = torch.abs(x1 - x2)
            # x = torch.sum(x, dim=1)
            # x = F.cosine_similarity(x1, x2)
        else:
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