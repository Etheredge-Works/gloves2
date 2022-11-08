
from matplotlib import transforms
import torchvision

import torch
import numpy as np
from torch.nn import functional as F
from torchvision.transforms import AutoAugment
from torchvision import transforms as T
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from siamese_datasets import TripletDataset, PairDataset
from nway import NwayDataset, NwayCallback
import torchmetrics
from icecream import ic
from pl_bolts.models.self_supervised import SimCLR
from torchvision.datasets import ImageNet
from model import *
from torchvision.datasets import ImageFolder



from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, StochasticWeightAveraging, LearningRateMonitor)
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    print("Loading data")
    train_trans = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandAugment(), # does help regulurization
        T.Resize((224, 224)),
        # T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_transform=T.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # ds = ImageFolder(
    #     '/data/stanford_dogs/Images',
    #     # '/nfs/datasets/stanford_dogs/Images',
    #     # transform=train_trans
    #     )
    # ds = ImageNet(
    #     # '/nfs/datasets/torch/imagenet_2012',
    #     '/data/imagenet_2012',
    #     # transform=train_trans,
    #     split="train",
    # )
    ds = torchvision.datasets.OxfordIIITPet(
        root='/nfs/datasets/torch', 
        download=False,
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
    ds_offset = len(ds.classes)
    personal = torchvision.datasets.ImageFolder(
        root='/nfs/datasets/torch/my-pets/',
        target_transform=lambda x: x+ds_offset, # offset for oxfordiiitpet
        # target_transform=lambda x: x+37, # offset for oxfordiiitpet
        # transform=train_trans,
        )
    personal_train, personal_val = torch.utils.data.random_split(
        personal, 
        [int(len(personal)*0.5), len(personal)-int(len(personal)*0.5)],
        generator=torch.Generator().manual_seed(4)
    )
    # TODO seed so all val items are the same pairs for comparisons (loss can then be used)
    
    val_ds = torchvision.datasets.OxfordIIITPet(
        root='/nfs/datasets/torch',
        download=False,
        # transform=T.Compose([
        #     torchvision.transforms.Resize((224, 224)),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     )
        # ]),
        split="test"
    )
    # ds = torch.utils.data.ConcatDataset([ds, personal_train])
    ds = torch.utils.data.ConcatDataset([ds, personal_train, val_ds])

    # ds = torch.utils.data.ConcatDataset([ds, personal_train, val_ds])
    # other_ds = torch.utils.data.ConcatDataset([val_ds, personal_val])
    # val_ds = personal_val
    
    # nways = 129
    # train_nway_ds = NwayDataset(ds, nways, False, ratio=0.1)
    # train_nway_dl = torch.utils.data.DataLoader(train_nway_ds, batch_size=nways, num_workers=32, shuffle=False)
    # train_nway_callback = NwayCallback(train_nway_dl, nways=nways, log_prefix="train/", eval_every=25)

    # val_nway_ds = NwayDataset(val_ds, nways, False, ratio=0.2)
    # val_nway_dl = torch.utils.data.DataLoader(val_nway_ds, batch_size=nways, num_workers=32, shuffle=False)
    # val_nway_callback = NwayCallback(val_nway_dl, nways=nways, log_prefix="val/", eval_every=25)

    print("Making trainer")

    ds = PairDataset(ds, anchor_transform=train_trans, other_transform=train_trans)
    other_ds = torch.utils.data.ConcatDataset([val_ds, personal_val])
    val_ds =PairDataset(
        # val_ds, 
        personal_val, 
        other_ds=other_ds, 
        # other_ds=other_ds, 
        anchor_transform=val_transform, 
        other_transform=val_transform,
        validation=True)

    print("Fit")
    model = SiameseNet("resnet18", 16, 512)
    # model = SiameseNet("resnet50", 16, 512)
    # model = SiameseNet("convnext_tiny", 16, 512)

    dl = torch.utils.data.DataLoader(
        ds, 
        batch_size=128, 
        num_workers=16, 
        pin_memory=True, 
        persistent_workers=True,
        shuffle=True,
        drop_last=True)
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=128, 
        num_workers=16, 
        persistent_workers=True,
        pin_memory=True, 
        shuffle=False)

    trainer_kwargs=dict(
        accelerator="auto", 
        devices=1, 
        # strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
        max_epochs=300,
        gradient_clip_val=1.0,
        # precision=16,
        callbacks=[
            ModelCheckpoint(
                monitor="val/acc",
                save_top_k=1,
                mode="max",
                dirpath="checkpoints",
                filename="best_model"
            ),
            EarlyStopping(
                monitor="val/acc",
                patience=50,
                mode="max"
            ),
            LearningRateMonitor(),
            # train_nway_callback,
            # val_nway_callback,
            # pl.callbacks.Lear
            # pl.callbacks.LearningRateMonitor(logging_interval="step"),
            # StochasticWeightAveraging(1e-4),
        ],
    )
    trainer = Trainer(
        logger=WandbLogger(
            project="siamese-v2",
        ),
        **trainer_kwargs
        )
    trainer.fit(model, dl, val_dataloaders=val_dl)