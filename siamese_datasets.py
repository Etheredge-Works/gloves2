from attr import has
import torch
import numpy as np
from multiprocessing import Pool
import os
import torchvision


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, ds, validation = False):
        self.ds = ds
        self.validation = validation
        print("len ds", len(ds))
        print("Creating label groups")
        self.label_groups = {label: [] for label in self.ds.class_to_idx.values()}

        for i, (_, l) in enumerate(self.ds):
            self.label_groups[l].append(i)

    def __getitem__(self, i):
        anchor, label = self.ds[i]
        # if self.validation:
            # np.random.seed(i)
        # get positive example
        pos_idx = i
        # while pos_idx == i: TODO could do this but rand aug will be enough
        pos_idx = np.random.choice(self.label_groups[label])

        pos = self.ds[pos_idx][0]
        # get negative example
        potential_labels = list(self.label_groups.keys() - set([label]))
        neg_label = np.random.choice(potential_labels)
        neg_idx = np.random.choice(self.label_groups[neg_label])
        neg = self.ds[neg_idx][0]
        
        return (anchor, pos, neg), label

    def __len__(self):
        return len(self.ds)


class PairDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        anchor_ds, 
        other_ds = None,
        anchor_transform=None, 
        other_transform=None,
        validation=False,
    ):
        self.ds = anchor_ds
        if other_ds is None:
            self.other_ds = anchor_ds
        else:
            self.other_ds = other_ds
            

        self.anchor_transform = anchor_transform
        self.other_transform = other_transform
        self.validation = validation

        print("len ds", len(anchor_ds))
        try: 
            self.label_groups = {label: [] for label in self.ds.class_to_idx.values()}
        except AttributeError:
            self.label_groups = {}
        # print(sorted(self.label_groups.keys()))
        print("Created label groups")

        # Disable transform for SPEED
        if isinstance(self.ds, torchvision.datasets.ImageFolder) or isinstance(self.ds, torchvision.datasets.ImageNet):
            print("ImageFolder")
            transform = self.ds.transform
            self.ds.transform = None
            loader = self.ds.loader
            self.ds.loader = lambda x: x
        elif isinstance(self.ds, torchvision.datasets.VisionDataset):
            print("VisionDataset")
            transform = self.ds.transforms
            self.ds.transforms = None
        else:
            print("other")

        for i, (_, l) in enumerate(self.other_ds):
            if l not in self.label_groups:
                self.label_groups[l] = [] # shouldn't happen but does
            self.label_groups[l].append(i)
        print(f"Len label groups: {len(self.label_groups)}")

        # Re-enable transform
        if isinstance(self.ds, torchvision.datasets.ImageFolder):
            self.ds.transform = transform
            self.ds.loader = loader
        elif isinstance(self.ds, torchvision.datasets.VisionDataset):
            self.ds.transforms = transform

    def __getitem__(self, i):
        if self.validation:
            np.random.seed(i)
            # This fixes the randomness of the training dataset as well
            # torch.manual_seed(i)

        anchor, label = self.ds[i]
        is_pos = np.random.random() > 0.5
        if is_pos:
            # get positive example
            other_idx = np.random.choice(self.label_groups[label])
            other = self.other_ds[other_idx][0]
            label = 1
        else:
            # get negative example
            potential_labels = list(self.label_groups.keys() - set([label]))
            other_label = np.random.choice(potential_labels)
            other_idx = np.random.choice(self.label_groups[other_label])
            other = self.other_ds[other_idx][0]
            label = 0

        if self.anchor_transform is not None:
            anchor = self.anchor_transform(anchor)
        if self.other_transform is not None:
            other = self.other_transform(other)
        
        if self.validation:
            # This fixes the randomness of the training dataset as well
            np.random.seed()

        return (anchor, other), label

    def __len__(self):
        return len(self.ds)

