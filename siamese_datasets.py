import torch
import numpy as np


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, ds, validation = False):
        self.ds = ds
        self.validation = validation
        print("len ds", len(ds))
        print("Creating label groups")
        self.label_groups = {}
        for i, (_, l) in enumerate(self.ds):
            if l not in self.label_groups:
                self.label_groups[l] = []
            self.label_groups[l].append(i)
        # self.label_groups = {label: [i for i, (_, l) in enumerate(self.ds) if l == label] for label in label_values}

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
    def __init__(self, ds):
        self.ds = ds
        print("len ds", len(ds))
        print("Creating label groups")
        self.label_groups = {}
        for i, (_, l) in enumerate(self.ds):
            if l not in self.label_groups:
                self.label_groups[l] = []
            self.label_groups[l].append(i)
        print(f"Len label groups: {len(self.label_groups)}")
        # self.label_groups = {label: [i for i, (_, l) in enumerate(self.ds) if l == label] for label in label_values}

    def __getitem__(self, i):
        anchor, label = self.ds[i]
        is_pos = np.random.random() > 0.5
        if is_pos:
            # get positibe example
            other_idx = np.random.choice(self.label_groups[label])
            other = self.ds[other_idx][0]
            label = 1
        else:
            # get negative example
            potential_labels = list(self.label_groups.keys() - set([label]))
            other_label = np.random.choice(potential_labels)
            other_idx = np.random.choice(self.label_groups[other_label])
            other = self.ds[other_idx][0]
            label = 0

        return (anchor, other), label

    def __len__(self):
        return len(self.ds)

