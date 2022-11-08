import torch
import numpy as np
import pytorch_lightning as pl


class NwayDataset(torch.utils.data.Dataset):
    def __init__(self, ds, nways=17, validation = False, ratio=1.0):
        self.ds = ds
        self.ratio = ratio
        self.idx_mapping = list(range(len(self.ds)))
        self.idx_mapping = np.random.choice(self.idx_mapping, int(len(self.ds) * ratio), replace=False)

        # if shuffle_ds:
        #     np.random.shuffle(self.ds)
        self.nways= nways
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
        base_idx = i // self.nways
        idx = self.idx_mapping[base_idx]
        anchor, label = self.ds[idx]

        if i % self.nways == 0:
            return anchor, 0
        # if self.validation:
            # np.random.seed(i)
        # get positive example
        if i % self.nways == 1:
            pos_idx = idx
            while pos_idx == idx: # TODO could do this but rand aug will be enough
                pos_idx = np.random.choice(self.label_groups[label])

            pos = self.ds[pos_idx][0]
            return pos, torch.tensor(1, dtype=torch.long)

        # get negative example
        potential_labels = list(self.label_groups.keys() - set([label]))
        neg_label = np.random.choice(potential_labels, )
        neg_idx = np.random.choice(self.label_groups[neg_label])
        neg = self.ds[neg_idx][0]
        return neg, torch.tensor(2, dtype=torch.long)


        
        return (anchor, pos, neg), label
    
    # def collate_fn(self, batch):
    #     anchors, positives, negatives = [], [], []
    #     for (anchor, pos, neg), _ in batch:
    #         anchors.append(anchor)
    #         positives.append(pos)
    #         negatives.append(neg)
    #     return (torch.stack(anchors), torch.stack(positives), torch.stack(negatives)), None

    def __len__(self):
        return int(len(self.ds)*self.ratio) * self.nways


class NwayCallback(pl.Callback):
    def __init__(
        self, 
        nway_dl, 
        dist_fn=None, 
        evaluation_limit=128,
        nways=17,
        log_prefix="train/",
        eval_every=1,
    ):
        self.nway_dl = nway_dl
        if dist_fn is None:
            self.dist_fn = torch.norm
        self.log_prefix = log_prefix
        self.nways = nways
        self.evaluation_limit = evaluation_limit
        self.eval_every = eval_every

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch % self.eval_every == 0:
            self.evaluate(trainer, pl_module)

    def evaluate(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.nway_dl):
                
                imgs, labels = batch
                # labels = labels.to("cpu")
                # assert labels[0] == 0, f"anchor label is {labels[0]}"
                # assert labels[1] == 1, f"pos label is {labels[1]}"
                # assert labels[2] == -1, f"neg label is {labels[2]}"
                # assert labels[3] == -1, f"neg label is {labels[3]}"

                imgs = imgs.to(pl_module.device)
                embs = pl_module(imgs)

                anchor_out = embs[0:1]
                pos_out = embs[1:2]
                negs_out = embs[2:]

                distances = torch.cat([self.dist_fn(anchor_out - pos_out, dim=1), self.dist_fn(anchor_out - negs_out, dim=1)], dim=0)
                # print(distances.shape)

                acc = (distances[0] < distances[1:]).sum().item() 

                # print(f"nway acc shape: {acc.shape}")
                # print(f"nway acc: {acc}")
                acc = int(acc == self.nways - 2)
                # print(f"nway acc: {acc}")
                pl_module.log(f"{self.log_prefix}nway_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

                pl_module.log(f"{self.log_prefix}nway_pos_dist", distances[0], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                pl_module.log(f"{self.log_prefix}nway_avg_neg_dists", torch.mean(distances[1:]), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                avg_dists = torch.mean(distances[1:] - distances[0])
                pl_module.log(f"{self.log_prefix}nway_pos_neg_dist_diff", avg_dists, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

                # if i == self.evaluation_limit:
                #     break
        pl_module.train()



        return super().on_train_epoch_end(trainer, pl_module)
