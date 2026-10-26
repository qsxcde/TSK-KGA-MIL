import torch
from torch.utils.data import Dataset


class MILDataset(Dataset):
    """Multiple-instance learning dataset with padded bag collation."""

    def __init__(self, bags, labels):
        self.bags = bags
        self.labels = labels

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag_data = torch.tensor(self.bags[idx], dtype=torch.float32)
        bag_label = torch.tensor(self.labels[idx], dtype=torch.long)
        return bag_data, bag_label

    def collate_fn(self, batch):
        data, labels = zip(*batch)
        max_instances = max(d.shape[0] for d in data)
        input_dim = data[0].shape[1]

        padded_data = []
        for bag in data:
            padding = torch.zeros(max_instances - bag.shape[0], input_dim)
            padded_data.append(torch.cat([bag, padding], dim=0))

        return torch.stack(padded_data), torch.stack(labels)
