import torch
import os
import yaml
from torch.utils.data import Dataset


class PregeneratedDataset(Dataset):

    def __init__(self, dataset_path, modifiers_dim):

        self.dataset_path = dataset_path

        dataset_info_path = os.path.join(dataset_path, 'dataset_info.yaml')
        with open(dataset_info_path, 'r') as yaml_file:
            dataset_info = yaml.load(yaml_file)
        self.dataset_info = dataset_info
        self.modifiers_dim = modifiers_dim

    def __getitem__(self, index):

        data_path = os.path.join(self.dataset_path, 'data_' + str(index) + '.pt')
        labels_path = os.path.join(self.dataset_path, 'labels_' + str(index) + '.pt')
        data = torch.load(data_path)
        labels = torch.load(labels_path)

        # Truncate brutally. This is fine as modifiers_dim >> actual modifier encoding (most dimensions are empty)
        if labels.shape[1] > self.modifiers_dim:
            labels = labels[:, :self.modifiers_dim]


        return data, labels

    def __len__(self):
        return self.dataset_info['Size']

    def __str__(self):
        return 'Pregenerated_' + self.dataset_info['Type']

    def summary(self):
        description = self.dataset_info
        description['Type'] = 'Pregenerated_' + self.dataset_info['Type']
        return description
