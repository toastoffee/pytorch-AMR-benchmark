import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as scio

# dataset_path = "../datasets/RML2016.10a_total_data.mat"
dataset_path = "./datasets/RML2016.10a_total_data.mat"


class RML2016aDataset(Dataset):
    def __init__(self):
        class_num = 11
        sample_num = 220000

        dataset_dict = scio.loadmat(dataset_path)
        self.data = torch.from_numpy(dataset_dict['data'])

        index_labels = dataset_dict['label']
        one_hot_labels: np.ndarray = np.eye(class_num)[index_labels]
        one_hot_labels = np.squeeze(one_hot_labels)

        self.labels = torch.from_numpy(one_hot_labels)
        self.snr = torch.from_numpy(dataset_dict['snr']).t()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item], self.snr[item]


if __name__ == "__main__":
    dataset = RML2016aDataset()
    pass