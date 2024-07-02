import torch
from torch.utils.data import Dataset
import scipy.io as scio

# dataset_path = "../datasets/RML2016.10a_total_data.mat"
dataset_path = "./datasets/RML2016.10a_total_data.mat"


class RML2016aDataset(Dataset):
    def __init__(self):
        dataset_dict = scio.loadmat(dataset_path)
        self.data = torch.from_numpy(dataset_dict['data'])
        self.labels = torch.from_numpy(dataset_dict['label']).t()
        self.snr = torch.from_numpy(dataset_dict['snr']).t()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item], self.snr[item]


if __name__ == "__main__":
    dataset = RML2016aDataset()