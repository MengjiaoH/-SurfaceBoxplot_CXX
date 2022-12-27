import numpy as np
import time 
import os
import torch 
from torch.utils.data.dataset import Dataset 
from torch.utils.data import DataLoader



class LoadData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir 
        self.data = np.load(data_dir)
        self.l = self.data.shape[1]
        # depths = self.data[:, self.l - 1]
        # minDepth = np.min(depths)
        # maxDepth = np.max(depths)
        # minval = -1
        # maxval = 1
        # self.data[:, self.l - 1] = (((self.data[:, self.l-1] - minDepth) * (maxval - minval)) / (maxDepth- minDepth)) + minval
        # print(np.min(self.data[:, self.l-1]), np.max(self.data[:, self.l-1]))
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index, :]
        
        #input_data = torch.FloatTensor(data[0:self.l-1])
        
        input0 = torch.FloatTensor([data[0]])
        input1 = torch.FloatTensor(data[1:self.l-1])
        output_data = torch.FloatTensor([data[self.l-1]])
        
        return input0, input1, output_data
    

if __name__ == "__main__":
    data_dir = "../build/training_data.txt"
    train_dataset = LoadData(data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
    input_d, output_d = next(iter(train_dataloader))
    print(input_d.shape, output_d.shape)
