import os 
import sys 
import numpy as np 
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm 
from data_loader import LoadData 
from network import SirenNet
from network2 import Network2
from utils import EarlyStopping, LRScheduler
from sklearn.model_selection import KFold

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

lr = 1e-05
batch_size = 1000
nepochs = 100
use_lr_scheduler = True
train_data_path = "./datasets/wind_pressure_200/training_data.npy"
start_epoch = 0
data_set = "wind_pressure_200_5"
k_folds = 3

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
checkpoint_dir = os.path.join("checkpoints")

if not os.path.exists(os.path.join("checkpoints", data_set)):
    os.mkdir(os.path.join("checkpoints", data_set))
checkpoint_dir = os.path.join("checkpoints", data_set)

prefix = "mlp"

if not os.path.exists(os.path.join(checkpoint_dir, prefix)):
    os.mkdir(os.path.join(checkpoint_dir, prefix))
checkpoint_dir = os.path.join(checkpoint_dir, prefix)

manualSeed = np.random.randint(0, 9999999, 1)
print("seed: ", manualSeed)
torch.manual_seed(manualSeed)

##! device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

dataset = LoadData(train_data_path)
# train_size = int(0.9 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
kfold = KFold(n_splits=k_folds, shuffle=True)

#model = SirenNet(15, 1024, 1, 6)
model = Network2(15, 3, 3, 128)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)

L1_Loss = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-06)

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=4, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler, num_workers =4, pin_memory = True)
    for epoch in range(start_epoch, start_epoch + nepochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for data0, data1, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                data0, data1, target = data0.to(device), data1.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data0, data1)
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                loss = L1_Loss(output, target)
                correct = (predictions == target).sum().item()
                accuracy = correct / batch_size
                writer.add_scalar("Loss/train", loss, epoch)
                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
        if (epoch + 1) % 1 == 0:
            with tqdm(test_dataloader, unit="batch") as testepoch:
                for test_data0, test_data1, test_target in testepoch:
                    testepoch.set_description(f"Epoch {epoch}")

                    test_data0, test_data1, test_target = test_data0.to(device), test_data1.to(device), test_target.to(device)
                   
                    test_output = model(test_data0, test_data1)
                    test_predictions = output.argmax(dim=1, keepdim=True).squeeze()
                    test_loss = L1_Loss(test_output, test_target)
                    correct = (test_predictions == test_target).sum().item()
                    accuracy = correct / batch_size
                    writer.add_scalar("Loss/test", test_loss, epoch)    
                    testepoch.set_postfix(test_loss=test_loss.item(), accuracy=100. * accuracy)
        if (epoch + 1) % 10 == 0:
            path = os.path.join(checkpoint_dir, "model_" + str(epoch+1) + ".pth")
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), path)
            else:
                torch.save(model.state_dict(), path)

