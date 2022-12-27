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
from utils import EarlyStopping, LRScheduler

lr = 1e-04
batch_size = 1000
nepochs = 100
use_lr_scheduler = True
train_data_path = "./datasets/training_data.npy"
start_epoch = 0
data_set = "wind_pressure_200_22"

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
checkpoint_dir = os.path.join("checkpoints")

if not os.path.exists(os.path.join("checkpoints", data_set)):
    os.mkdir(os.path.join("checkpoints", data_set))
checkpoint_dir = os.path.join("checkpoints", data_set)

prefix = "models"

if not os.path.exists(os.path.join(checkpoint_dir, prefix)):
    os.mkdir(os.path.join(checkpoint_dir, prefix))
checkpoint_dir = os.path.join(checkpoint_dir, prefix)

manualSeed = 999
torch.manual_seed(manualSeed)

##! device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

dataset = LoadData(train_data_path)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)

model = SirenNet(15, 1024, 1, 6)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)

L1_Loss = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-06)

for epoch in range(start_epoch, start_epoch + nepochs):
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            loss = L1_Loss(output, target)
            correct = (predictions == target).sum().item()
            accuracy = correct / batch_size

            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
    if (epoch + 1) % 10 == 0:
        with tqdm(test_dataloader, unit="batch") as testepoch:
            for test_data, test_target in testepoch:
                testepoch.set_description(f"Epoch {epoch}")

                test_data, test_target = test_data.to(device), test_target.to(device)
               
                test_output = model(test_data)
                test_predictions = output.argmax(dim=1, keepdim=True).squeeze()
                test_loss = L1_Loss(test_output, test_target)
                correct = (test_predictions == test_target).sum().item()
                accuracy = correct / batch_size
                
                testepoch.set_postfix(test_loss=test_loss.item(), accuracy=100. * accuracy)
    