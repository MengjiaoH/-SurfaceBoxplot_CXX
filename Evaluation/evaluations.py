import numpy as np 
import time
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import sys 
sys.path.append("../Training/")
from data_loader import LoadData 
from network import SirenNet
from network2 import Network2

import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataPath = "../datasets/wind_pressure_200/Lead_10.txt"
    modelPath = "./models/model_mlp.pth"
    
    start_model = time.time()
    device = torch.device("cuda:0")
    # model = SirenNet(15, 1024, 1, 6)
    model = Network2(15, 3, 3, 128)
    if device == torch.device("cpu"):
        model.load_state_dict(torch.load(modelPath, map_location=device))
    else:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(modelPath))
    model.to(device)
    end_model = time.time()
    print("loading model time:", end_model - start_model)

    start_loader = time.time()
    dataset = LoadData(dataPath)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=4, drop_last=False)
    end_loader = time.time()
    print("loading data time:", end_loader - start_loader)

    start_infer = time.time()
    for d, data in enumerate(dataloader):
        input, target = data[0].to(device), data[1].to(device)
        output = model(input)
        output_cpu = output.detach().cpu().numpy()
        if d == 0:
            results = output_cpu
        else:
            results = np.concatenate((results, output_cpu), axis=0)
    end_infer = time.time()
    print("inference time: ", end_infer - start_infer)

    ## calculate difference 
    minval = 0
    maxval = 91 ## number of combinations if select 2 out of 14
    preds = []
    targets = []
    data = np.loadtxt(dataPath)
    for i in range(data.shape[0]):
        target = data[i, 15]
        pred = int((((results[i] - (-1)) * (maxval - minval)) / (1 - (-1))) + minval)
        preds.append(pred)
        targets.append(target)
    targets = np.array(targets)
    preds = np.array(preds)

    targets = np.reshape(targets, (15, 121, 240))
    preds = np.reshape(preds, (15, 121, 240))

    # minval = np.min(targets)
    # maxval = np.max(targets)
    # targets = ((targets - minval) / (maxval - minval)) * (1 - 0) + 0
    # minval = np.min(preds)
    # maxval = np.max(preds)
    # preds = ((preds - minval) / (maxval - minval)) * (1 - 0) + 0
    # diffs =np.abs(targets - preds)

    target_depths = []
    pred_depth = []

    for i in range(15):
        target = targets[i, :, :]
        target_sum = np.sum(target)
        pred = preds[i, :, :]
        pred_sum = np.sum(pred)
        target_depths.append(target_sum)
        pred_depth.append(pred_sum)

    target_sort_index = np.argsort(np.array(target_depths))
    pred_sort_index = np.argsort(np.array(pred_depth))

    print(target_depths)
    print(pred_depth)
    print(target_sort_index)
    print(pred_sort_index)

    '''
    fig, axs = plt.subplots(5, 3)
    for i in range(5):
        axs[i, 0].set_aspect("equal")
        axs[i, 1].set_aspect("equal")
        axs[i, 2].set_aspect("equal")
        axs[i, 0].imshow(targets[i, :, :])
        axs[i, 1].imshow(preds[i, :, :])
        axs[i, 2].imshow(diffs[i, :, :])

    plt.show()
    '''

    


        

