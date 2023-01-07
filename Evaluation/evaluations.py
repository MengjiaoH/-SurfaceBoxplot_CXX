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

# A Naive recursive Python program to find minimum number
# operations to convert str1 to str2


def editDistance(str1, str2, m, n):

	# If first string is empty, the only option is to
	# insert all characters of second string into first
	if m == 0:
		return n

	# If second string is empty, the only option is to
	# remove all characters of first string
	if n == 0:
		return m

	# If last characters of two strings are same, nothing
	# much to do. Ignore last characters and get count for
	# remaining strings.
	if str1[m-1] == str2[n-1]:
		return editDistance(str1, str2, m-1, n-1)

	# If last characters are not same, consider all three
	# operations on last character of first string, recursively
	# compute minimum cost for all three operations and take
	# minimum of three values.
	return 1 + min(editDistance(str1, str2, m, n-1), # Insert
				   editDistance(str1, str2, m-1, n), # Remove
				   editDistance(str1, str2, m-1, n-1) # Replace
				)


if __name__ == "__main__":
    dataPath = "../datasets/wind_pressure_200/NPY_format/training_data_09.npy"
    modelPath = "./models/weight_decay/model_500_wd_1e3.pth"
    
    start_model = time.time()
    device = torch.device("cuda:0")
    model = SirenNet(15, 1024, 1, 6)
    # model = Network2(15, 3, 3, 128)
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
        input0, target = data[0].to(device), data[1].to(device)
        output = model(input0)
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
    data = np.load(dataPath)
    
    # print(np.min(data[:, 15]), np,max(data[:, 15]))
    for i in range(data.shape[0]):
        target = int((((data[i, 15] - (-1)) * (maxval - minval)) / (1 - (-1))) + minval)
        pred = int((((results[i] - (-1)) * (maxval - minval)) / (1 - (-1))) + minval)
        preds.append(pred)
        targets.append(target)
    targets = np.array(targets)
    preds = np.array(preds)
    diff = np.abs(target - preds)
    error_per_voxel = np.sum(diff) / (15 * 240 * 121)
    targets = np.reshape(targets, (15, 121, 240))
    preds = np.reshape(preds, (15, 121, 240))
    print("error per voxel: ", error_per_voxel)

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

    print("target depth: ")
    print(target_depths)
    print("pred depth: ")
    print(pred_depth)
    print("target sorting: ")
    print(target_sort_index)
    print("pred sorting: ")
    print(pred_sort_index)
    print(editDistance(target_sort_index, pred_sort_index, 15, 15))

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

    


        

