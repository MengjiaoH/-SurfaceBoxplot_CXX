import numpy as np 
import matplotlib.pyplot as plt 

data_dir = "../datasets/wind_pressure_200/NPY_format/"
for dd in range(45):
    temp = data_dir + "training_data_" + str(dd).zfill(2) + ".npy"
    
    data = np.load(temp)
    print("data shape", data.shape)

    depths = []
    for d in data:
        depths.append(d[15])

    depths = np.array(depths)
    depths = np.reshape(depths, (15, 121, 240))

    c = 15
    r = 15
    fig, axs = plt.subplots(r, c, figsize=(20, 20), sharey=True)

    for i in range(15):
        depth = depths[i, :, :]
        for j in range(15):
            depth1 = depths[j, :, :]
            diff = np.abs(depth1 - depth)
            row = i 
            col = j 

        # row = int(i % r)
        # col = int(i / r)
        # print(row, col)
        
            img = axs[row, col].imshow(diff, vmin=-1, vmax=1, cmap="rainbow")
            # axs[row, col].set_title('Member ' + str(i+1))

    fig.tight_layout()
    # plt.show()
    plt.savefig('ts_depth_diff_' + str(dd) + '.png')