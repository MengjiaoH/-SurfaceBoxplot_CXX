import numpy as np 
import matplotlib.pyplot as plt 


data = np.loadtxt("./total_depth.txt")

num_ts = data.shape[0] // 2

width = 0.35

for i in range(45):
    target = data[2 * i + 0]
    pred = data[2 * i + 1]
    diff = np.sum(target - pred)
    print(i, diff/(240 * 121))

'''

labels = []
for m in range(15):
    labels.append('M' + str(m+1))
print(labels)

x = np.arange(len(labels))  # the label locations

c = 5
r = 4
fig, axs = plt.subplots(r, c, sharey=True)

for i in range(20):
    temp = i + 25
    target = data[2 * temp, :]
    pred = data[2 * temp + 1, :]
    
    row = int(i % r)
    col = int(i / r)
    # print(row, col)
    
    rects1 = axs[row, col].bar(x - width/2, target, width, label='Target')
    rects2 = axs[row, col].bar(x + width/2, pred, width, label='Predict')
    # axs[row, col].set_ylabel('Total Depth')
    axs[row, col].set_title('Time step ' + str(temp+1))
    axs[row, col].set_xticks(x, labels)
    # axs[row, col].legend()

    # axs[row, col].bar_label(rects1, padding=3)
    # axs[row, col].bar_label(rects2, padding=3)

# fig.tight_layout()
plt.show()

'''
