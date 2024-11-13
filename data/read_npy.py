#导入所需的包
import numpy as np

#导入npy文件路径位置
test = np.load('D:\\Code\\paperCode\\LogCL\\data\\ICEWS14\\his_dict\\train_s_r.npy', allow_pickle=True)

#print("Number of rows:", test.shape[0])
# 只打印前104
#print(test[:104])
print(test)