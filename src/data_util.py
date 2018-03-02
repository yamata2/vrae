import glob
import numpy as np
from IPython import embed

def read_target(data_dir, data_dim):
    filenames = glob.glob(data_dir + "/*.npy")
    filenames.sort()

    min_len = 100000
    for f in filenames:
        length = np.load(f).shape[1]
        if length < min_len:
            min_len = length
            
    data  = []
    for f in filenames:
        idx = np.load(f)[2, :min_len]
        idx = np.array(idx, dtype=np.int32)
        one_hot = np.identity(data_dim)[idx]
        data.append(one_hot)

    data = np.array(data)
    data = np.transpose(data, [1,0,2])
    
    return data
