import torch
from torch.utils.data import Dataset
import joblib as jb
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class dataset(Dataset):
    def __init__(self, x, y):
        super(dataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx,:], self.y[idx]

    def __len__(self):
        return len(self.y)
    

def get_datasets(path, test_size):

    table = pd.read_csv(path).drop(["ID","日序"],axis=1)
    x = table.values[:,:-1]

    X = np.hstack([np.log(x[:,0]).reshape(-1,1),
                 x[:,1].reshape(-1,1),
                 x[:,2].reshape(-1,1)])

    y = table.values[:,-1].reshape(-1,1)
    y = np.log(y)   

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y)
    jb.dump(y_train,"y_scaler")   # You need to save this for future iniverse tranformation. Since the transformer varies with the different splition of training and testing set, we strongly recommand you save every scaler together with the models trained. 
    x_train, x_test, y_train, y_test = train_test_split(X, y_train, test_size=test_size)

    training_set = dataset(x_train, y_train)
    testing_set = dataset(x_test, y_test)
    
    return training_set, testing_set, y_scaler


class eva_dataset(Dataset):
    def __init__(self, x):
        super(eva_dataset, self).__init__()
        self.x = x

    def __getitem__(self, idx):
        return self.x[idx,:].reshape(-1,1)

    def __len__(self):
        return len(self.x)
    
def get_month_img(DEM_path, VCD_path):
    
    h = np.zeros(25920000)
    alt = np.array(pi.open(DEM_path).getdata())

    raw_img = np.array([alt,h]).transpose()

    column = np.array(pi.open(VCD_path).getdata())
    zero_tag = (column<=0)  
    log_column = np.log(column+zero_tag)
    
    return np.hstack([log_column.reshape(-1,1),raw_img]), zero_tag
