import torch 
from tqdm import tqdm
import time 
import torch.nn 
import os 
from torch.utils.data import Dataset, DataLoader 
import numpy as np 
import config 
from PIL import Image 
import cv2 

class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.calss_name = os.listdir(root_dir)
        self.data = os.listdir(root_dir)
        # files = os.listdir(root_dir)
        # self.data += list(zip(files, [0] * len(files)))
        
        # for index, name in enumerate(self.calss_name):
        #     files = os.listdir(os.path.join(root_dir, name))
        #     self.data += list(zip(files, [index] * len(files)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # img_file, label = self.data[index]
        img_file = self.data[index]
        image = cv2.imread(os.path.join(self.root_dir, img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        both_transform = config.both_transforms(image=image)['image']
        low_res = config.lowres_transform(image=both_transform)['image']
        high_res = config.highres_transform(image=both_transform)['image']
        return low_res, high_res

