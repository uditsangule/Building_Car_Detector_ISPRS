import os
import time

import cv2
import torch

from utils import *

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ISPRS_Dataset(torch.utils.data.Dataset):
    def __init__(self , image_dir , mask_dir=None,enhance=False,factor=1, cache=True):
        super(ISPRS_Dataset, self).__init__()
        self.image_dir = image_dir
        if mask_dir is not None:
            self.mask_dir = mask_dir
        else:
            self.mask_dir = None
        _ , _ , self.image_name = next(os.walk(image_dir))
        self.enhance=enhance
        self.factor = factor
        
        self.cache = cache
        self.pixel= {0: (0, 0, 0),  # non_objects-->black
                        1: (255, 0, 0),  # building-->red
                        2: (0, 255, 255)}  # cars-->cyan
        self.pixel_inv = {v: k for k, v in self.pixel.items()}
        print(self.pixel_inv)
        if self.cache:
            self.capacity = len(self.image_name) * 1
            print('capacity:',self.capacity)
            self.image_cache = {}
            self.label_cache = {}

    def __len__(self):
        return len(self.image_name) #get length of read)images
    
    def __getitem__(self,idx):
        #idx = random.randint(0, len(self.image_name) - 1)
        if self.cache == True and (idx in self.image_cache.keys()):
            data = self.image_cache[idx]
            print('taken_image from cache',idx)
        else:
            image = cv2.imread(self.image_dir + self.image_name[idx])
            if image.shape[1] > 512:
                image=cv2.resize(image ,(512,512))
                flag=True
            data = 1 / 255 * np.array(image.transpose((2, 0, 1)),
                                        dtype='float32')
            if self.enhance==True:
                data = nn.functional.interpolate(torch.from_numpy(data), mode='linear' ,size=(512,512))
            if self.cache==True :#and idx < self.capacity:
                self.image_cache[idx] = data
        if self.mask_dir is not None:
            if self.cache==True and (idx in self.label_cache.keys()):
                label = self.label_cache[idx]
            else:
                label = cv2.imread(self.mask_dir+self.image_name[idx].replace('_RGB' , '_label'))
                if label.shape[1] > 512:
                    label = cv2.resize(label ,(512,512))
                label = np.array(color_to_label(label, palette=self.pixel_inv) , dtype = 'int64')
                if self.cache==True: #and idx < self.capacity :
                    self.label_cache[idx]= label
            return (torch.from_numpy(data), torch.from_numpy(label))
        return torch.from_numpy(data)

        
if __name__ == '__main__':
    path = os.getcwd()
    size=512
    ds_name = ['Sample_dataset_{}'.format(size) , 'Custom_Dataset_ISPRS_{}'.format(size)]

    ch=0
    img_dir = path+'/{}/train/image/'.format(ds_name[ch])
    mask_dir = path +'/{}/train/mask/'.format(ds_name[ch])
    print(img_dir)
    trainset = ISPRS_Dataset(img_dir,mask_dir, cache=True)
    print(trainset.__len__())
    trainloader = torch.utils.data.DataLoader(trainset , batch_size=1 )
    img = []
    for _ in range(2):
        tic = time.time()
        for i , (image , mask) in enumerate(trainloader):
            if i % 20 ==0:
                print(i)
        print(time.time()-tic)


