import os

import cv2
import numpy as np
from patchify import patchify

from utils import *


def desired_mask(mask):
    mask_pallette = {0: (0, 0, 0),  # non_objects-->black
                     1: (255, 0, 0),  # building-->red
                     2: (0, 255, 255)}  # cars-->cyan

    inversion = {v: k for k, v in mask_pallette.items()}
    return label_to_color(color_to_label(mask,palette=inversion),palette=mask_pallette)


def create_dataset(size):
    root_ds = os.getcwd()
    src_img_dir = root_ds +'/Dataset_ISPRS/2_Ortho_RGB/'
    src_mask_dir = root_ds + '/Dataset_ISPRS/5_Labels_all/'
    dest_dir = root_ds + '/Addition_{}/'.format(size)
    sample_dir = root_ds + '/Sample_aset_{}/'.format(size)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs(sample_dir , exist_ok= True )

    # Create folders to hold images and masks
    folders = ['train/image', 'train/mask', 'valid/image', 'valid/mask', 'test/image',
               'test/mask']
    for f in folders :
        os.makedirs(dest_dir + f , exist_ok=True)
        os.makedirs(sample_dir + f , exist_ok=True)
    ps = size
    #step = 6000-(6000//ps)*ps
    step = 512
    count = discard = 0
    for i, f in enumerate(sorted(os.listdir(src_mask_dir))):
        if f.endswith('.tif'):
            print(f)
            large_image = cv2.imread(src_img_dir+f.replace('_label' , '_RGB'))
            print(large_image.shape)
            large_mask = cv2.imread(src_mask_dir+f)
            patches_img = patchify(large_image , (ps,ps,3) , step=step)
            patches_mask =patchify(large_mask , (ps,ps,3) , step=step)

            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    new_mask = desired_mask(patches_mask[i][j][0])
                    bg_cnt = np.sum(new_mask == np.array((0,0,0)))/(new_mask.shape[0]*3*new_mask.shape[1])
                    if bg_cnt > 0.92 and bg_cnt < 0.95:
                        count += 1
                        #print(bg_cnt) #should be less than 0.98 [1 means all black]
                        if random.randint(1,8) == 3:
                            cv2.imwrite(dest_dir+'test/image/'+f.replace('_label.tif' , '_1_['+str(i)+'_'+str(j)+']_RGB.png'), np.array(patches_img[i][j][0]))
                            cv2.imwrite(dest_dir+'test/mask/'+f.replace('_label.tif' , '_1_['+str(i)+'_'+str(j)+']_label.png'), new_mask)
                            if random.randint(1,4) == 3:
                                cv2.imwrite(sample_dir + 'test/image/' + f.replace('_label.tif', '1['+str(i)+'_'+str(j)+']_RGB.png'), np.array(patches_img[i][j][0]))
                                cv2.imwrite(sample_dir + 'test/mask/' + f.replace('_label.tif', '1['+str(i)+'_'+str(j)+']_label.png'), new_mask)
                        else:
                            cv2.imwrite(dest_dir + 'train/image/' + f.replace('_label.tif','_1_[' + str(i) + '_' + str(j) + ']_RGB.png'),np.array(patches_img[i][j][0]))
                            cv2.imwrite(dest_dir + 'train/mask/' + f.replace('_label.tif', '_1_['+str(i)+'_'+str(j)+']_label.png'),new_mask)
                            if random.randint(1,4) == 3 :
                                cv2.imwrite(sample_dir + 'train/image/' + f.replace('_label.tif', '1['+str(i)+'_'+str(j)+']_RGB.png'), np.array(patches_img[i][j][0]))
                                cv2.imwrite(sample_dir + 'train/mask/' + f.replace('_label.tif', '1['+str(i)+'_'+str(j)+']_label.png'), new_mask)

                    else:
                        discard+=1

    print(count ,discard)

if __name__ == '__main__':
    create_dataset(size = 512)
    #create_dataset(size=1024)
