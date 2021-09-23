
import os
import torch
import cv2
import numpy as np
from keras.utils import normalize
#from utils import *
from matplotlib import pyplot as plt
from patchify import patchify , unpatchify
import tifffile
def check_slice_stick(slice , dir_path):
	patch_size= 608
	img_dataset = []
	for i , image_f in enumerate(os.listdir(dir_path)):
		print(i, image_f)
		if image_f.endswith('.tif') and i*2 <= 8:

			image = tifffile.imread(dir_path+'/'+image_f)
			print(image.shape)
			#size_x = (image.shape[1]//patch_size)*patch_size
			#size_y = (image.shape[0] // patch_size) * patch_size
			patches = patchify(image , (patch_size,patch_size , 3), step=528)
			predicted_patches = []
			print(patches.shape)
			for i in range(patches.shape[0]):
				for j in range(patches.shape[1]):
					#print(i,j)
					single_patch = patches[i, j, :, :]
					#single_patch = np.expand_dims(single_patch , 0)
					#img_dataset.append(single_patch)
					print(single_patch.shape)
					plt.figure(figsize=(5,5))
					plt.imshow(single_patch[0])

					#cv2.imshow('image', single_patch)
					#cv2.waitKey(0)
					#single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1), 2)
					#single_patch_input = np.expand_dims(single_patch_norm, 0)

					# Predict and threshold for values above 0.5 probability
					#single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(np.uint8)
					#predicted_patches.append(single_patch_prediction)
			else:
				continue
		plt.show()
		cv2.destroyAllWindows()

		print(len(img_dataset))
			#predicted_patches = np.array(predicted_patches)

def check_sample(image):
	img = cv2.imread(image)
	print(img.shape)
	img = cv2.resize(img , (1024 , 1024))
	#cv2.imshow('frame' , img)
	#cv2.waitKey(0)
	cv2.destroyAllWindows()
	print(img.shape)
	ps =512
	patches = patchify(img , (ps,ps) , step=512)
	print(patches.shape)


def check_read():
	path = '/media/udit/LENOVO_USB_HDD/Github_projects/Building and Car Detection Evigway/AerialImageDataset/test/images/bellingham2.tif'
	img = cv2.imread(path)
	print(img.shape)
	print(img.dtype)
	cv2.imshow('tif', img)
	cv2.waitKey(0)
	cv2.DestroyAllWindows()

if __name__ == '__main__':
	root_DS = os.getcwd()
	Datset_list = ['potsdam']
	Labels_list = {0: 'non_object', 1: 'buildings', 2: 'cars'}
	n_class = 2
	weight = torch.ones(n_class)
	data_folder = root_DS + '2_Ortho_RGB/top_potsdam_{}_RGB.tif'
	label_folder = root_DS + '5_Labels_all/top_potsdam_{}_label.tif'
	Windows = (512, 512)
	mask_pallette = {0: (0, 0, 0),  # non_objects-->black
					 1: (0, 0, 255),  # building-->blue
					 2: (255, 255, 0)}  # cars-->yellow'''

	inversion = {v: k for k, v in mask_pallette.items()}
	#check_sample(root_DS+'/Sample_2.png')
	check_slice_stick(slice=1, dir_path=root_DS+'/Dataset_ISPRS/2_Ortho_RGB/')
'''def train():
    trainset = ISPRS_Dataset(img_dir, mask_dir)
    print(trainset.__len__())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    print(summary(model1 , (3,input_size,input_size)))
    total_loss = []
    start_time = timeit.default_timer()
    for e in range(epoch):
        for i , (image , mask ) in enumerate(trainloader):
            if device == 'cuda:0':
                image = image.to(device)
                mask = mask.to(device)
            optimizer.zero_grad()
            mask_pred = model1(image)
            mask_pred = mask_pred.squeeze(0)
            nw_loss = criterion(mask_pred,mask)
            total_loss.append(nw_loss.item())
            nw_loss.backward()
            if i%100 == 0:
                print('[epoch:{}/{} loss:{}]'.format(e , epoch , nw_loss.item()))
        torch.save(model1.state_dict() , model_path)
        plt.figure(figure=(10,10))
        plt.plot(total_loss)
        plt.show()
    print('Training_Ends:{} at {}'.format('unet' , (timeit.default_timer()-start_time)/60.))'''