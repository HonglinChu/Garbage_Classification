import torch
import torchvision
from torchvision import datasets,models,transforms
from  matplotlib import pyplot as plt
import numpy as np 

device=torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

TRAIN='./data/garbage_classify/train'
VAL='./data/garbage_classify/val'

train_data=datasets.ImageFolder(TRAIN)
val_data=datasets.ImageFolder(VAL)
#print(train_data.imgs)

from PIL import Image
fig=plt.figure(figsize=(25,8))

for idx,img in enumerate(train_data.imgs[:9]):
    print(idx)
    img_path=img[0]
    img_name=img_path.split('/')[-1]
    img_idx=img[1]
    
    img=Image.open(img[0])
    ax=fig.add_subplot(3,3,idx+1,xticks=[],yticks=[])
    plt.imshow(img)
    ax.set_title('{}-{}'.format(img_idx,img_name))
    
plt.show()
    








