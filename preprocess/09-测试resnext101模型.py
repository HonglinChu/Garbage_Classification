import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as  plt

import gcnet.resnet as resnet

#(1) read img
img_path='./preprocess/images/cat.jpg'
img=Image.open(img_path)

#(2) preprocess
preprocess=transforms.Compose([
    transforms.Resize((256,256)), #缩放最大边=256
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),# 归一化[0,1]
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]) #标准化
])
input_img=preprocess(img)# c,h,w,=3,224,244
# plt.imshow(input_img)
# plt.show()

input_batch=input_img.unsqueeze(0) #b,c,h,w =1,3,224,224

#(3) load model
model=resnet.resnext101_32x16d_wsl() #下载模型操作  /home/ubuntu/.cache/torch/checkpoints/ig_resnext101_32x16-c6f796b0.pth
model.eval() #最后一层，线性模型输入是2048，输出是1000   (fc): Linear(in_features=2048, out_features=1000, bias=True)
# print(model_ft)

#(4) input
if torch.cuda.is_available():
    input_batch=input_batch.to('cuda')#load image to gpu
    model.to('cuda')
with torch.no_grad():
    output=model(input_batch) #输出1000个类别
print(output[0].shape)

#(5) index
res=torch.nn.functional.softmax(output[0],dim=0)

#(6) result->list
res=res.cpu().numpy().tolist()
max_v=0
index=0
for i ,c in  enumerate(res):
    if max_v<c:
        max_v=c
        index=i
print(max_v,index)

#（7） output name
import codecs
ImageNet1k_label_dict={}
for line in codecs.open('./data/ImageNet1k_label.txt','r'):

    line=line.strip()
   # print(line)
    label_id=line.split(':')[0]
    name=line.split(':')[1]
    name=name.replace('\xa0','') #必须要有这一步操作
    name=name.replace(' ','') 
    #print(name)
    ImageNet1k_label_dict[int(label_id)]=name

print(ImageNet1k_label_dict[index])
    




