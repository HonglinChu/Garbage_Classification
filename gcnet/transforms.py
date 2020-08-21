#(1) transforms 
import io
import torchvision.transforms as transforms
from PIL import Image

#数据预处理
preprocess=transforms.Compose([
        transforms.Resize((256,256)), #缩放最大边=256
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),# 归一化[0,1]
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]) #标准化
    ])

def transforms_image(img_bytes):
    
    #img=Image.open(io.Bytes(img_bytes))# 转换成图片,import io
    img = Image.open(io.BytesIO(img_bytes))
    img=preprocess(img) #图片预处理
    img_batch=img.unsqueeze(0) #b c h w 
    return img_batch

