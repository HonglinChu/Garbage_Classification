
import torch

import torchvision.transforms as transforms
from PIL import Image

#(1) transforms 
import io
def transforms_image(img_bytes):
    preprocess=transforms.Compose([
        transforms.Resize((256,256)), #缩放最大边=256
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),# 归一化[0,1]
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]) #标准化
    ])
    #img=Image.open(io.Bytes(img_bytes))# 转换成图片,import io
    img = Image.open(io.BytesIO(img_bytes))
    img=preprocess(img) #图片预处理
    img_batch=img.unsqueeze(0) #b c h w 
    return img_batch

#(2) Get Imagenet1k_label
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

#(3) APP 
import time
from flask import Flask,request,jsonify
from collections import OrderedDict

app=Flask(__name__)

#(4) testing example
@app.route('/') #默认 methods='GET' #testing #通过curl http://...... 来访问app
def hello():
    return 'Nice To Meet You!'

#(5)  predict network
#(5.1)load model and predict
import gcnet.resnet as resnet
device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
model= resnet.resnext101_32x16d_wsl()
model.to(device)
model.eval() # 预测问题，指定eval

#@app.route 类似于c语言中的 case
@app.route('/predict',methods=['POST']) #客户端的请求中的url在服务器端的python文件中的@app.route()中寻找
def predict():
    #(5.2)get input data
    file=request.files['image']
    bytes=file.read()
    img_batch=transforms_image(img_bytes=bytes)
    input_batch=img_batch.to(device)

    #(5.3) predict
    with torch.no_grad():
        t1=time.time()
        output=model(input_batch)
        t2=time.time()
        consume=int((t2-t1)*1000) # calculate time-ms

    #(5.4) API 
    pre=torch.nn.functional.softmax(output[0],dim=0)
    pre_list=pre.cpu().numpy().tolist()
    pre_dict={}
    for i ,v in enumerate(pre_list):
        pre_dict[i]=v

    #(5.5) filter
    data_list=[]
    topK=4
    for label_prob in sorted(pre_dict.items(),key=lambda x:x[1], reverse=True)[:topK]: #升序排列,找到前4个
        label = int(label_prob[0])
        res = {'label':label,'prob':label_prob[1],'name':ImageNet1k_label_dict[label]}
        data_list.append(res)

    #(5.6) return JSON format output
    result=OrderedDict(error=0,errmsg='success',consume=consume,data=data_list)
    return jsonify(result)

if  __name__=='__main__':
    app.run()   