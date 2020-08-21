
import torch
from flask import Flask, request
from gcnet.json_utils import jsonify
from gcnet.train import GarbageClassifier
from gcnet.transforms import transforms_image
import time
from collections import OrderedDict
import codecs

# 获取所有配置参数
app = Flask(__name__)
# 设置编码-否则返回数据中文时候-乱码
app.config['JSON_AS_ASCII'] = False  #防止中文乱码
#
class_id2name = {0:'其他垃圾',1:'厨余垃圾',2:'可回收物',3:'有害垃圾'}

# for line in codecs.open('data/garbage_label.txt', 'r', encoding='utf-8'):
#     line = line.strip()
#     _id = line.split(":")[0]
#     _name = line.split(":")[1]
#     class_id2name[int(_id)] = _name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
print('Pytorch garbage-classification Serving on {} ...'.format(device))
num_classes = len(class_id2name)
model_name = 'resnext101_32x16d'
model_path = './models/checkpoint/checkpoint_8.pth.tar'  #args.resume # --resume checkpoint/garbage_resnext101_model_2_1111_4211.pth
# print("model_name = ",model_name)
# print("model_path = ",model_path)

GCNet= GarbageClassifier(model_name, num_classes,ngpu=0, feature_extract=True)
GCNet.model.to(device)      # 设置模型运行环境
# 如果要使用cpu环境,请指定 map_location='cpu' 
state_dict=torch.load(model_path)['state_dict'] #state_dict=torch.load(model_path)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:] # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
# load params
GCNet.model.load_state_dict(new_state_dict)
GCNet.model.eval()

@app.route('/')
def hello():
    return "Nice To Meet You!"

@app.route('/predict', methods=['POST'])
def predict():
    # 获取输入数据
    file = request.files['image']
    img_bytes = file.read()
    # 特征提取
    feature = transforms_image(img_bytes)
    feature = feature.to(device)  # 在device 上进行预测
    # 模型预测
    with torch.no_grad():
        t1 = time.time()
        outputs = GCNet.model.forward(feature) #????
        consume = (time.time() - t1) * 1000 #ms
        consume = int(consume)

    # API 结果封装
    label_c_mapping = {}
    ## The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    ## 通过softmax 获取每个label的概率
    outputs = torch.nn.functional.softmax(outputs[0], dim=0)

    pred_list = outputs.cpu().numpy().tolist()

    for i, prob in enumerate(pred_list):
        label_c_mapping[int(i)] = prob
    ## 按照prob 降序，获取topK = 4
    dict_list = []
    for label_prob in sorted(label_c_mapping.items(), key=lambda x: x[1], reverse=True)[:4]:
        label = int(label_prob[0])
        result = {'label': label, 'c': label_prob[1], 'name': class_id2name[label]}
        dict_list.append(result)
    ## dict 中的数值按照顺序返回结果
    result = OrderedDict(error=0, errmsg='success', consume=consume, data=dict_list)
    return jsonify(result)

if __name__ == '__main__':
    # curl -X POST -F file=@cat_pic.jpeg http://localhost:5000/predict
    app.run()
