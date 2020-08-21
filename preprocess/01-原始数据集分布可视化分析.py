#（1）
import os
import cv2
from os import walk
data_path='./data/garbage_classify/raw_data'
for  (dirpath,dirnames,filenames) in walk(data_path):
    print('dirpath:',dirpath)
    # print('dirnames:',dirnames)  sub dir
    print('filenames:',filenames[:5])

#（2）二级分类不同类别的分布
from glob import glob
data_path='./data/garbage_classify/raw_data'
def get_image_txt_info(path):
    data_path_txt=glob(os.path.join(path,'*.txt')) #all txt files
    img_list=[]
    img2label_dic={}
    label_count_dic={}
    for txt_path in data_path_txt:
        with open(txt_path,'r') as f:
            line=f.readline()# read a line
        #print(line)
        line=line.strip()# delete pre ' ' and  last ' 
        img_name=line.split(',')[0] # img_2778.jpg
        img_label=int(line.split(',')[1]) # 7
        img_name_path=os.path.join(data_path,img_name)   #image
        #print(img_name_path)#
        img_list.append({'img_name_path':img_name_path,'img_label':img_label})
        #image_name:img_label
        img2label_dic[img_name]=img_label
        #[img_label ,img_count] statistic 
        img_label_count=label_count_dic.get(img_label,0)#不存在则初始化为0
        if img_label_count:
            label_count_dic[img_label]+=1
        else:
            label_count_dic[img_label]=1
    return img_list,img2label_dic,label_count_dic
img_list,img2label_dic,label_count_dic=get_image_txt_info(data_path)


#可视化每一个标签对应的label，通过pyecharts绘制图表

from pyecharts import options as opts #pyecharts相关参数
from pyecharts.charts import Bar  
label_dict = {
    "0": "其他垃圾/一次性快餐盒",
    "1": "其他垃圾/污损塑料",
    "2": "其他垃圾/烟蒂",
    "3": "其他垃圾/牙签",
    "4": "其他垃圾/破碎花盆及碟碗",
    "5": "其他垃圾/竹筷",
    "6": "厨余垃圾/剩饭剩菜",
    "7": "厨余垃圾/大骨头",
    "8": "厨余垃圾/水果果皮",
    "9": "厨余垃圾/水果果肉",
    "10": "厨余垃圾/茶叶渣",
    "11": "厨余垃圾/菜叶菜根",
    "12": "厨余垃圾/蛋壳",
    "13": "厨余垃圾/鱼骨",
    "14": "可回收物/充电宝",
    "15": "可回收物/包",
    "16": "可回收物/化妆品瓶",
    "17": "可回收物/塑料玩具",
    "18": "可回收物/塑料碗盆",
    "19": "可回收物/塑料衣架",
    "20": "可回收物/快递纸袋",
    "21": "可回收物/插头电线",
    "22": "可回收物/旧衣服",
    "23": "可回收物/易拉罐",
    "24": "可回收物/枕头",
    "25": "可回收物/毛绒玩具",
    "26": "可回收物/洗发水瓶",
    "27": "可回收物/玻璃杯",
    "28": "可回收物/皮鞋",
    "29": "可回收物/砧板",
    "30": "可回收物/纸板箱",
    "31": "可回收物/调料瓶",
    "32": "可回收物/酒瓶",
    "33": "可回收物/金属食品罐",
    "34": "可回收物/锅",
    "35": "可回收物/食用油桶",
    "36": "可回收物/饮料瓶",
    "37": "有害垃圾/干电池",
    "38": "有害垃圾/软膏",
    "39": "有害垃圾/过期药物"
}

#import matplotlib.pyplot as  plt
#首先对label_count_dic 按照key进行排序
label_count_dic=dict(sorted(label_count_dic.items()))#默认按照key进行排序
#x=label_count_dic.keys() #0-39  
x=['{}-{}'.format(label_index,label_dict[str(label_index)]) for label_index in label_count_dic]
y=label_count_dic.values()#count
x=list(x)
y=list(y)
#初始化
bar=Bar(init_opts=opts.InitOpts(width='1100px',height='500px'))
bar.add_xaxis(xaxis_data=x)
bar.add_yaxis(series_name='',yaxis_data=y)
#设置全局变量
bar.set_global_opts(
    title_opts=opts.TitleOpts(title='垃圾分类-不同类别的数据分布'),        #增加标题
    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15)) #
    )
#展示我们的图表
# bar.load_javascript()
# bar.render_notebook()
bar.render('./preprocess/01.html')
#print('Done!')

#（3）宽，高比例分布
import os
import json
from glob import glob
from PIL import Image
data_path='./data/garbage_classify/raw_data'
all_img_path=os.path.join(data_path,'*.jpg')
#img=Image.open(image_path) #w,h
img_path_list=glob(all_img_path) #扫描符合条件的所有路径
data_list=[]
filename='data_list.json'
if os.path.exists(filename):
    with open(filename) as file_obj:
        data_list=json.load(file_obj)
else:
    for img_path in img_path_list:
        img=Image.open(img_path) #读取过程需要1分钟
        w,h=img.size
        r=float('{:.02f}'.format(w/h))
        img_name=img_path.split('/')[-1]
        img_id=img_name.split('.')[0].split('_')[-1]
        img_label=img2label_dic[img_name]
        data_list.append([int(img_id),w,h,r,int(img_label)])
    with open(filename,'w') as file_obj:
        json.dump(data_list,file_obj)
    
#print(w,h,ratio)
#对单变量进行数据分析，使用直方图来完成
#Python中的seaborn可视化工具库进行展示
import seaborn as sns #导入可视化库
import numpy as np
import matplotlib.pyplot as plt

ratio_list=[data[3] for data in data_list]
ratio_list=list(filter(lambda x:x>0.5 and x<=2,ratio_list))#数据过滤操作保存区间内的值
sns.set()
np.random.seed(0)
ax=sns.distplot(ratio_list)
plt.show()#通过可视化可以知道ratio主要集中在0.5～1.5之间
print('Done!')