
#分为四个大类
import os
import cv2
#（2）二级分类不同类别的分布
from glob import glob

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
#                 0-5             6-13          14-36        37-39
label_4_name={0:'其他垃圾', 1:'厨余垃圾', 2:'可回垃圾', 3:'有害垃圾'}
label_4_count={0:0,1:0,2:0,3:0}
#import matplotlib.pyplot as  plt
#首先对label_count_dic 按照key进行排序
label_count_dic=dict(sorted(label_count_dic.items()))#默认按照key进行排序 0-39
for i in range(len(label_count_dic)):
    if i<=5:
        label_4_count[0]+=label_count_dic[i]
    elif i>5 and i<=13:
        label_4_count[1]+=label_count_dic[i]
    elif i>13 and i<=36:
        label_4_count[2]+=label_count_dic[i]
    else:
        label_4_count[3]+=label_count_dic[i]

#x=label_count_dic.keys() #0-39  
# x=['{}-{}'.format(label_index,label_dict[str(label_index)]) for label_index in label_count_dic]
# y=label_count_dic.values()#count
x=label_4_name.values()
y=label_4_count.values()
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
bar.render('./preprocess/04.html')
#print('Done!')
