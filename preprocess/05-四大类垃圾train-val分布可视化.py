#训练数据和验证数据可视化分布,主要是分析每一种标签对应的数量
import os
import  codecs
from collections import Counter

from pyecharts import options as opts
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

base_path='./data/garbage_classify'
type=['train','val']
res={}
temp={}
for i in type:
    sub_path=os.path.join(base_path,i+'.txt')
    #读取 统计不同label的数量,构建数据
    label_list=[]
    for line in codecs.open(sub_path,'r'): #读取txt文件
        line=line.strip()
        index=line.split('\t')[1]#0-39
        label_list.append(int(index))
    temp=dict(Counter(label_list))#统计每一个数字出现了多少次
    res[i]=dict(sorted(temp.items()))#按照关键字排序
   
#创建Bar
bar=Bar(init_opts=opts.InitOpts(width='1000px',height='500px'))
#                 0-5             6-13          14-36        37-39
label_4_name={0:'其他垃圾', 1:'厨余垃圾', 2:'可回垃圾', 3:'有害垃圾'}
x=list(label_4_name.values())

label_4_count={0:0, 1:0, 2:0, 3:0}

temp=res['train']
for j in range(len(temp)):
    if j<=5:
        label_4_count[0]+=temp[j]
    elif j>5 and j<=13:
        label_4_count[1]+=temp[j]
    elif j>13 and j<=36:
        label_4_count[2]+=temp[j]
    else:
        label_4_count[3]+=temp[j]    
y_train=list(label_4_count.values())

temp=res['val']
label_4_count={0:0, 1:0, 2:0, 3:0}
for j in range(len(temp)):
    if j<=5:
        label_4_count[0]+=temp[j]
    elif j>5 and j<=13:
        label_4_count[1]+=temp[j]
    elif j>13 and j<=36:
        label_4_count[2]+=temp[j]
    else:
        label_4_count[3]+=temp[j]  
y_val=list(label_4_count.values())
#设置title
bar.set_global_opts(
    title_opts=opts.TitleOpts(title='garbage-classify: Train/Val'),
    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)) #
    )

bar.add_xaxis(xaxis_data=x)
# yaxis_data ---> y_axis
bar.add_yaxis(series_name='train',y_axis=y_train)
bar.add_yaxis(series_name='val',y_axis=y_val)

#保存
bar.render('./preprocess/05.html')