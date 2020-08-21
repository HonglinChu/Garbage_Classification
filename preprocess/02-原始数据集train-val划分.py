

#（4）切分验证集-0.2和训练集-0.8
#img_path_list包括了图片路径和对应标签
import random
import json
import os
import cv2
from os import walk
#二级分类不同类别的分布
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

random.shuffle(img_list) #image_label image_name_path
train_size=int(len(img_list)*0.8)
train_list=img_list[:train_size]
val_list=img_list[train_size:]

#生成 train.txt文件和val.txt文件
import shutil # copy
path='./data/garbage_classify/'
type={'train':train_list,'val':val_list}
for key in type:
    with open(os.path.join(path,key+'.txt'),'w') as f:
        for img_dict in type[key]:
            img_name_path=img_dict['img_name_path']
            img_label=img_dict['img_label']
            f.write('{}\t{}\n'.format(img_name_path,img_label))

            #生成train or val
            sub_path=os.path.join(path,key,str(img_label))
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
            #图片数据copy
            shutil.copy(img_name_path,sub_path)


























