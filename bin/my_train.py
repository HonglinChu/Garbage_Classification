import os
import argparse
from gcnet.train import train

#在命令行模式运行的时候不要开启一下命令
#import multiprocessing
#multiprocessing.set_start_method('spawn',True)
# for (dirpath,dirname,filename) in os.walk((data_dir)): #查看目录结构
#     if filename:
#         print('*'*100)
#         print('dirpath:',dirpath)
#         print('dirname:',dirname)
#         print('filename:',filename)

#metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.
#dest - 解析后的参数名称，默认情况下，对于可选参数选取最长的名称，中划线转换为下划线.

data_path='./data/garbage_classify_4'
save_path='./models/checkpoint'

if __name__ == '__main__':
    # 创建一个参数的解析对象
    parser = argparse.ArgumentParser(description='Pytorch garbage Training ')

    # 设置参数信息
    ## 模型名称  choices=['resnext101_32x8d', 'resnext101_32x16d'],
    parser.add_argument('--model_name', default='resnext101_32x16d', type=str,help='model_name selected in train')
    
    parser.add_argument('--data_path', default=data_path, type=str,help='the path of training data')

    parser.add_argument('--save_path', default=save_path, type=str,help='the path of model to save')

    # 学习率 metavar='LR'
    parser.add_argument('--lr',  default=0.001, type=float,help='initital learning rate 1e-2,12-4,0.001')

    # batch_size
    parser.add_argument('--batch_size',default=64, type=int,help='batch size')
    # num_works
    parser.add_argument('--num_workers', default=8,type=int,help='num_workers')
    # ngpu
    parser.add_argument('--ngpu', default=1,type=int,help='num_gpu')
    
    # 模型的存储路径metavar='PATH', metavar='PATH',
    parser.add_argument('--resume', default='./models/checkpoint/ngpu_checkpoint_8.pth.tar', type=str, help='path to latest checkpoint')
    
    # evaluate
    parser.add_argument('--evaluate',  default=0, type=int,help='choose if to evaluate')

    # parser.add_argument('--checkpoint', default="checkpoint", type=str, help='path to save checkpoint')
    ## 模型迭代次数 metavar='N',
    parser.add_argument('--epochs', default=10, type=int,  help='number of total epochs to run')

    # 图片分类g metavar='N',
    parser.add_argument('--num_classes', default=4, type=int,  help='number of classes')

    # 从那个epoch 开始训练 metavar='N',
    parser.add_argument('--start_epoch', default=1, type=int,  help='manual epoch number')

    # 进行参数解析
    args = parser.parse_args()
    # 训练
    train(args)


