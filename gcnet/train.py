#(1)导入相关的库
#(2)输入参数处理
#(3)数据加载预处理
#(4)工具类:日志,优化器 
#(5)模型加载,训练,评估,保存
import os
import torch 
import time
import torchvision
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn 
from torchvision import datasets,models,transforms
from torch.utils.data import DataLoader
from sklearn import metrics  #计算混淆矩阵
from gcnet.transforms import preprocess
from gcnet.classifier import GarbageClassifier
from gcnet.utils import  AverageMeter, save_checkpoint,accuracy
from gcnet.logger import Logger

#data_path='./data/garbage_classify_test'

class_id2name={0:'其他垃圾',1:'厨余垃圾',2:'可回收物', 3:'有害垃圾'}
                
def train(args):

    data_path=args.data_path
    save_path=args.save_path
    #(1) load data
    TRAIN='{}/train'.format(data_path)
    VAL='{}/val'.format(data_path)
    train_data=datasets.ImageFolder(root=TRAIN, transform=preprocess)
    val_data=datasets.ImageFolder(root=VAL, transform=preprocess)
    
    class_list = [class_id2name[i] for i in list(range(len(train_data.class_to_idx.keys())))]

    train_loader=DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    val_loader=DataLoader(val_data, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False)  


    #(2) model inital
    GCNet=GarbageClassifier(args.model_name,args.num_classes,args.ngpu,feature_extract=True)

    #(3) Evaluation:Confusion Matrix:Precision  Recall F1-score
    criterion=nn.CrossEntropyLoss()

    #(4) Optimizer
    optimizer=torch.optim.Adam(GCNet.model.parameters(), args.lr) 

    #(5) load checkpoint 断点重新加载,制定开始迭代的位置
    epochs=args.epochs
    start_epoch=args.start_epoch
    if args.resume:
        # --resume checkpoint/checkpoint.pth.tar
        # load checkpoint
        print('Resuming from checkpoint...')
        assert os.path.isfile(args.resume),'Error: no checkpoint directory found!!'
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        state_dict  = checkpoint['state_dict']
        optim = checkpoint['optimizer']
        #if # create new OrderedDict that does not contain `module.`
        ##由于之前的模型是在多gpu上训练的，因而保存的模型参数，键前边有‘module’，需要去掉，和训练模型一样构建新的字典
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     head = k[:7]
        #     if head == 'module.':
        #         name = k[7:] # remove `module.`
        #     else:
        #         name = k
        #     new_state_dict[name] = v
        GCNet.model.load_state_dict(state_dict)
        optimizer.load_state_dict(optim)

    # #评估: 混淆矩阵；准确率、召回率、F1-score
    # if args.evaluate and args.resume:
    #     print('\nEvaluate only')
    #     test_loss, test_acc, predict_all,labels_all = GCNet.test_model(val_loader,criterion,test=True)
    #     print('Test Loss:%.8f,Test Acc:%.2f' %(test_loss,test_acc))
    #     # 混淆矩阵
    #     report = metrics.classification_report(labels_all,predict_all,target_names=class_list,digits=4)
    #     confusion = metrics.confusion_matrix(labels_all,predict_all)
    #     print('\n report ',report)
    #     print('\n confusion',confusion)
    #     return

    #(6) model train and val
    best_acc=0
    if not args.ngpu:
        logger = Logger(os.path.join(save_path,'log.txt'),title=None)
    else:
        logger = Logger(os.path.join(save_path,'log_ngpu.txt'),title=None)
    ## 设置logger 的头信息
    logger.set_names(['LR', 'epoch', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    for epoch in range(start_epoch,epochs+1):
        print('[{}/{}] Training'.format(epoch,args.epochs))
        # train
        train_loss,train_acc = GCNet.train_model(train_loader,criterion,optimizer)
        # val
        test_loss,test_acc = GCNet.test_model(val_loader,criterion,test=None)
        # 核心参数保存logger
        logger.append([args.lr, int(epoch), train_loss, test_loss, train_acc, test_acc])
        print('train_loss:%f, val_loss:%f, train_acc:%f,  val_acc:%f' % ( train_loss, test_loss, train_acc, test_acc,))
        #保存模型
        is_best = test_acc > best_acc
        best_acc = max(test_acc,best_acc)
        if not args.ngpu:
            name='checkpoint_'+str(epoch)+'.pth.tar'
        else:
            name='ngpu_checkpoint_'+str(epoch)+'.pth.tar'
        save_checkpoint({
            'epoch':epoch,
            'state_dict':GCNet.model.state_dict(),
            'train_acc':train_acc,
            'test_acc':test_acc,
            'best_acc':best_acc,
            'optimizer':optimizer.state_dict()

        }, is_best, checkpoint=save_path, filename=name)
        print('Best acc:')
        print(best_acc)















    
