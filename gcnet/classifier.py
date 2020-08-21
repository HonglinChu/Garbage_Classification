import time
import torch
import torch.nn as nn
import numpy as np
from progress.bar import Bar
from gcnet import resnet
from gcnet.utils import  AverageMeter, accuracy

class GarbageClassifier:

    def __init__(self,model_name,num_classes,ngpu,feature_extract=True):
        
        self.name='GarbageClassifier'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_name=='resnext101_32x16d':
            model=resnet.resnext101_32x16d_wsl()#加载模型,默认1000 类别    
        elif model_name=='resnext101_32x8d':
            model=resnet.resnext101_32x8d()#加载模型,默认1000 类别
        else:
            model=resnet.resnet50()

        if feature_extract:
            for param in model.parameters():
                # 不需要更新梯度，冻结某些层的梯度
                param.requires_grad = False

        input_feat=model.fc.in_features #获取全连接层的输入特征
       
        model.fc=nn.Sequential(   
            nn.Dropout(0.2) ,#防止过拟合 , 重新定义全连接层
            nn.Linear(in_features=input_feat,out_features=num_classes) 
        )
        
        if ngpu: 
            model = nn.DataParallel(model,device_ids=[0,1])
        model.to(self.device)
        print('Total params:%.2fM'%(sum(p.numel() for p in model.parameters())/1000000.0))#打印模型参数数量
        
        self.model=model

    def train_model(self,train_loader, criterion, optimizer):
    
        # 定义保存更新变量
        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        self.model.train()

        # 训练每批数据，然后进行模型的训练
        ## 定义bar 变量
        bar = Bar('Processing',max = len(train_loader))
        for batch_index, (inputs, targets) in enumerate(train_loader):
            data_time.update(time.time() - end)
            # move tensors to GPU if cuda is_available
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # 在进行反向传播之前，我们使用zero_grad方法清空梯度
            optimizer.zero_grad()
            # 模型的预测
            outputs = self.model(inputs)
            # 计算loss
            loss = criterion(outputs, targets)
            # backward pass:
            loss.backward()
            # perform as single optimization step (parameter update)
            optimizer.step()

            # 计算acc和变量更新
            prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 1))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            ## 把主要的参数打包放进bar中
            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                batch=batch_index + 1,
                size=len(train_loader),
                data=data_time.val,
                bt=batch_time.val,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg
            )
            bar.next()
        bar.finish()
        return (losses.avg, top1.avg)


    def test_model(self,val_loader, criterion,test = None):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        predict_all = np.array([],dtype=int)
        labels_all = np.array([],dtype=int)

        self.model.eval()
        end = time.time()

        # 训练每批数据，然后进行模型的训练
        ## 定义bar 变量
        bar = Bar('Processing', max=len(val_loader))
        for batch_index, (inputs, targets) in enumerate(val_loader):
            data_time.update(time.time() - end)
            # move tensors to GPU if cuda is_available
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # 模型的预测
            outputs = self.model(inputs)
            # 计算loss
            loss = criterion(outputs, targets)

            # 计算acc和变量更新
            prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 1))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            # 评估混淆矩阵的数据
            targets = targets.data.cpu().numpy() # 真实数据的y数值
            predic = torch.max(outputs.data,1)[1].cpu().numpy() # 预测数据y数值
            labels_all = np.append(labels_all,targets) # 数据赋值
            predict_all = np.append(predict_all,predic)

            ## 把主要的参数打包放进bar中
            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                batch=batch_index + 1,
                size=len(val_loader),
                data=data_time.val,
                bt=batch_time.val,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg
            )
            bar.next()
        bar.finish()

        if test:
            return (losses.avg, top1.avg,predict_all,labels_all)
        else:
            return (losses.avg, top1.avg)
    
       