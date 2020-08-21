'''
Some helper functions for PyTorch, including:
'''
import torch
import os

__all__ = ['AverageMeter', 'get_optimizer', 'save_checkpoint','accuracy']

def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(),
                               args.lr)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(),
                                   args.lr)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(),
                                args.lr)
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    # 保存断点信息
    filepath = os.path.join(checkpoint, filename)
    print('checkpoint filepath = ',filepath)
    torch.save(state, filepath)
    # 模型保存
    if is_best:
        # model_name ='best_'+str(state['epoch']) + '_' + str(
        #     int(round(state['train_acc'] * 100, 0))) + '_' + str(
        #     int(round(state['test_acc'] * 100, 0))) + '.pth'
        model_name='best_'+filename
        #print('Validation loss decreased  Saving model ..,model_name = ', model_name)
        model_path = os.path.join(checkpoint, model_name)
        #print('model_path = ',model_path)
        #torch.save(state['state_dict'], model_path)
        torch.save(state, model_path)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"
          Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res