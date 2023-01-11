#!/usr/bin/env python
# coding=utf-8


import torch
import torch.nn.functional as F

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""   
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print('pred:',pred)
    # print('target:',target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print('correct:',correct)
    # print(correct.shape)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

        res.append(correct_k.mul_(100.0 / batch_size))
    return res