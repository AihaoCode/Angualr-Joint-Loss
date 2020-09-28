import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

class SimSoftmaxLoss(nn.Module):
    def __init__(self, num_classes=100,lr=1):
        super(SimSoftmaxLoss, self).__init__()
        self.lr=lr
        self.class_num=num_classes
    def forward(self, input, target):
        '''
        inputs:
            input:[batch,feature_size]
            target: LongTensor,[batch]
        return:
            loss:scalar
        '''
        #input=input+0.1

        batch_size=input.shape[0]
        input=input/(torch.norm(input,p=2,dim=1,keepdim=True))
        
        similarity=torch.mm(input,input.t())

        labels = target.view(batch_size, -1).cpu()
        labels = torch.zeros(batch_size, self.class_num).scatter_(1, labels, 1).cuda()

        similarity=similarity.cuda()
        similarity_1=-torch.log(similarity)
        similarity_2=-torch.log(1.0001-similarity+torch.eye(batch_size).cuda())



        labels_mat=torch.mm(labels,labels.t())
        labels_mat=labels_mat.cuda()
        labels1=labels_mat
        num1 = labels1.nonzero().size()[0]
        labels2=1-labels_mat
        num2=labels2.nonzero().size()[0]
        loss=2*torch.mul(similarity_1,labels1).sum()/num1+0.6*torch.mul(similarity_2,labels2).sum()/num2

        return loss
