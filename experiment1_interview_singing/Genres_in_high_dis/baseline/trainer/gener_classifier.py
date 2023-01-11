import torch
import torch.nn as nn
import torch.nn.functional as F

from   torch.autograd import Variable
import torch.nn.init as init

############################################################################## GradReverse ###############################################################################
class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # print('梯度反转前的梯度：',grad_output)
        grad_output = grad_output.neg() * ctx.constant
        # print('梯度反转后的梯度：',grad_output)
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

############################################################################### DOMAIN CLASSIFIER ###############################################################################
class Domain_classifier(nn.Module):
    def __init__(self):
      super(Domain_classifier, self).__init__()
      self.fc1 = nn.Linear(256,11)
    #   self.bn1 = nn.BatchNorm1d(11,momentum=0.99)
      # self.fc2 = nn.Linear(256,11)

    def forward(self, input,constant):

    #   input = GradReverse.grad_reverse(input, constant)
      output1 = self.fc1(input)
      # output2 = self.bn2(self.fc2(output1))
      output2 = F.log_softmax(output1, 1)

      return output2





