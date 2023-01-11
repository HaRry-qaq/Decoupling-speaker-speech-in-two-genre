import torch
import torch.nn as nn
import torch.nn.functional as F


from   torch.autograd import Variable
import torch.nn.init as init

####################################################################### CLASS CLASSIFIER ###############################################################################
class Class_classifier(nn.Module):
    def __init__(self):
      super(Class_classifier, self).__init__()
      # 256-1024
      self.fc1 = nn.Linear(256,256)
      self.bn1 = nn.BatchNorm1d(256,momentum=0.99)
      # self.fc2 = nn.Linear(512,num_classes)
      # self.bn2 = nn.BatchNorm1d(num_classes,momentum=0.99)


    def forward(self, input):
      # tanh
      output1 = self.bn1(self.fc1(input))
      # output2 = self.fc2(input)
      # output3 = self.bn2(output2)
      # output4 = F.tanh(output3)
      return output1