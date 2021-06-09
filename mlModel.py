import torch
import torch.nn as nn
import torch.nn.functional as F

class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class MLModel():

    @staticmethod
    def parse(string):
        #parse string into MLModel

        return MLModel()

    def __repr__(self):
        #turn MLModel into string

        return "x y z"
