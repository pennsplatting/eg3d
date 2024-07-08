
import torch
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        # x.shape: b c h w
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# from torch.autograd import Variable
# def main():
#     # x = Variable(torch.FloatTensor([[[[1, 2], [3, 1]], [[1, 2], [3, 1]]]]).view(1, 2, 2, 2), requires_grad=True)
#     # x = Variable(torch.FloatTensor([[[[1, 3], [4, 3]], [[1, 3], [4, 3]]]]).view(1, 2, 2, 2), requires_grad=True)
#     # x = Variable(torch.FloatTensor([[[[1, 1], [1, 1]], [[2, 2], [3, 3]], [[1, 1], [1, 1]]]]).view(1, 2, 3, 2), requires_grad=True)
#     x = Variable(torch.FloatTensor([[[1, 2, 3], [4, 5, 1]], [[1, 2, 3], [4, 5, 1]], [[1, 2, 3], [4, 5, 1]]]).view(1, 3, 2, 3), requires_grad=True)
#     addition = TVLoss()
#     z = addition(x)
#     print(x)
#     print(z.data)
#     z.backward()
#     print(x.grad)

# if __name__ == "__main__":
#     main()
