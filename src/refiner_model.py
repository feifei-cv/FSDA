import torch.nn as nn
from torch.nn import functional as F
import copy
from torch.autograd import Function

device = 'cuda'


class GradRevLayer(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # print('***********************')
        grad_input = grad_output.neg() * ctx.beta
        # print(grad_input)
        return grad_input, None


class RefineAction(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(RefineAction, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
             [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)


    def forward(self, x):
        x = GradRevLayer.apply(x, 1.0)
        out_feat = self.conv_1x1(x)
        for layer in self.layers:
            out_feat = layer(out_feat)
        out = self.conv_out(out_feat)
        return out


# class RefineAction(nn.Module):
#     def __init__(self, num_layers, num_f_maps, dim, num_classes):
#         super(RefineAction, self).__init__()
#         self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
#         self.block1 = DilatedResidualLayer(1, num_f_maps, num_f_maps)
#         self.block6 = DilatedResidualLayer(12, num_f_maps, num_f_maps)
#         self.block12 = DilatedResidualLayer(24, num_f_maps, num_f_maps)
#         self.block18 = DilatedResidualLayer(36, num_f_maps, num_f_maps)
#         self.conv_1x1_output = nn.Conv1d(num_f_maps * 4, num_classes, 1, 1)
#         # self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
#
#
#     def forward(self, x):
#         x = GradRevLayer.apply(x, 1.0)
#         feat = self.conv_1x1(x)
#         block1 = self.block1(feat)
#         block6 = self.block6(feat)
#         block12 = self.block12(feat)
#         block18 = self.block18(feat)
#         out = self.conv_1x1_output(torch.cat([block1, block6, block12, block18], dim=1))
#         # out = self.conv_out(out)
#
#         return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(p=0.55, inplace=True)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)



