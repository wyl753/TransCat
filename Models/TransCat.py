from .backbone.resnet import ResNet
from .fpn.trans_enc import MSDeformAttnPixelDecoder
from addict import Dict
from .seg_head.TransposedTCB import Decoder
from .utils import Bottleneck

import torch.nn as nn


class TransCatHead(nn.Module):
    def __init__(self,  input_shape):
        super().__init__()
        self.fpn = self.FPN_init(input_shape)
        # self.PD = PartialDecoder()
        self.segmentation_head = self.SegmentationHead()

    def FPN_init(self, input_shape):
        common_stride = 4
        transformer_dropout = 0.1
        transformer_nheads = 8
        transformer_dim_feedforward = 1024
        transformer_enc_layers = 6
        conv_dim = 256
        transformer_in_features = ['res2', "res3", "res4", "res5"],
        fpn = MSDeformAttnPixelDecoder(input_shape,
                                       transformer_dropout,
                                       transformer_nheads,
                                       transformer_dim_feedforward,
                                       transformer_enc_layers,
                                       conv_dim,
                                       transformer_in_features,
                                       common_stride)
        return fpn
    def SegmentationHead(self):
        predictor = Decoder()
        return predictor

    def forward(self, features):
        output, feature, feature1 = self.fpn.forward_features(features)
        # Out_res, Out_trans = self.PD(feature), self.PD(feature1)
        predictions = self.segmentation_head(feature, feature1)
        return predictions

class TransCatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = self.build_backbone()
        self.sem_seg_head = TransCatHead(self.backbone_feature_shape)

    def build_backbone(self):
        # model_type = 'resnet50'

        channels = [256, 512, 1024, 2048]
        backbone = ResNet(Bottleneck, [3, 4, 6, 3])
        backbone.init_weights()
        self.backbone_feature_shape = dict()
        for i, channel in enumerate(channels):
            self.backbone_feature_shape[f'res{i + 2}'] = Dict({'channel': channel, 'stride': 2 ** (i + 2)})
        return backbone

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        features = self.backbone(x)
        outputs = self.sem_seg_head(features)
        return outputs


