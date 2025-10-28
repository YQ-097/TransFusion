import torch
import torch.nn as nn
from ChannelAttention import ChannelAttentionBlock
from SpatialAttention import SpatialAttentionBlock
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding=1, bias=True):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.gelu = nn.GELU()#nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv2d(x)
        out =  self.gelu(out)
        return out

class select_ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding=1, bias=True):
        super(select_ConvLayer, self).__init__()
        self.conv2d1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv2d2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias)
        self.gelu = nn.GELU()#nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv2d1(x)
        out = self.conv2d2(out)
        out = self.gelu(out)
        return out


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()

        ##############编码 VGG19######################
        self.layer1 = nn.Sequential(*[features[i] for i in range(0, 4)])
        self.layer2 = nn.Sequential(*[features[i] for i in range(4, 9)])
        self.layer3 = nn.Sequential(*[features[i] for i in range(9, 18)])
        self.layer4 = nn.Sequential(*[features[i] for i in range(18, 27)])
        self.layer5 = nn.Sequential(*[features[i] for i in range(27, 36)])
        #############################################

        ##################解码########################

        self.upsample1_1 = nn.ConvTranspose2d(512, 512, 2, stride=2)

        self.upsample2_2 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.upsample3_3 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.upsample4_4 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv1_1_1 = ConvLayer(512 + 512, 768, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1_1_2 = ConvLayer(768, 512, kernel_size=3, stride=1, padding=1, bias=True)

        # 第二层cnn
        self.conv2_1_1 = ConvLayer(256 + 256, 256 + 256, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.conv2_1_2 = ConvLayer(256 + 256, 256 + 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv2_2_1 = ConvLayer(256 + 256, 256, kernel_size=3, stride=1, padding=1,bias=True)  # 步长调试时候结合图像大小选择
        self.conv2_2_2 = ConvLayer(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        # 第三层cnn
        self.conv3_1_1 = ConvLayer(128 + 128, 128 + 128, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.conv3_1_2 = ConvLayer(128 + 128, 128 + 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_2_1 = ConvLayer(128+ 128, 128, kernel_size=3, stride=1, padding=1,bias=True)  # 步长调试时候结合图像大小选择
        self.conv3_2_2 = ConvLayer(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_3_1 = ConvLayer(128, 128, kernel_size=3, stride=1, padding=1,bias=True)  # 步长调试时候结合图像大小选择
        self.conv3_3_2 = ConvLayer(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        # 第四层cnn
        self.conv4_1_1 = ConvLayer(64 + 64, 64 + 64, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.conv4_1_2 = ConvLayer(64 + 64, 64 + 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_2_1 = ConvLayer(64 + 64, 64 + 64, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.conv4_2_2 = ConvLayer(64 + 64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_3_1 = ConvLayer(64, 64, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.conv4_3_2 = ConvLayer(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_4_1 = ConvLayer(64, 32, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.conv4_4_2 = ConvLayer(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_out1 = ConvLayer(32, 16, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        #self.conv_out2 = ConvLayer(16, 1, kernel_size=3, stride=1, padding=1, bias=True)  # 步长调试时候结合图像大小选择
        self.conv_out2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        ############################################


        ###########Feature selection layer##########
        self.lyr1_CA = ChannelAttentionBlock(channel=64 , reduction=1, kernel_size=3)
        self.lyr2_CA = ChannelAttentionBlock(channel=128, reduction=1, kernel_size=3)
        self.lyr3_CA = ChannelAttentionBlock(channel=256, reduction=2, kernel_size=3)
        self.lyr4_CA = ChannelAttentionBlock(channel=512, reduction=4, kernel_size=3)
        self.lyr5_CA = ChannelAttentionBlock(channel=512, reduction=4, kernel_size=3)


        self.lyr1_SA = SpatialAttentionBlock()
        self.lyr2_SA = SpatialAttentionBlock()
        self.lyr3_SA = SpatialAttentionBlock()
        self.lyr4_SA = SpatialAttentionBlock()
        self.lyr5_SA = SpatialAttentionBlock()

        self.sa_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # size=input_size[2:]

        self.lyr1_sigmoid = nn.Sigmoid()
        self.lyr2_sigmoid = nn.Sigmoid()
        self.lyr3_sigmoid = nn.Sigmoid()
        self.lyr4_sigmoid = nn.Sigmoid()
        ############################################

        # #selection out
        self.f_layer1_conv_out = select_ConvLayer(64 , 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.f_layer2_conv_out = select_ConvLayer(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.f_layer3_conv_out = select_ConvLayer(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.f_layer4_conv_out = select_ConvLayer(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.f_layer5_conv_out = select_ConvLayer(512, 512, kernel_size=3, stride=1, padding=1, bias=True)


        if init_weights:
            self._initialize_weights()

    def forward(self, x, y):
        ir_layer1_feature, ir_layer2_feature, ir_layer3_feature, ir_layer4_feature, ir_layer5_feature = self.encode(x)
        vis_layer1_feature, vis_layer2_feature, vis_layer3_feature, vis_layer4_feature, vis_layer5_feature = self.encode(y)
        layer1_feature, layer2_feature, layer3_feature, layer4_feature, layer5_feature =  self.feature_selection(ir_layer1_feature, ir_layer2_feature, ir_layer3_feature, ir_layer4_feature, ir_layer5_feature, vis_layer1_feature, vis_layer2_feature, vis_layer3_feature, vis_layer4_feature, vis_layer5_feature)

        x = self.decode(layer1_feature, layer2_feature, layer3_feature, layer4_feature, layer5_feature)

        return x

    def feature_selection(self, ir_layer1_feature, ir_layer2_feature, ir_layer3_feature, ir_layer4_feature, ir_layer5_feature, vis_layer1_feature, vis_layer2_feature, vis_layer3_feature, vis_layer4_feature, vis_layer5_feature):


        # layer5##################
        ir_layer5_feature, vis_layer5_feature  = self.lyr5_CA(ir_layer5_feature, vis_layer5_feature)
        layer5_feature = (ir_layer5_feature + vis_layer5_feature) / 2
        layer5_samap = self.lyr5_SA(layer5_feature)
        layer5_feature = layer5_feature + layer5_samap * layer5_feature
        ###########################

        # layer4###################
        ir_layer4_feature, vis_layer4_feature = self.lyr4_CA(ir_layer4_feature, vis_layer4_feature)
        layer4_feature = (ir_layer4_feature + vis_layer4_feature) / 2
        #layer4_samap = self.lyr4_sigmoid(self.lyr4_SA(layer4_feature) + self.sa_upsample(layer5_samap))
        layer4_samap = self.lyr4_SA(layer4_feature)
        layer4_feature = layer4_feature + layer4_samap * layer4_feature
        #############################

        # layer3#####################
        ir_layer3_feature, vis_layer3_feature = self.lyr3_CA(ir_layer3_feature,vis_layer3_feature)
        layer3_feature = (ir_layer3_feature + vis_layer3_feature) / 2
        #layer3_samap = self.lyr3_sigmoid(self.lyr3_SA(layer3_feature) + self.sa_upsample(layer4_samap))
        layer3_samap = self.lyr3_SA(layer3_feature)
        layer3_feature = layer3_feature + layer3_samap * layer3_feature
        ##########################

        # layer2##################
        ir_layer2_feature, vis_layer2_feature = self.lyr2_CA(ir_layer2_feature, vis_layer2_feature)
        layer2_feature = (ir_layer2_feature + vis_layer2_feature) / 2
        #layer2_samap = self.lyr2_sigmoid(self.lyr2_SA(layer2_feature) + self.sa_upsample(layer3_samap))
        layer2_samap = self.lyr2_SA(layer2_feature)
        layer2_feature = layer2_feature + layer2_samap * layer2_feature
        #################################

        #layer1###############
        ir_layer1_feature, vis_layer1_feature = self.lyr1_CA(ir_layer1_feature, vis_layer1_feature)
        layer1_feature = (ir_layer1_feature + vis_layer1_feature) / 2
        #layer1_samap = self.lyr1_sigmoid(self.lyr1_SA(layer1_feature) + self.sa_upsample(layer2_samap))
        layer1_samap = self.lyr1_SA(layer1_feature)
        layer1_feature = layer1_feature + layer1_samap * layer1_feature
        ######################

        layer1_feature = self.f_layer1_conv_out(layer1_feature)

        layer2_feature = self.f_layer2_conv_out(layer2_feature)

        layer3_feature = self.f_layer3_conv_out(layer3_feature)

        layer4_feature = self.f_layer4_conv_out(layer4_feature)

        layer5_feature = self.f_layer5_conv_out(layer5_feature)

        return layer1_feature, layer2_feature, layer3_feature, layer4_feature, layer5_feature

    def encode(self, x):
        x = self.layer1(x)
        layer1_feature = x
        x = self.layer2(x)
        layer2_feature = x
        x = self.layer3(x)
        layer3_feature = x
        x = self.layer4(x)
        layer4_feature = x
        x = self.layer5(x)
        layer5_feature = x
        # x = self.layer6(x)
        return layer1_feature, layer2_feature, layer3_feature, layer4_feature, layer5_feature

    def decode(self, layer1_feature, layer2_feature, layer3_feature, layer4_feature, layer5_feature):

        # # 第一层
        layer5_feature = self.upsample1_1(layer5_feature)
        x1_1 =  torch.cat([layer5_feature, layer4_feature], 1)  # 64*4 -- 64
        x1_1 = self.conv1_1_1(x1_1)  # 32 conv1_1_2
        x1_1 = self.conv1_1_2(x1_1)

        # 第二层
        #layer4_feature = self.upsample2_1(layer4_feature)
        x1_1 = self.upsample2_2(x1_1)

        x2_1 = torch.cat([x1_1, layer3_feature], 1)
        x2_1 = self.conv2_1_1(x2_1)
        x2_1 = self.conv2_1_2(x2_1)

        x2_2 = self.conv2_2_1(x2_1)
        x2_2 = self.conv2_2_2(x2_2)

        # 第三层
        x2_2 = self.upsample3_3(x2_2)

        x3_1 = torch.cat([x2_2, layer2_feature], 1)
        x3_1 = self.conv3_1_1(x3_1)
        x3_1 = self.conv3_1_2(x3_1)

        x3_2 = self.conv3_2_1(x3_1)
        x3_2 = self.conv3_2_2(x3_2)

        x3_3 = self.conv3_3_1(x3_2)
        x3_3 = self.conv3_3_2(x3_3)

        # 第四层
        x3_3 = self.upsample4_4(x3_3)

        x4_1 = torch.cat([x3_3, layer1_feature], 1)
        x4_1 = self.conv4_1_1(x4_1)
        x4_1 = self.conv4_1_2(x4_1)

        x4_2 = x4_1
        x4_2 = self.conv4_2_1(x4_2)
        x4_2 = self.conv4_2_2(x4_2)

        x4_3 = x4_2
        x4_3 = self.conv4_3_1(x4_3)
        x4_3 = self.conv4_3_2(x4_3)

        x4_4 = x4_3
        x4_4 = self.conv4_4_1(x4_4)
        x4_4 = self.conv4_4_2(x4_4)


        x = self.conv_out1(x4_4)  # 64 - 16
        x = self.conv_out2(x)  # 16 - 1
        x = self.relu(x)

        return [x]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
