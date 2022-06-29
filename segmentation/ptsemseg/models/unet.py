# import torch
# import torch.nn as nn

# # from ptsemseg.models.utils import unetConv2, unetUp


# def conv_block(in_dim, out_dim, act_fn, is_batchnorm):
#     model = nn.Sequential(
#         nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm2d(out_dim) if is_batchnorm else nn.Identity(),
#         act_fn,
#     )
#     return model


# def conv_trans_block(in_dim, out_dim, act_fn, is_batchnorm):
#     model = nn.Sequential(
#         nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3,
#                            stride=2, padding=1, output_padding=1),
#         nn.BatchNorm2d(out_dim) if is_batchnorm else nn.Identity(),
#         act_fn,
#     )
#     return model


# def maxpool():
#     pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#     return pool


# def conv_block_2(in_dim, out_dim, act_fn, is_batchnorm):
#     model = nn.Sequential(
#         conv_block(in_dim, out_dim, act_fn, is_batchnorm),
#         nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm2d(out_dim) if is_batchnorm else nn.Identity()
#     )
#     return model


# # class unet(nn.Module):
# #     def __init__(
# #         self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True
# #     ):
# #         super(unet, self).__init__()
# #         self.is_deconv = is_deconv
# #         self.in_channels = in_channels
# #         self.is_batchnorm = is_batchnorm
# #         self.feature_scale = feature_scale

# #         filters = [64, 128, 256, 512, 1024]
# #         filters = [int(x / self.feature_scale) for x in filters]

# #         # downsampling
# #         self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
# #         self.maxpool1 = nn.MaxPool2d(kernel_size=2)

# #         self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
# #         self.maxpool2 = nn.MaxPool2d(kernel_size=2)

# #         self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
# #         self.maxpool3 = nn.MaxPool2d(kernel_size=2)

# #         self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
# #         self.maxpool4 = nn.MaxPool2d(kernel_size=2)

# #         self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

# #         # upsampling
# #         self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
# #         self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
# #         self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
# #         self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

# #         # final conv (without any concat)
# #         self.final = nn.Conv2d(filters[0], n_classes, 1)

# #     def forward(self, inputs):
# #         import pudb
# #         pudb.set_trace()
# #         conv1 = self.conv1(inputs)
# #         maxpool1 = self.maxpool1(conv1)

# #         conv2 = self.conv2(maxpool1)
# #         maxpool2 = self.maxpool2(conv2)

# #         conv3 = self.conv3(maxpool2)
# #         maxpool3 = self.maxpool3(conv3)

# #         conv4 = self.conv4(maxpool3)
# #         maxpool4 = self.maxpool4(conv4)

# #         center = self.center(maxpool4)
# #         up4 = self.up_concat4(conv4, center) #         up3 = self.up_concat3(conv3, up4)
# #         up2 = self.up_concat2(conv2, up3)
# #         up1 = self.up_concat1(conv1, up2)

# #         final = self.final(up1)

# #         return final


# class unet(nn.Module):
#     def __init__(self, n_classes=21, in_channels=3, size=1.0, use_leakyrelu=True):
#         super(unet, self).__init__()
#         is_batchnorm=True
#         # act_fn =  
#         # act_fn = nn.ReLU(inplace=True)

#         def get_act():
#             return nn.ReLU(inplace=True)


#         self.num_filter = int(64 * size)

#         # prune it before the max pooling
#         self.down_1 = conv_block_2(
#             in_channels, self.num_filter, get_act(), is_batchnorm)
#         self.pool_1 = maxpool()
#         self.down_2 = conv_block_2(
#             self.num_filter*1, self.num_filter*2, get_act(), is_batchnorm)
#         self.pool_2 = maxpool()
#         self.down_3 = conv_block_2(
#             self.num_filter*2, self.num_filter*4, get_act(), is_batchnorm)
#         self.pool_3 = maxpool()
#         self.down_4 = conv_block_2(
#             self.num_filter*4, self.num_filter*8, get_act(), is_batchnorm)
#         self.pool_4 = maxpool()

#         self.bridge = conv_block_2(
#             self.num_filter*8, self.num_filter*16, get_act(), is_batchnorm)

#         self.trans_1 = conv_trans_block(
#             self.num_filter*16, self.num_filter*8, get_act(), is_batchnorm)
#         self.up_1 = conv_block_2(
#             self.num_filter*16, self.num_filter*8, get_act(), is_batchnorm)
#         self.trans_2 = conv_trans_block(
#             self.num_filter*8, self.num_filter*4, get_act(), is_batchnorm)
#         self.up_2 = conv_block_2(
#             self.num_filter*8, self.num_filter*4, get_act(), is_batchnorm)
#         self.trans_3 = conv_trans_block(
#             self.num_filter*4, self.num_filter*2, get_act(), is_batchnorm)
#         self.up_3 = conv_block_2(
#             self.num_filter*4, self.num_filter*2, get_act(), is_batchnorm)
#         self.trans_4 = conv_trans_block(
#             self.num_filter*2, self.num_filter*1, get_act(), is_batchnorm)
#         self.up_4 = conv_block_2(
#             self.num_filter*2, self.num_filter*1, get_act(), is_batchnorm)

#         self.out = nn.Sequential(
#             nn.Conv2d(self.num_filter, n_classes, 3, 1, 1),
#             nn.Tanh(),
#         )

#     def forward(self, input):
#         down_1 = self.down_1(input)
#         pool_1 = self.pool_1(down_1)
#         down_2 = self.down_2(pool_1)
#         pool_2 = self.pool_2(down_2)
#         down_3 = self.down_3(pool_2)
#         pool_3 = self.pool_3(down_3)
#         down_4 = self.down_4(pool_3)
#         pool_4 = self.pool_4(down_4)

#         bridge = self.bridge(pool_4)

#         trans_1 = self.trans_1(bridge)
#         concat_1 = torch.cat([trans_1, down_4], dim=1)
#         up_1 = self.up_1(concat_1)
#         trans_2 = self.trans_2(up_1)
#         concat_2 = torch.cat([trans_2, down_3], dim=1)
#         up_2 = self.up_2(concat_2)
#         trans_3 = self.trans_3(up_2)
#         concat_3 = torch.cat([trans_3, down_2], dim=1)
#         up_3 = self.up_3(concat_3)
#         trans_4 = self.trans_4(up_3)
#         concat_4 = torch.cat([trans_4, down_1], dim=1)
#         up_4 = self.up_4(concat_4)

#         out = self.out(up_4)
#         return out


# if __name__ == "__main__":
#     import torch
#     from qsparse import prune, convert
#     net = unet(is_batchnorm=False)
#     inp = torch.rand(1, 3, 256, 256)
#     out = net(inp)
#     netp = convert(net, prune(sparsity=0.5, start=100,
#                               interval=4, repetition=20, window_size=16),
#                    activation_layers=[nn.Identity],
#                    excluded_activation_layer_indexes=[(nn.Identity, (0,)), ])
#     outp = netp(inp)
#     print(1)


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)





class unet(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, size=1.0, bilinear=False):
        super(unet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        def s(i):
            return int(i * size)


        self.inc = DoubleConv(in_channels, s(64))
        self.down1 = Down(s(64), s(128))
        self.down2 = Down(s(128), s(256))
        self.down3 = Down(s(256), s(512))
        factor = 2 if bilinear else 1
        self.down4 = Down(s(512), s(1024 // factor))
        self.up1 = Up(s(1024), s(512 // factor), bilinear)
        self.up2 = Up(s(512), s(256 // factor), bilinear)
        self.up3 = Up(s(256), s(128 // factor), bilinear)
        self.up4 = Up(s(128), s(64), bilinear)
        self.outc = OutConv(s(64), n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits