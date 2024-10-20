import torch
import torch.nn as nn

class UWCNN(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.init_layers()
        # self.init_weight()
        self.get_parameters()

    def init_layers(self):
        
        self.conv2d_dehaze1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.dehaze1_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze2_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze3_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze4 = nn.Conv2d(3+16+16+16, 16, 3, 1, 1)
        self.dehaze4_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze5 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze5_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze6 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze6_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze7 = nn.Conv2d(51+48, 16, 3, 1, 1)
        self.dehaze7_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze8 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze8_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze9 = nn.Conv2d(16, 16, 3, 1, 1)
        self.dehaze9_relu = nn.ReLU(inplace=True)

        self.conv2d_dehaze10 = nn.Conv2d(99+48, 3, 3, 1, 1)

    def init_weight(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.normal_(m.weight)
                nn.init.zeros_(m.bias)

    def get_parameters(self):
                
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.net_parameters = {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x):
        
        image_conv1 = self.dehaze1_relu(self.conv2d_dehaze1(x))
        image_conv2 = self.dehaze2_relu(self.conv2d_dehaze2(image_conv1))
        image_conv3 = self.dehaze3_relu(self.conv2d_dehaze3(image_conv2))

        dehaze_concat1 = torch.cat([image_conv1, image_conv2, image_conv3, x], dim=1)
        image_conv4 = self.dehaze4_relu(self.conv2d_dehaze4(dehaze_concat1))
        image_conv5 = self.dehaze5_relu(self.conv2d_dehaze5(image_conv4))
        image_conv6 = self.dehaze6_relu(self.conv2d_dehaze6(image_conv5))

        dehaze_concat2 = torch.cat([dehaze_concat1, image_conv4, image_conv5, image_conv6], dim=1)
        image_conv7 = self.dehaze7_relu(self.conv2d_dehaze7(dehaze_concat2))
        image_conv8 = self.dehaze8_relu(self.conv2d_dehaze8(image_conv7))
        image_conv9 = self.dehaze9_relu(self.conv2d_dehaze9(image_conv8))

        dehaze_concat3 = torch.cat([dehaze_concat2, image_conv7, image_conv8, image_conv9], dim=1)
        image_conv10 = self.conv2d_dehaze10(dehaze_concat3)
        out = x + image_conv10

        return out
