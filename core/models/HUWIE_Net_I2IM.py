import torch.nn as nn

class HUWIE_Net_I2IM(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        self.init_layers()
        # self.init_weight()
        self.get_parameters()

    def init_layers(self):
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # Image-to-Image Module
        
        self.i2im_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)       
        self.i2im_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.i2im_in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.i2im_conv7 = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # print(self)
        
    def init_weight(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

    def get_parameters(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.net_parameters = {'Total': total_num, 'Trainable': trainable_num}
        
    def forward(self, x):
        
        h = self.relu(self.i2im_in1(self.i2im_conv1(x)))
        h = self.relu(self.i2im_in2(self.i2im_conv2(h)))
        h = self.relu(self.i2im_in3(self.i2im_conv3(h)))
        h = self.relu(self.i2im_in4(self.i2im_conv4(h)))
        h = self.relu(self.i2im_in5(self.i2im_conv5(h)))
        h = self.relu(self.i2im_in6(self.i2im_conv6(h)))
        h = self.i2im_conv7(h)
        h += x
        i2im_out = self.sigmoid(h)
                
        return i2im_out






