import torch
import torch.nn as nn
import torch.nn.functional as F

class HUWIE_Net_PIM(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        self.init_layers()
        # self.init_weight()
        self.get_parameters()

    def init_layers(self):
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # Physics Informed Module
        
        self.pim_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)       
        self.pim_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pim_in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.pim_conv7 = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

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
                
        h2 = self.relu(self.pim_in1(self.pim_conv1(x)))
        h2 = self.relu(self.pim_in2(self.pim_conv2(h2)))
        h2 = self.relu(self.pim_in3(self.pim_conv3(h2)))
        h2 = self.relu(self.pim_in4(self.pim_conv4(h2)))
        h2 = self.relu(self.pim_in5(self.pim_conv5(h2)))
        h2 = self.relu(self.pim_in6(self.pim_conv6(h2)))
        t = self.pim_conv7(h2)
        t += x
        t = self.sigmoid(t)
        
        dark = self.dark_channel(x)
        A = self.atmospheric_light(x, dark)
        
        eps = 1e-05
        fb_1 = torch.div((x - A), (t + eps)) + A
        pim_out = self.sigmoid(fb_1)
                        
        return pim_out
        
    def dark_channel(self, img):
            
        patch_size = 15
            
        # Step 1: Disable the red channel, use only blue and green channels
        no_red_img = img[:, 1:, :, :]  # Tensor of shape (batch_size, 2, H, W) (green and blue channels)
        
        # Step 2: Find the minimum values in each channel of the image
        min_img, _ = torch.min(no_red_img, dim=1, keepdim=True)
            
        # Step 3: Perform min pooling over the minimum values
        dark = -F.max_pool2d(-min_img, kernel_size=patch_size, stride=1, padding=patch_size//2)
            
        return dark
            
    def atmospheric_light(self, img, dark_channel):
            
        # Flatten the image and dark channel map
        flat_img = img.view(img.size(0), img.size(1), -1)  # (batch_size, 3, H*W)
        flat_dark = dark_channel.view(dark_channel.size(0), dark_channel.size(1), -1)  # (batch_size, 1, H*W)
        
        # Select the brightest 0.1% of pixels in the dark channel
        num_pixels = flat_dark.size(dim=2)
        num_top_pixels = int(0.001 * num_pixels)
        _, indices = torch.topk(flat_dark, k=num_top_pixels, dim=2, largest=True, sorted=False)
        
        # Retrieve the RGB values of the selected pixels and find the maximum value
        A = torch.gather(flat_img, 2, indices.expand(-1, img.size(1), -1)).max(dim=2)[0]
            
        A = A.unsqueeze(2).unsqueeze(3)
            
        return A







