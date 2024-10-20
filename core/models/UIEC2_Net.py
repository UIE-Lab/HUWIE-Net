import torch
import torch.nn as nn

class UIEC2_Net(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        self.init_layers()
        # self.init_weight()
        self.get_parameters()

    def init_layers(self):
        
        self.rgb2hsv = rgb2hsv()
        self.hsv2rgb = hsv2rgb()
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # RGB Pixel-Level Block
        
        self.rgb_con1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rgb_in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)       
        self.rgb_con2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rgb_in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.rgb_con3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rgb_in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.rgb_con4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rgb_in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.rgb_con5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rgb_in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.rgb_con6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rgb_in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.rgb_con7 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # HSV Global-Adjust Block

        self.M = 11
        self.e_conv1 = nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.e_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.e_conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.e_conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.e_conv7 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.e_convfc = nn.Linear(in_features=64, out_features=44, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        
        # Attention Map Block
               
        self.con1 = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in1 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False) 
        self.con2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in2 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in3 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in4 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in5 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in6 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.con7 = nn.Conv2d(64, 6, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

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
        
        h = self.lrelu(self.rgb_in1(self.rgb_con1(x)))
        h = self.lrelu(self.rgb_in2(self.rgb_con2(h)))
        h = self.lrelu(self.rgb_in3(self.rgb_con3(h)))
        h = self.lrelu(self.rgb_in4(self.rgb_con4(h)))
        h = self.relu(self.rgb_in5(self.rgb_con5(h)))
        h = self.relu(self.rgb_in6(self.rgb_con6(h)))
        h = self.rgb_con7(h)
        h = self.sigmoid(h)
        
        rgb_pixel_level_block_out = h[:,0:3,:,:]
        
        hsv_global_adjust_block_in_to_hsv = self.rgb2hsv(rgb_pixel_level_block_out)
        h2 = torch.cat([hsv_global_adjust_block_in_to_hsv, hsv_global_adjust_block_in_to_hsv], dim=1)
        
        batch_size = h2.size()[0]
        
        h2 = self.lrelu(self.e_conv1(h2))
        h2 = self.maxpool(h2)
        h2 = self.lrelu(self.e_conv2(h2))
        h2 = self.maxpool(h2)
        h2 = self.lrelu(self.e_conv3(h2))
        h2 = self.maxpool(h2)
        h2 = self.lrelu(self.e_conv4(h2))
        h2 = self.lrelu(self.e_conv7(h2))
        h2 = self.avgpool(h2)
        h2 = h2.view(batch_size, -1)
        h2 = self.e_convfc(h2)
        
        H, S, V, H2S = torch.split(h2, self.M, dim=1)
        H_in = hsv_global_adjust_block_in_to_hsv[:, 0:1, :, :]
        S_in = hsv_global_adjust_block_in_to_hsv[:, 1:2, :, :]
        V_in = hsv_global_adjust_block_in_to_hsv[:, 2:3, :, :]
        
        H_out = piecewise_linear_curve(H_in, H, self.M)
        S_out1 = piecewise_linear_curve(S_in, S, self.M)
        S_out2 = piecewise_linear_curve(H_in, H2S, self.M)
        S_out = (S_out1 + S_out2) / 2
        V_out = piecewise_linear_curve(V_in, V, self.M)
        
        S_out = sgn(S_out)
        V_out = sgn(V_out)
        
        h2 = torch.cat([H_out, S_out, V_out], dim=1)
        
        hsv_global_adjust_block_out = self.hsv2rgb(h2)
        
        attention_map_block_in = torch.cat([x, rgb_pixel_level_block_out, hsv_global_adjust_block_out], dim=1)
        
        h3 = self.lrelu(self.in1(self.con1(attention_map_block_in)))
        h3 = self.lrelu(self.in2(self.con2(h3)))
        h3 = self.lrelu(self.in3(self.con3(h3)))
        h3 = self.lrelu(self.in4(self.con4(h3)))
        h3 = self.lrelu(self.in5(self.con5(h3)))
        h3 = self.lrelu(self.in6(self.con6(h3)))
        h3 = self.con7(h3)
        attention_map_block_out = self.sigmoid(h3)
        
        output = 0.5 * attention_map_block_out[:, 0:3, :, :] * rgb_pixel_level_block_out + \
            0.5 * attention_map_block_out[:, 3:6, :, :] * hsv_global_adjust_block_out
            
        return output

def piecewise_linear_curve(x, param, M):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    b, c, w, h = x.shape
    k0 = param[:, 0].view(b, c, 1, 1).expand(b, c, w, h)
    
    val = k0
    
    for i in range(M-1):
        
        val = val + (param[:, i + 1] - param[:, i]).view(b, c, 1, 1).expand(b, c, w, h) * sgn(M * x - i * torch.ones(x.shape).to(device))
            
    return val

def sgn(x):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    zero = torch.zeros(x.shape).to(device)
    one = torch.ones(x.shape).to(device)
    
    s1 = torch.where(x < 0, zero, x)
    s2 = torch.where(s1 > 1, one, s1)
    
    return s2

class rgb2hsv(nn.Module):
    
    def __init__(self):
        
        super(rgb2hsv, self).__init__()

    def forward(self, rgb):
        
        batch, c, w, h = rgb.size()
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        
        V, max_index = torch.max(rgb, dim=1)
        
        min_rgb = torch.min(rgb, dim=1)[0]
        v_plus_min = V - min_rgb
        S = v_plus_min / (V + 0.0001)
        
        H = torch.zeros_like(rgb[:, 0, :, :])
                
        mark = max_index == 0
        H[mark] = 60 * (g[mark] - b[mark]) / (v_plus_min[mark] + 0.0001)
        mark = max_index == 1
        H[mark] = 120 + 60 * (b[mark] - r[mark]) / (v_plus_min[mark] + 0.0001)
        mark = max_index == 2
        H[mark] = 240 + 60 * (r[mark] - g[mark]) / (v_plus_min[mark] + 0.0001)

        mark = H < 0
        H[mark] += 360
        H = H % 360
        H = H / 360
        
        HSV_img = torch.cat([H.view(batch, 1, w, h), S.view(batch, 1, w, h), V.view(batch, 1, w, h)], 1)
        
        return HSV_img
    
class rgb2hsv_2(nn.Module):
    
    def __init__(self):
        
        super(rgb2hsv_2, self).__init__()

    def forward(self, rgb):
        
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
        
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.
        hsv_h /= 6.
        
        hsv_s = torch.where(cmax == 0, (delta / cmax) * 0, delta / cmax)
        hsv_v = cmax
        
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)
    
class hsv2rgb(nn.Module):
    
    def __init__(self):
        
        super(hsv2rgb, self).__init__()

    def forward(self, hsv):
        
        batch, c, w, height = hsv.size()
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        
        htemp = (h * 360) % 360
        h = htemp / 360
        vs = torch.div(torch.mul(v, s), 60)
        
        R1_delta = torch.clamp(torch.add(torch.mul(h, 360), -60), min=0, max=60)
        R2_delta = torch.clamp(torch.add(torch.mul(h, 360), -240), min=0, max=60)
        
        G1_delta = torch.clamp(torch.add(torch.mul(h, 360), 0), min=0, max=60)
        G2_delta = torch.clamp(torch.add(torch.mul(h, 360), -180), min=0, max=60)
        
        B1_delta = torch.clamp(torch.add(torch.mul(h, 360), -120), min=0, max=60)
        B2_delta = torch.clamp(torch.add(torch.mul(h, 360), -300), min=0, max=60)
        
        one_minus_s = torch.mul(torch.add(s, -1), -1)
        R_1 = torch.add(v, torch.mul(vs, R1_delta), alpha=-1)
        R_2 = torch.mul(vs, R2_delta)
        R = torch.add(R_1, R_2)

        G_1 = torch.add(torch.mul(v, one_minus_s), torch.mul(vs, G1_delta))
        G_2 = torch.mul(vs, G2_delta)
        G = torch.add(G_1, G_2, alpha=-1)
        
        B_1 = torch.add(torch.mul(v, one_minus_s), torch.mul(vs, B1_delta))
        B_2 = torch.mul(vs, B2_delta)
        B = torch.add(B_1, B_2, alpha=-1)

        del h, s, v, vs, R1_delta, R2_delta, G1_delta, G2_delta, B1_delta, B2_delta, one_minus_s, R_1, R_2, G_1, G_2, B_1, B_2

        R = torch.reshape(R, (batch, 1, w, height))
        G = torch.reshape(G, (batch, 1, w, height))
        B = torch.reshape(B, (batch, 1, w, height))
        rgb_img = torch.cat([R, G, B], 1)

        return rgb_img

class hsv2rgb_2(nn.Module):
    
    def __init__(self):
        
        super(hsv2rgb_2, self).__init__()

    def forward(self, hsv):
        
        hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        
        _c = hsv_l * hsv_s
        _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
        _m = hsv_l - _c
        _o = torch.zeros_like(_c)
        
        idx = (hsv_h * 6.).type(torch.uint8)
        idx = (idx % 6).expand(-1, 3, -1, -1)
        
        rgb = torch.empty_like(hsv)
        rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
        rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
        rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
        rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
        rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
        rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
        
        rgb += _m
        
        return rgb
    
    
    
    
    
    
    
