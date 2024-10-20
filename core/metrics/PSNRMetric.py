import torch
import torch.nn as nn

class PSNRMetric(nn.Module):
    
    def __init__(self):
        
        super(PSNRMetric, self).__init__()
            
    def forward(self, toutputs, tlabels):
        
        data_range = 255
        
        mse = torch.mean((toutputs - tlabels)**2, dim=[1, 2, 3]) * data_range**2
        
        psnr = 10 * torch.log10(data_range**2 / mse)
        
        return psnr.item()


