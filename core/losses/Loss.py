import torch.nn as nn

from .L1Loss import L1Loss
from .SSIMLoss import SSIMLoss

class Loss(nn.Module):
    
    def __init__(self):
        
        super(Loss, self).__init__()
        
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        
        self.l1_loss_weight = 1.0
        self.ssim_loss_weight = 1.0
        
        self.loss_fn_num = 3      
        self.loss_name = ['l1_loss', 'ssim_loss', 'total_loss']
        
    def forward(self, enhanced, original):

        # L1 loss
        l1_loss = self.l1_loss(enhanced, original)
        
        # SSIM loss
        ssim_loss = self.ssim_loss(enhanced, original)
        
        # Combine losses with weights
        total_loss = (self.l1_loss_weight * l1_loss + self.ssim_loss_weight * ssim_loss)
        
        return [l1_loss, ssim_loss, total_loss], [self.l1_loss_weight, self.ssim_loss_weight, 1.0]
    

