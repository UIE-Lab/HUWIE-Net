import torch.nn as nn

class L1Loss(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.loss = nn.L1Loss(reduction = 'mean')
        
    def forward(self, input, target):
          
        return self.loss(input, target)


