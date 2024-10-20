import torch.nn as nn

class MSELoss(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.loss = nn.MSELoss(reduction = 'mean')
        
    def forward(self, input, target):
                
        return self.loss(input, target)


