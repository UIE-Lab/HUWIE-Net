import torch
import torch.nn as nn

class MSEMetric(nn.Module):
    
    def __init__(self):
        
        super(MSEMetric, self).__init__()
            
    def forward(self, toutputs, tlabels):
        
        data_range = 255
        
        mse = torch.mean((toutputs - tlabels)**2, dim=[1, 2, 3]) * data_range**2
               
        return mse.item()


