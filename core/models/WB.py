import torch
import torch.nn as nn
import cv2
import numpy as np

class WB(nn.Module):

    def __init__(self):
        
        super().__init__()
                
    def forward(self, x):
        
        x2 = torch.squeeze(x, 0)
        x3 = self.white_balance(x2)
        output = torch.unsqueeze(x3, 0)
                
        return output

    def white_balance(self, tensor):
        
        # Input check
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor (torch.Tensor).")
        
        if tensor.dim() != 3 or tensor.size(0) != 3:
            raise ValueError("Input tensor must be in RGB format (C, H, W).")
        
        # Convert the tensor to a NumPy array (in H, W, C format)
        image_np = tensor.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)  # From [0, 1] range to [0, 255] range
        
        # Use the SimpleWB algorithm for white balance
        wb = cv2.xphoto.createSimpleWB()
        balanced_image = wb.balanceWhite(image_np)
        
        # Convert the NumPy array back to a PyTorch tensor
        image_balanced_tensor = torch.from_numpy(balanced_image).float().div(255).permute(2, 0, 1)
        
        return image_balanced_tensor
