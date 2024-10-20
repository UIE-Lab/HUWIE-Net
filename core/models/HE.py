import torch
import torch.nn as nn
import cv2
import numpy as np

class HE(nn.Module):

    def __init__(self):
        
        super().__init__()
                
    def forward(self, x):
        
        x2 = torch.squeeze(x, 0)
        x3 = self.color_histogram_equalization(x2)
        output = torch.unsqueeze(x3, 0)
                
        return output
    
    def color_histogram_equalization(self, tensor):
        """
        Function that performs histogram equalization of an RGB image in the RGB color space on a PyTorch tensor.
    
        Args:
        - tensor (torch.Tensor): RGB image as a PyTorch tensor (in C, H, W format).
    
        Returns:
        - torch.Tensor: PyTorch tensor with histogram equalization applied.
        """
        
        # Input check
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor (torch.Tensor).")
        
        if tensor.dim() != 3 or tensor.size(0) != 3:
            raise ValueError("Input tensor must be in RGB format (C, H, W).")
        
        # Convert the tensor to a NumPy array (in H, W, C format)
        image_np = tensor.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)  # From [0, 1] range to [0, 255] range
        
        # Split RGB channels
        channels = cv2.split(image_np)
        
        # Perform histogram equalization on each channel
        equalized_channels = []
        for channel in channels:
            equalized_channel = cv2.equalizeHist(channel)
            equalized_channels.append(equalized_channel)
        
        # Merge equalized channels
        image_equalized_np = cv2.merge(equalized_channels)

        # Convert the histogram-equalized NumPy array back to a PyTorch tensor
        image_equalized_tensor = torch.from_numpy(image_equalized_np).float().div(255).permute(2, 0, 1)
        
        return image_equalized_tensor

