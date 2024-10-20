import torch
import torch.nn as nn
import numpy as np
import cv2

from ._UDCP import AtmosphericLight
from ._UDCP import Transmission
from ._UDCP import RefinedTransmission
from ._UDCP import SceneRadiance
from ._UDCP import DarkChannel

class UDCP(nn.Module):

    def __init__(self):
        
        super().__init__()
                
    def forward(self, x):
        
        x2 = torch.squeeze(x, 0)
        x3 = self.udcp(x2)
        output = torch.unsqueeze(x3, 0)
                
        return output

    def udcp(self, tensor):
        
        # Input check
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor (torch.Tensor).")
        
        if tensor.dim() != 3 or tensor.size(0) != 3:
            raise ValueError("Input tensor must be in RGB format (C, H, W).")
        
        # Convert the tensor to a NumPy array (in H, W, C format)
        image_np = tensor.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)  # From [0, 1] range to [0, 255] range
        
        bgr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        patch_size = 15
        dark_channel = DarkChannel(bgr_image, patch_size)
        atmospheric_light = AtmosphericLight(dark_channel, bgr_image)
        _transmission = Transmission(bgr_image, atmospheric_light, patch_size)
        refined_transmission = RefinedTransmission(_transmission, bgr_image)
        scene_radiance = SceneRadiance(bgr_image, refined_transmission, atmospheric_light)
        
        rgb_image = cv2.cvtColor(scene_radiance, cv2.COLOR_BGR2RGB)
                
        scene_radiance_tensor = torch.from_numpy(rgb_image).float().div(255).permute(2, 0, 1)
        
        return scene_radiance_tensor
