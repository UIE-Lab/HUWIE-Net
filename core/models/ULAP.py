import torch
import torch.nn as nn
import cv2
import numpy as np

from ._ULAP import BackgroundLight
from ._ULAP import DepthMapEstimation
from ._ULAP import DepthMin
from ._ULAP import GlobalStretching
from ._ULAP import GuidedFilter
from ._ULAP import RefinedTransmission
from ._ULAP import SceneRadiance
from ._ULAP import Transmission

class ULAP(nn.Module):

    def __init__(self):
        
        super().__init__()
                
    def forward(self, x):
        
        x2 = torch.squeeze(x, 0)
        x3 = self.ulap(x2)
        output = torch.unsqueeze(x3, 0)
                
        return output

    def ulap(self, tensor):
        
        # Input check
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor (torch.Tensor).")
        
        if tensor.dim() != 3 or tensor.size(0) != 3:
            raise ValueError("Input tensor must be in RGB format (C, H, W).")
        
        # Convert the tensor to a NumPy array (in H, W, C format)
        image_np = tensor.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)  # From [0, 1] range to [0, 255] range
        
        bgr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        depth_map = DepthMapEstimation(bgr_image)
        depth_map = GlobalStretching(depth_map)
        background_light = BackgroundLight(bgr_image, depth_map) * 255
        d_0 = DepthMin(bgr_image, background_light)
        d_f = 8 * (depth_map + d_0)
        transmissionB, transmissionG, transmissionR = Transmission(d_f)
        refined_transmission = RefinedTransmission(transmissionB, transmissionG, transmissionR, bgr_image)
        scene_radiance = SceneRadiance(bgr_image, refined_transmission, background_light)  
        
        rgb_image = cv2.cvtColor(scene_radiance, cv2.COLOR_BGR2RGB)
        
        scene_radiance_tensor = torch.from_numpy(rgb_image).float().div(255).permute(2, 0, 1)
        
        return scene_radiance_tensor
