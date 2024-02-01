#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from training.gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix
import math
from ipdb import set_trace as st

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam(nn.Module):
    def __init__(self, width, height, znear, zfar, device=None):
        super().__init__()
        self.image_width = width
        self.image_height = height    
        self.FoVy = 0
        self.FoVx = 0
        self.znear = znear
        self.zfar = zfar
        self.projection_matrix = None
        self.world_view_transform = None
        self.full_proj_transform = None
        self.camera_center = None
        # if not torch.all(self.world_view_transform==0): 
        #     view_inv = torch.inverse(self.world_view_transform) 
        # else:
        #     view_inv = self.world_view_transform
        # self.camera_center = view_inv[3][:3]
        
        
        
    def update_transforms(self, intrinsics, world_view_transform, device=None):
        # intrinsics = c[:, 16:25].view(-1, 3, 3)
        focal = intrinsics[0,0,1] * self.image_width # check
        # focal = intrinsics[0,0,0] * self.image_width # check
        fov = 2*torch.arctan(self.image_width/2/focal)*180./math.pi
        self.FoVx = self.FoVy = fov
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1).to(world_view_transform.device)
        
        self.world_view_transform = world_view_transform
        self.full_proj_transform = world_view_transform @ self.projection_matrix
        if not torch.all(self.world_view_transform==0): 
            view_inv = torch.inverse(self.world_view_transform) 
        else:
            view_inv = self.world_view_transform
        self.camera_center = view_inv[3][:3]
        
    def update_transforms2(self, intrinsics, c2w, device=None):
        # intrinsics are normalized by image size, rather than in pixel units
        self.FoVx = 2*torch.arctan(1/(2*intrinsics[0,0,0])).to(c2w.device)
        self.FoVy = 2*torch.arctan(1/(2*intrinsics[0,1,1])).to(c2w.device)
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1).to(c2w.device)
        
        if not torch.all(c2w==0): 
            w2c = torch.inverse(c2w) 
        else:
            w2c = c2w
        
        self.world_view_transform = w2c.transpose(0, 1).to(c2w.device)
        self.full_proj_transform = (self.world_view_transform @ self.projection_matrix).to(c2w.device)
        self.camera_center = c2w[:3,3].to(c2w.device)

    def update_transforms_batch(self, intrinsics, c2w, device=None):
        # intrinsics are normalized by image size, rather than in pixel units
        self.FoVx = 2*torch.arctan(1/(2*intrinsics[0,0])).to(c2w.device)
        self.FoVy = 2*torch.arctan(1/(2*intrinsics[1,1])).to(c2w.device)
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1).to(c2w.device)
        
        if not torch.all(c2w==0): 
            w2c = torch.inverse(c2w) 
        else:
            w2c = c2w
        
        self.world_view_transform = w2c.transpose(0, 1).to(c2w.device)
        self.full_proj_transform = (self.world_view_transform @ self.projection_matrix).to(c2w.device)
        self.camera_center = c2w[:3,3].to(c2w.device)

## TODO: implement inheritance
# class MiniCam2(MiniCam):
#     def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
#                  image_name, uid,
#                  trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
#                  ):
#         super(MiniCam, self).__init__()
    

