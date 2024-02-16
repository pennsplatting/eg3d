# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer
# from training.volumetric_rendering.renderer_3dmm import ImportanceRenderer ##replace with splatting renderer later
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib

from ipdb import set_trace as st
import cv2
import torch.nn as nn
import torch.nn.functional as F

from training.gaussian_splatting.gaussian_model import GaussianModel
from training.gaussian_splatting.cameras import MiniCam
from training.gaussian_splatting.renderer import render as gs_render
from training.gaussian_splatting.utils.graphics_utils import getWorld2View, getProjectionMatrix
import numpy as np

from plyfile import PlyData
import scipy.io as sio

# FIXME: replace with gt texture for debug
# import sys
# sys.path.append('/home/zxy/eg3d/eg3d/data')
# from gt.get_uv_texture import get_uv_texture


def print_grad(name, grad):
    print(f"{name}:")
    if torch.all(grad==0):
        print("grad all 0s")
        return 
    # print(grad)
    print('\t',grad.max(), grad.min(), grad.mean())
    
@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        plane_resolution,
        sh_degree           = 3,    # Spherical harmonics degree.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        # TODO: maybe synthesize uv texture directly with stylegan2 backbone (img_channels=3)
        # so that there's no need for decoder
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=plane_resolution, img_channels=16, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        # self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        # self.text_decoder = TextureDecoder(96, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': (sh_degree + 1) ** 2 * 3})
        self.text_decoder = TextureDecoder(96, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 3})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        
        self._last_planes = None
                
        ### -------- gaussian splatting render --------
        self.gaussian_splatting_use_sr = False
        self.sh_degree = sh_degree
        # self.gaussian = None
        # self.viewpoint_camera = None
        
        # initialize camera and gaussians
        
        ## gaussian_splatting rendering resolution: 
        ## V1: gaussian rendering -> 64 x 64 -> sr module --> 256 x 256: self.gaussian_splatting_use_sr = True
        ## v2: gausian rendering -> 256 x 256 : self.gaussian_splatting_use_sr = False
        if self.gaussian_splatting_use_sr:
            image_size = self.neural_rendering_resolution # chekc
        else:
            image_size = 512
        z_near, z_far = 0.1, 2 # TODO: find suitable value for this
        self.viewpoint_camera = MiniCam(image_size, image_size, z_near, z_far)
        self.gaussian = GaussianModel(self.sh_degree, plane_resolution)
    
        
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        ws = self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return ws
            
    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # real_img = self.gt_uv_map(c)
        
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

    
        ## FIXME: replace with gt texture for debug 
        # textures = torch.tensor(get_uv_texture(), dtype=torch.float, device="cuda") # (53215, 3)
        
        original_eg3d = False
        if original_eg3d:
            ## ----- original eg3d version -----
            # Create a batch of rays for volume rendering
            ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

            # Create triplanes by running StyleGAN backbone
            N, M, _ = ray_origins.shape
            
            planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1]) # torch.Size([4, 3, 32, 256, 256])

            # Perform volume rendering
            feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
            # feature_samples: [4, 4096, 32]

            # Reshape into 'raw' neural-rendered image
            H = W = self.neural_rendering_resolution
            feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
            depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

            # Run superresolution to get final image
            rgb_image = feature_image[:, :3] # rgb: torch.Size([4, 3, 64, 64]), feature_image: torch.Size([4, 32, 64, 64])
            st() # check feature image.shape
            sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
            
            # ## debug grad
            # planes.requires_grad_(True)
            # planes.register_hook(lambda grad: print_grad("planes.requires_grad", grad))
            # rgb_image.requires_grad_(True)
            # rgb_image.register_hook(lambda grad: print_grad("rgb_image.requires_grad", grad))
            
            ### ----- original eg3d version [END] ----- 
        
        
        else:
            
            ### ----- gaussian splatting -----
            feature_gen_batch = planes.permute(0, 2, 3, 1) # (B, K, H, W) -> (B, H, W, K)
            rgb_image_batch = []
            alpha_image_batch = [] # mask

            ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, 128)
            # raterization
            white_background = True
            bg_color = [1,1,1] if white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            # print(f"--textures_gen_batch: min={textures_gen_batch.min()}, max={textures_gen_batch.max()}, mean={textures_gen_batch.mean()}, shape={textures_gen_batch.shape}")
            
            for _cam2world_matrix, feature_gen, _ray_origins, _ray_directions in zip(cam2world_matrix, feature_gen_batch, ray_origins, ray_directions):
                self.viewpoint_camera.update_transforms2(intrinsics, _cam2world_matrix)

                ## TODO: can gaussiam splatting run batch in parallel?
                # textures = F.grid_sample(texture_gen[None], self.raw_uvcoords.unsqueeze(1), align_corners=False) # (1, 48, 1, 5023)
                # textures.requires_grad_(True) 
                # textures.register_hook(lambda grad: print_grad("--textures.requires_grad", grad))
                
                self.gaussian.update(feature_gen, _ray_origins, _ray_directions)
                res = gs_render(self.viewpoint_camera, self.gaussian, None, background)
                _rgb_image = res["render"]
                _alpha_image = res["alpha"]
                ## FIXME: output from gs_render should have output rgb range in [0,1], but now have overflowed to [0,20+]
                
                rgb_image_batch.append(_rgb_image[None])
                alpha_image_batch.append(_alpha_image[None])

            
            rgb_image = torch.cat(rgb_image_batch) # [4, 3, gs_res, gs_res]
            alpha_image = torch.cat(alpha_image_batch)
            ## FIXME: try different normalization method to normalize rgb image to [-1,1]
            # rgb_image = (rgb_image - 0.5) * 2

            # print(f"-rgb_image: min={rgb_image.min()}, max={rgb_image.max()}, mean={rgb_image.mean()}, shape={rgb_image.shape}")
            
            # rgb_image.requires_grad_(True)
            # rgb_image.register_hook(lambda grad: print_grad("rgb_image.requires_grad", grad))
            
            ## TODO: the below superresolution shall be kept?
            ## currently keeping the sr module below. TODO: shall we replace the feature image by texture_uv_map or only the sampled parts?
            if self.gaussian_splatting_use_sr:
                sr_image = self.superresolution(rgb_image, feature_gen_batch, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
            else:
                sr_image = rgb_image
                rgb_image = rgb_image[:,:,::8, ::8] # TODO: FIXME change this downsample to a smoother gaussian filtering
            
           
            ## TODO: render depth_image. May not explicitly calculated from the face model since its the same for all.
            # depth_image = torch.zeros_like(rgb_image) # (N, 1, H, W)
            ### ----- gaussian splatting [END] -----

        return {'image': sr_image, 'image_raw': rgb_image, 'image_mask': alpha_image}
    
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        # return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)
        ## TODO: map the planes to the 3DMM model
        st()
        print('')
        face_model = tdmmModel.fit_feature_planes(planes)
        ## TODO2: check whether the self.decoder can be kept or not
        return self.renderer.run_model(face_model, self.decoder, coordinates, directions, self.rendering_kwargs)


    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

# decode features to SH
class TextureDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features):
        # features (4, 96, 256, 256) -> (4, 16*3, 256, 256)
        # Aggregate features
        sampled_features = sampled_features.permute(0,2,3,1)
        x = sampled_features

        N, H, W, C = x.shape
        x = x.reshape(N*H*W, C)
        
        x = self.net(x)
        
        ## added sigmoid to make x in range 0~1
        x = torch.sigmoid(x)*(1 + 2*0.001) - 0.001
    
        x = x.reshape(N, H, W, -1)
        return x.permute(0, 3, 1, 2)
