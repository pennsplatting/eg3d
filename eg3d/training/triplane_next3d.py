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
from training.volumetric_rendering.renderer_next3d import Pytorch3dRasterizer, face_vertices, generate_triangles, transform_points, batch_orth_proj, angle2matrix
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib

from ipdb import set_trace as st
import torch.nn as nn
from pytorch3d.io import load_obj
import torch.nn.functional as F

from training.gaussian_splatting.gaussian_model import GaussianModel
from training.gaussian_splatting.cameras import MiniCam
from training.gaussian_splatting.renderer import render as gs_render
from pytorch3d.renderer import look_at_view_transform
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
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        # self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.text_decoder = TextureDecoder(96, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': (sh_degree + 1) ** 2 * 3})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        
        self._last_planes = None
        
        ### TODO: -------- 3dmm face model --------
        self.uv_resolution = 256
        
        self.load_face_model()
        ### -------- 3dmm face model [end] --------

        
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
            image_size = 512 # ffhq: 512, 3dmm: 256
        # z_near, z_far = 0.01, 100 # TODO: find suitable value for this. 0.01 and 100 for overfitting 3dmm
        z_near, z_far = 1000, -100 # as in face3d ## NO affect
        print(f"z_near, z_far: {z_near, z_far}")
        self.viewpoint_camera = MiniCam(image_size, image_size, z_near, z_far)
        self.gaussian = GaussianModel(self.sh_degree, self.verts)
        # setattr(self,"gaussian", GaussianModel(self.sh_degree, self.verts))
        # setattr(self,"sh_degree",sh_degree)
        
    
    def process_uv_with_res(self, uv_coords, uv_h = 256, uv_w = 256):
    # discard this method: this will map uv coord values to [0,res], while grid sample takes uv in [-1,1]
        uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
        uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
        uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
        uv_coords = np.stack((uv_coords))
        return uv_coords
    
    def process_uv(self, uv_coords, uv_h = 256, uv_w = 256):
        return uv_coords*2-1.0
    
    def load_face_model(self):
        _overfitting = True
        # verts_path = '../dataset_preprocessing/3dmm/gs_colored_vertices_700norm.ply' # aligned with eg3d mesh in both scale and cam coord
        ## align with the actual training space of 3dmm, rather than the saved ply space
        verts_path = '../dataset_preprocessing/3dmm/gs_flipped_uv_textured_vertices_700norm.ply' # aligned with eg3d mesh in both scale and cam coord
        verts_path_ovft = '../dataset_preprocessing/3dmm/points3d.ply' # overfitting the 3dmm data
        
        plydata = PlyData.read(verts_path)
        
        verts_rgb = np.stack([plydata['vertex'][ax] for ax in ['red', 'green', 'blue']], axis=-1) # [V,3], 0~255
        self.register_buffer('verts_rgb', torch.tensor(verts_rgb, dtype=torch.float, device='cuda'))

        box_center = self.rendering_kwargs['box_warp'] / 2
        box_scale = 512
        if _overfitting:
            print(f"Loading vertices from {verts_path_ovft}")
            plydata = PlyData.read(verts_path_ovft)
            box_center = 0.
            box_scale = 1.0
        print(f"box_center: {box_center }; box_scale: {box_scale}")    
        verts = np.stack([plydata['vertex'][ax] for ax in ['x', 'y', 'z']], axis=-1) # [V,3]
        # normalize to [-0.5, 0.5] and place the center at origin
        verts_norm = verts /box_scale  - box_center
        verts_norm = torch.tensor(verts_norm, dtype=torch.float, device='cuda')
        # verts_norm[:,1:3] *= -1
        verts_norm[:,:] *= -1
        # st()
        # verts_norm[:,0] *= -1
        self.register_buffer('verts', verts_norm)
        print(f"self.verts.center:{self.verts.mean(dim=0)}")
        
        
        # load uv coords & gt uv map
        uv_h, uv_w = self.uv_resolution, self.uv_resolution
        uv_coord_path = '../dataset_preprocessing/3dmm/BFM_UV.mat'
        C = sio.loadmat(uv_coord_path)
        uv_coords = C['UV'].copy(order = 'C') #(53215, 2) = [V, 2]
    
        uv_coords_processed = self.process_uv(uv_coords, uv_h, uv_w) #(53215, 2)
        # uv_coords_processed = uv_coords_processed.astype(np.int32)
        self.register_buffer('raw_uvcoords', torch.tensor(uv_coords_processed[None], dtype=torch.float, device='cuda')) #[B, V, 2]
        
            
    def load_aligend_verts(self, ply_path):
        plydata = PlyData.read(ply_path)
        # normalize to [-0.5, 0.5] and place the center at origin
        v_np = np.array(plydata['vertex'].data.tolist()) / 512.0 - self.rendering_kwargs['box_warp'] / 2
        return torch.tensor(v_np, dtype=torch.float, device='cuda')
        
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    
    def getWorld2View_from_eg3d_c(self, c2w):
        # print(f"c2w before:{c2w}")
        c2w[:, :3, 1:3] *= -1 # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        # print(f"c2w after:{c2w}")
        
        # transpose the R in c2w
        if not torch.all(c2w==0):
            w2c = torch.linalg.inv(c2w)
        else:
            w2c = torch.zeros_like(c2w)
        
        # batch_R = w2c[:,:3,:3]
        # batch_R_trans = batch_R.transpose(1,2).contiguous()
        # w2c[:,:3,:3] = batch_R_trans
        # st()
        return w2c
        # return w2c.contiguous() ## if the above have bug, try to make it contiguous
    
    
    
            
    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
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
        overfitting = True
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
         
            ### ----- original eg3d version [END] ----- 
        
        
        else:
            
            ### ----- gaussian splatting -----
            # Reshape output into three 32-channel planes
            ## TODO: convert from triplane to 3DMM here, maybe need a net to decode the feature to RGB or SH
            # textures_gen_batch = planes # (4, 96, 256, 256)
            textures_gen_batch = self.text_decoder(planes) # (4, 96, 256, 256) -> (4, SH, 256, 256), range [0,1]
                
            # camera setting 
            world_view_transform_batch = self.getWorld2View_from_eg3d_c(cam2world_matrix).transpose(2,1) # (4, 4, 4) 
    
            rgb_image_batch = []
            
            # print(f"--textures_gen_batch: min={textures_gen_batch.min()}, max={textures_gen_batch.max()}, mean={textures_gen_batch.mean()}, shape={textures_gen_batch.shape}")
            for world_view_transform, textures_gen in zip(world_view_transform_batch,textures_gen_batch):
                self.viewpoint_camera.update_transforms(intrinsics, world_view_transform, direct_fov=overfitting)

                ## TODO: can gaussiam splatting run batch in parallel?
                textures = F.grid_sample(textures_gen[None], self.raw_uvcoords.unsqueeze(1), align_corners=False) # (1, 96, 1, 5023)
               
                self.gaussian.update_rgb_textures(self.verts_rgb)
             
                # raterization
                white_background = False
                bg_color = [1,1,1] if white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                
                _rgb_image = gs_render(self.viewpoint_camera, self.gaussian, None, background)["render"]
                ## FIXME: output from gs_render should have output rgb range in [0,1], but now have overflowed to [0,20+]
                
                rgb_image_batch.append(_rgb_image[None])
            
            rgb_image = torch.cat(rgb_image_batch) # [4, 3, gs_res, gs_res]
            ## FIXME: try different normalization method to normalize rgb image to [0,1]
            rgb_image = (rgb_image / rgb_image.max() - 0.5) * 2
            # print(f"-rgb_image: min={rgb_image.min()}, max={rgb_image.max()}, mean={rgb_image.mean()}, shape={rgb_image.shape}")
            
            ## TODO: the below superresolution shall be kept?
            ## currently keeping the sr module below. TODO: shall we replace the feature image by texture_uv_map or only the sampled parts?
            if self.gaussian_splatting_use_sr:
                sr_image = self.superresolution(rgb_image, textures_gen_batch, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
            else:
                sr_image = rgb_image
                rgb_image = rgb_image[:,:,::2, ::2] # TODO: FIXME change this downsample to a smoother gaussian filtering
            
           
            ## TODO: render depth_image. May not explicitly calculated from the face model since its the same for all.
            depth_image = torch.zeros_like(rgb_image) # (N, 1, H, W)
            ### ----- gaussian splatting [END] -----

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}
    
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
