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
# from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.renderer_3dmm import ImportanceRenderer ##replace with splatting renderer later
from training.volumetric_rendering.renderer_next3d import Pytorch3dRasterizer, face_vertices, generate_triangles, transform_points, batch_orth_proj, angle2matrix
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib

from ipdb import set_trace as st
import torch.nn as nn
from pytorch3d.io import load_obj
import torch.nn.functional as F

from gaussian_splatting.gaussian_model import GaussianModel
from gaussian_splatting.cameras import Camera
from gaussian_splatting.renderer import render

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
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
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        
        self._last_planes = None
        
        ### TODO: -------- next3d 3d face model --------
        self.topology_path = '/mnt/kostas-graid/datasets/xuyimeng/ffhq/head_template.obj' # DECA model
        self.load_lms = True
        
        # set pytorch3d rasterizer
        self.uv_resolution = 256
        self.rasterizer = Pytorch3dRasterizer(image_size=256)
        
        verts, faces, aux = load_obj(self.topology_path)
        uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
        faces = faces.verts_idx[None,...]

        # faces
        dense_triangles = generate_triangles(self.uv_resolution, self.uv_resolution)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None,:,:].contiguous())
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords) #[bz, ntv, 2]
        
        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3] 
        #TODO: The above concat all 1s to the original uv_coods (self.raw_uvcoords). Why?
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces) #(N, F, 3)
        self.register_buffer('face_uvcoords', face_uvcoords) # (N, F, 3, 3)

        self.orth_scale = torch.tensor([[5.0]])
        self.orth_shift = torch.tensor([[0, -0.01, -0.01]])
        # st() # check all data can be loaded
        self.verts = verts
        ### TODO: -------- next3d 3d face model [end] --------

    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        ## TODO: convert from triplane to 3DMM here
        textures = planes
        
        ### ----- original eg3d version -----
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
        # st() # check feature image.shape
        ### ----- original eg3d version [END] ----- 
        
        ### ----- next3d version -----
        # split vertices and landmarks 
        batch_size = ws.shape[0]
        if self.load_lms:
            ## TODO: to find out where the v is passed in
            v = (self.verts).repeat(batch_size,1,1).to(ws.device)
            v, lms = v[:, :5023], v[:, 5023:]
        rendering_views = [
            [0, 0, 0],
            [0, 90, 0],
            # [0, -90, 0],
            # [90, 0, 0]
        ]# use next3d rendering views first. If can rendered correctly, replace with eg3d c2w/w2c! Be aware!
        
        rendering_images, alpha_images, uvcoords_images, lm2ds = self.rasterize(v, lms, textures, rendering_views, batch_size, ws.device)

        ### ----- next3d version [END] -----
        
        st() # save rendering result from self.rasterize
        
        ### ----- gaussian splatting -----
        # camera setting 
        # TODO: the inputs?
        viewpoint_camera = Camera() 

        # create a guassian model using generated texture
        gaussian = GaussianModel(sh_degree=4)
        gaussian.create_from_generated_texture(v, textures)

        # raterization
        white_background = False
        bg_color = [1,1,1] if white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        rgb_image = render(viewpoint_camera, gaussian, None, background)

        ### ----- gaussian splatting [END] -----
                
        ## TODO: the below superresolution shall be kept?
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}
    
    ## next3d rasterize
    def rasterize(self, v, lms, textures, tforms, batch_size, device):
        rendering_images, alpha_images, uvcoords_images, transformed_lms = [], [], [], []


        for tform in tforms:
            v_flip, lms_flip = v.detach().clone(), lms.detach().clone()
            v_flip[..., 1] *= -1; lms_flip[..., 1] *= -1
            # rasterize texture to three orthogonal views
            st()
            tform = angle2matrix(torch.tensor(tform).reshape(1, -1)).expand(batch_size, -1, -1).to(device)
            transformed_vertices = (torch.bmm(v_flip, tform) + self.orth_shift.to(device)) * self.orth_scale.to(device)
            transformed_vertices = batch_orth_proj(transformed_vertices, torch.tensor([1., 0, 0]).to(device))
            transformed_vertices[:,:,1:] = -transformed_vertices[:,:,1:]
            transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10


            transformed_lm = (torch.bmm(lms_flip, tform) + self.orth_shift.to(device)) * self.orth_scale.to(device)
            transformed_lm = batch_orth_proj(transformed_lm, torch.tensor([1., 0, 0]).to(device))[:, :, :2]
            transformed_lm[:,:,1:] = -transformed_lm[:,:,1:]


            faces = self.faces.detach().clone()[..., [0,2,1]].expand(batch_size, -1, -1)
            attributes = self.face_uvcoords.detach().clone()[:, :, [0,2,1]].expand(batch_size, -1, -1, -1)


            rendering = self.rasterizer(transformed_vertices, faces, attributes, 256, 256)
            alpha_image = rendering[:, -1, :, :][:, None, :, :].detach()
            ## TODO: remove the following process of next3D
            # uvcoords_image = rendering[:, :-1, :, :]; grid = (uvcoords_image).permute(0, 2, 3, 1)[:, :, :, :2]
            # mask_face_eye = F.grid_sample(self.uv_face_mask.expand(batch_size,-1,-1,-1).to(device), grid.detach(), align_corners=False) 
            # alpha_image = mask_face_eye * alpha_image
            # if self.fill_mouth:
            #     alpha_image = fill_mouth(alpha_image)
            # uvcoords_image = mask_face_eye * uvcoords_image
            rendering_image = F.grid_sample(textures, grid.detach(), align_corners=False)


            rendering_images.append(rendering_image)
            alpha_images.append(alpha_image)
            uvcoords_images.append(None)
            # uvcoords_images.append(uvcoords_image)
            transformed_lms.append(transformed_lm)


        rendering_image_side = rendering_images[1] + rendering_images[2] # concatenate two side-view renderings
        alpha_image_side = (alpha_images[1].bool() | alpha_images[1].bool()).float()
        rendering_images = [rendering_images[0], rendering_image_side, rendering_images[3]]
        alpha_images = [alpha_images[0], alpha_image_side, alpha_images[3]]


        return rendering_images, alpha_images, uvcoords_images, transformed_lms
    
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
