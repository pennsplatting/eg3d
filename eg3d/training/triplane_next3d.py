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

from training.gaussian_splatting.gaussian_model import GaussianModel
from training.gaussian_splatting.cameras import MiniCam
from training.gaussian_splatting.renderer import render as gs_render
from pytorch3d.renderer import look_at_view_transform
from training.gaussian_splatting.utils.graphics_utils import getWorld2View, getProjectionMatrix
import numpy as np

from plyfile import PlyData
import math

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sh_degree,                  # Spherical harmonics degree.
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
        self.topology_path = '/home/1TB/model/head_template.obj' # DECA model
        self.verts_path = '/home/1TB/model/seed0000_head_template2_xzymean125.ply' # aligned with eg3d mesh in both scale and cam coord
        self.load_lms = True
        
        # set pytorch3d rasterizer
        self.uv_resolution = 256
        # self.rasterizer = Pytorch3dRasterizer(image_size=256)
        
        _, faces, aux = load_obj(self.topology_path)
        verts = self.load_aligend_verts(self.verts_path)
        
        uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
        faces = faces.verts_idx[None,...]

        # faces
        dense_triangles = generate_triangles(self.uv_resolution, self.uv_resolution)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None,:,:].contiguous())
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords[:,:verts.shape[0]]) #[bz, ntv, 2]
        
        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3] 
        #TODO: The above concat all 1s to the original uv_coods (self.raw_uvcoords). Why?
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        # TODO: FIXME change to a version where the uvcoords have the same number as verts
        self.register_buffer('uvcoords', uvcoords)
        # self.register_buffer('uvcoords', uvcoords[:,:verts.shape[0]])
        # st()
        self.register_buffer('uvfaces', uvfaces) #(N, F, 3)
        self.register_buffer('face_uvcoords', face_uvcoords) # (N, F, 3, 3)

        # self.orth_scale = torch.tensor([[5.0]])
        # self.orth_shift = torch.tensor([[0, -0.01, -0.01]])
        self.register_buffer('verts', verts)
        ### -------- next3d 3d face model [end] --------

        
        ### -------- gaussian splatting render --------
        self.gaussian_splatting_use_sr = False
        self.sh_degree = sh_degree
        self.gaussian = None
        self.viewpoint_camera = None
        
    def load_aligend_verts(self, ply_path):
        plydata = PlyData.read(ply_path)
        v_np = np.array(plydata['vertex'].data.tolist())
        return torch.tensor(v_np)
        
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def project3d_to_2d(self, intrinsic_matrix, c2w, fname):
        import numpy as np 
        import cv2 

        # Define intrinsic camera parameters 
        focal_length = 500
        image_width = 256
        image_height = 256
        # intrinsic_matrix = np.array([ 
        #     [focal_length, 0, image_width/2], 
        #     [0, focal_length, image_height/2], 
        #     [0, 0, 1] 
        # ]) 
        cam_matrix = intrinsic_matrix.detach().cpu().numpy()
        cam_matrix[:2] = cam_matrix[:2]*image_height
        cam_matrix = np.matmul(cam_matrix, c2w.detach().cpu().numpy()[:3])
        # cam_matrix = cam_matrix_torch.detach().cpu().numpy()
        # # st()

        # Define extrinsic camera parameters 
        # rvec = np.array([0, 0, 0], dtype=np.float32) 
        # tvec = np.array([0, 0, 0], dtype=np.float32).reshape(-1, 1)
        # rvec = np.zeros((3, 1), np.float32) 
        # tvec = np.zeros((3, 1), np.float32) 

        # Generate 3D points on a paraboloid 
        u_range = np.linspace(0, 1, num=40) 
        v_range = np.linspace(0, 1, num=40) 
        u, v = np.meshgrid(u_range, v_range) 
        x = u 
        y = v 
        z = u**2 + v**2

        points_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3) 

        # Project 3D points onto 2D plane 
        # points_2d, _ = cv2.projectPoints(points_3d, 
        #                                 rvec, tvec, 
        #                                 cam_matrix, 
        #                                 None) 
      
        points_3d_homo = np.concatenate([points_3d, np.zeros((points_3d.shape[0],1), np.float32)], axis=-1)
        # st()
        points_homo = np.matmul(cam_matrix, points_3d_homo.T).T
        points_2d = points_homo[:,:2] / points_homo[:,2:3]
        points_2d = points_2d[:,None]
        # st()
        # Plot 2D points 
        img = np.zeros((image_height, image_width), 
                    dtype=np.uint8) 
        # st()
        for point in points_2d.astype(int): 
            img = cv2.circle(img, tuple(point[0]), 2, 255, -1) 
            
        cv2.imwrite(fname, img)
        print('Saved to ', fname)
        if not intrinsic_matrix.sum()==0:
            st()
        # cv2.imshow('Image', img) 
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 
    
    def getWorld2View_from_eg3d_c(self, c2w):
        # transpose the R in c2w
        if not torch.all(c2w==0):
            w2c = torch.linalg.inv(c2w)
        else:
            w2c = torch.zeros_like(c2w)
        
        batch_R = w2c[:,:3,:3]
        batch_R_trans = batch_R.transpose(1,2).contiguous()
        w2c[:,:3,:3] = batch_R_trans
        return w2c
        # return w2c.contiguous() ## if the above have bug, try to make it contiguous

            
    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # intrinsics = c[:, 16:25].view(-1, 3, 3)
        
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # # Create a batch of rays for volume rendering
        # ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # # Create triplanes by running StyleGAN backbone
        # N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        ## TODO: convert from triplane to 3DMM here
        textures_gen_batch = planes
        
        ### ----- original eg3d version -----
        # planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1]) # torch.Size([4, 3, 32, 256, 256])

        # # Perform volume rendering
        # feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        # # feature_samples: [4, 4096, 32]

        # # Reshape into 'raw' neural-rendered image
        # H = W = self.neural_rendering_resolution
        # feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        # depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # # Run superresolution to get final image
        # rgb_image = feature_image[:, :3] # rgb: torch.Size([4, 3, 64, 64]), feature_image: torch.Size([4, 32, 64, 64])
        # # st() # check feature image.shape
        ### ----- original eg3d version [END] ----- 
        
        ### ----- gaussian splatting -----
        # camera setting 
        # TODO: determine the inputs
        # width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform
        # TODO: chenge 
        # R, T = look_at_view_transform(10, 0, 0)
        # world_view_transform = getWorld2View(R, T).transpose(0, 1).cuda() 
        world_view_transform_batch = self.getWorld2View_from_eg3d_c(cam2world_matrix)

        ## gaussian_splatting rendering resolution: 
        ## V1: gaussian rendering -> 64 x 64 -> sr module --> 256 x 256: self.gaussian_splatting_use_sr = True
        ## v2: gausian rendering -> 256 x 256 : self.gaussian_splatting_use_sr = False
        if self.gaussian_splatting_use_sr:
            image_size = self.neural_rendering_resolution # chekc
        else:
            image_size = 256
        
        # # replace the hard-coded focal with intrinsics from c
        # focal = 1015 
        # focal = intrinsics[0,0,1] * half_image_width * 2 # check
        # fovy = fovx = 2*torch.arctan(half_image_width/focal)*180./math.pi
        
        # TODO: modify z-near and z-far in projection matrix
        # projection_matrix = getProjectionMatrix(0.01, 50, fovx, fovy).transpose(0, 1)
        z_near, z_far = 10, 500
        # st()
        # projection_matrix = getProjectionMatrix(z_near, z_far, fovx, fovy).transpose(0, 1).to(world_view_transform_batch.device)
        rgb_image_batch = []
        
        # initialize camera and gaussians
        self.viewpoint_camera = MiniCam(c, image_size, image_size, z_near, z_far, world_view_transform_batch.device)
        self.gaussian = GaussianModel(self.sh_degree)
        
        for world_view_transform, textures_gen in zip(world_view_transform_batch,textures_gen_batch):
            # full_proj_transform = world_view_transform @ projection_matrix
            self.viewpoint_camera.update_transforms(world_view_transform)

            # create a guassian model using generated texture
            # TODO: change the input feature channel
            # gaussian = GaussianModel(sh_degree=3)
            # map textures to gaussian features 
    
            ## TODO: can gaussiam splatting run batch in parallel?
            textures = F.grid_sample(textures_gen[None], self.raw_uvcoords.detach().unsqueeze(1), align_corners=False)
            # gaussian.create_from_generated_texture(self.verts, textures)
            self.gaussian.create_from_ply2(self.verts, textures)

            # raterization
            white_background = False
            bg_color = [1,1,1] if white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            _rgb_image = gs_render(self.viewpoint_camera, self.gaussian, None, background)["render"]
            rgb_image_batch.append(_rgb_image[None])
        
        rgb_image = torch.cat(rgb_image_batch) # [4, 3, gs_res, gs_res]

        ### ----- gaussian splatting [END] -----
                
        ## TODO: the below superresolution shall be kept?
        ## currently keeping the sr module below. TODO: shall we replace the feature image by texture_uv_map or only the sampled parts?
        # sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        if self.gaussian_splatting_use_sr:
            sr_image = self.superresolution(rgb_image, textures_gen_batch, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        else:
            sr_image = rgb_image
            rgb_image = rgb_image[:,:,::8, ::8] # TODO: FIXME change this downsample to a smoother gaussian filtering
            
        ## TODO: render depth_image. May not explicitly calculated from the face model since its the same for all.
        depth_image = torch.zeros_like(rgb_image) # (N, 1, H, W)

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
