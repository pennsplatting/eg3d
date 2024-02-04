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
from training.volumetric_rendering.renderer import ImportanceRenderer, sample_from_planes, generate_planes
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib

from ipdb import set_trace as st
import cv2
import torch.nn as nn
import torch.nn.functional as F

# from training.gaussian_splatting.gaussian_model import GaussianModel
from training.gaussian_splatting.gaussian_model_xyz_offset import GaussianModel_OffsetXYZ as GaussianModel
from training.gaussian_splatting.gaussian_model_with_bg import GaussianModel_WithBg as GaussianModel_BG
from training.gaussian_splatting.cameras import MiniCam
from training.gaussian_splatting.renderer import render as gs_render
from training.gaussian_splatting.renderer_with_bg import render as gs_render_with_bg
from training.gaussian_splatting.utils.graphics_utils import getWorld2View, getProjectionMatrix

import numpy as np

from plyfile import PlyData
import scipy.io as sio
import random
import copy
import trimesh
import os

from scipy.io import loadmat
import os.path as osp

# FIXME: replace with gt texture for debug
# import sys
# sys.path.append('/home/zxy/eg3d/eg3d/data')
# from gt.get_uv_texture import get_uv_texture


def load_mesh_to_tensor(file_path, return_color=False, device='cuda'):
    mesh = trimesh.load(file_path, process=False)
    
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    # faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    if return_color:
        vertex_colors = torch.tensor(mesh.visual.vertex_colors, dtype=torch.float32, device=device) / 255.0  # Normalize to [0, 1]
        return vertices, vertex_colors
    return vertices

def load_mesh_with_colors(file_path):
    # Load the mesh with color information
    mesh = trimesh.load(file_path, process=False)

    # Access the vertices and vertex colors
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    vertex_colors = torch.tensor(mesh.visual.vertex_colors, dtype=torch.float32) / 255.0  # Normalize to [0, 1]

    return vertices, vertex_colors


def print_grad(name, grad):
    print(f"{name}:")
    if torch.all(grad==0):
        print("grad all 0s")
        return 
    # print(grad)
    print('\t',grad.max(), grad.min(), grad.mean())

def combine_list_of_gs(pc_list: [GaussianModel]):
    # get the attr of each gs
    pc_sh = set([_pc.active_sh_degree for _pc in pc_list])
    assert len(pc_sh) == 1
    pc_active_sh_degree = pc_sh.pop()
    
    pc_xyz = torch.cat([_pc.get_xyz for _pc in pc_list], dim=0)
    pc_features = torch.cat([_pc.get_features for _pc in pc_list], dim=0)
    pc_opacity = torch.cat([_pc.get_opacity for _pc in pc_list], dim=0)
    pc_scaling = torch.cat([_pc.get_scaling for _pc in pc_list], dim=0)
    pc_rotation = torch.cat([_pc.get_rotation for _pc in pc_list], dim=0)
    
    # create a new gs with combined attr
    combined_gs = GaussianModel(pc_active_sh_degree, copy.deepcopy(pc_xyz))
    
    # TODO: change the shape of features
    combined_gs.update_sh_texture_combine(pc_features)
    
    # to update gaussian bank
    # def 
    #     V, C = feature_uv.shape # (1, 48, 1, 5023)
    #     features = feature_uv.reshape(V,3,C//3).contiguous() # [5023, 3, 16] [V, 3, C']
       
    #     self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()#.requires_grad_(True) # [V, 1, 3]
    #     self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()#.requires_grad_(True)# [V, sh degree - 1, 3]
    combined_gs.update_opacity(pc_opacity)
    # def update_opacity(self, opacity):
    #     self._opacity = opacity

    combined_gs.update_scaling(pc_scaling)
    # def update_scaling(self, scaling, max_s=None, min_s=None):  
    #     # clamp settings
    #     self.max_s = max_s
    #     self.min_s = min_s
        
    #     # update
    #     self._scaling = scaling
    
    combined_gs.update_rotation(pc_rotation)
    # def update_rotation(self, rotation):
    #     self._rotation = rotation
    
    return combined_gs
    

    
    
@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        num_gaussians,              # Number of gaussian bases in the bank.
        optimize_gaussians, 
        init_from_the_same_canonical, 
        render_fg_bg_separately,
        no_activation_in_decoder,
        bg_resolution, 
        bg_depth,
        low_res_training,
        regularize_fg_opacity = True,
        sh_degree           = 3,    # Spherical harmonics degree.
        sr_num_fp16_res     = 0,
        text_decoder_kwargs = {},   # GS TextureDecoder
        bg_decoder_kwargs   = {},   # GS BackgroundDecoder
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
        self.planes_channels_bg = 32
        self.planes_channels_uv = 32*3
        assert self.planes_channels_bg > 0
        self.planes_channels = self.planes_channels_uv + self.planes_channels_bg
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=self.planes_channels, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        
        self._last_planes = None
        
        ### TODO: -------- 3dmm face model --------
        self.uv_resolution = 256
        
        self.load_face_model()
        ### -------- 3dmm face model [end] --------

        
        ### -------- gaussian splatting render --------
        self.gaussian_splatting_use_sr = False
        self.render_fg_bg_separately = render_fg_bg_separately


        # use colors_precomp instead of shs in gaussian_model.render()
        self.use_colors_precomp = False
        # when use_colors_precomp, SHdegree=0
        # sh_degree = sh_degree * (1-self.use_colors_precomp)
        sh_degree = 0
        print(f"G->SHdegree={sh_degree}, use_colors_precomp={self.use_colors_precomp}")
        self.sh_degree = sh_degree
    
        # initialize camera and gaussians
        
        z_near, z_far = 0.1, 2 # TODO: find suitable value for this
        
        ## gaussian_splatting rendering resolution: 
        ## V1: gaussian rendering -> 64 x 64 -> sr module --> 256 x 256: self.gaussian_splatting_use_sr = True
        ## v2: gausian rendering -> 256 x 256 : self.gaussian_splatting_use_sr = False
        # res
        self.low_res_training = low_res_training
        self.regularize_fg_opacity = regularize_fg_opacity

        if self.low_res_training:
            image_size = self.neural_rendering_resolution
            bg_resolution = min(bg_resolution, image_size)
        else:
            if self.gaussian_splatting_use_sr:
                image_size = self.neural_rendering_resolution # check
            else:
                image_size = 512
        
        self.viewpoint_camera = MiniCam(image_size, image_size, z_near, z_far)
        self.image_size = image_size # for recording only
        
        # create a bank of gaussian models
        self.num_gaussians = num_gaussians
        print(f"We have init {self.num_gaussians} gaussians.\n")  

        # by default

        self.feature_structure = 'UV'
        
        # raterization
        self.white_background = True
        # print(f"GS bg is white:{white_background}: black:{not white_background}")
        bg_color = [1,1,1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.alpha_option = 'alpha'
        assert self.alpha_option in ['alpha', 'silhouette']
        self.normalize_rgb_image = True
        
        # load reduced index for front face only model
        self.keep_only_front_face_UV()

        self.init_from_the_same_canonical = False
        
        if self.init_from_the_same_canonical:
            self.feature_structure = f'UV_{self.raw_uvcoords.shape[1]}'
            
            # init gaussian bank from the same canonical face
            for i in range(1, self.num_gaussians+1): 
                setattr(self, f'g{i}', GaussianModel(self.sh_degree, copy.deepcopy(self.verts), i))
        else:
            
            # init gaussian bank from regressed 3DMM parameters
            target_mean = self.verts.mean(dim=0)
            
            scale_factor = 1 / 3.4 # observed by the scale difference: eg3d = scale_factor * 3dmm
            # obj_folder = '/home/ritz/eg3d/eg3d/data/gaussian_bank/front_face_gaussian_1k/home/xuyimeng/Repo/eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/00000_1k_no_rotation/epoch_20_000000'
            obj_folder = '/home/xuyimeng/Repo/eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/00000_1k_no_rotation/epoch_20_000000'
            obj_paths = [os.path.join(obj_folder, i) for i in sorted(os.listdir(obj_folder)) if i.endswith('obj')]
            total_different = len(obj_paths)
            
            for i in range(1, self.num_gaussians+1):
                obj_index = (2*(i-1))%total_different
                file_path = obj_paths[obj_index]
                print(f"getting the {obj_index} .obj from {file_path}")
                
                vertices = load_mesh_to_tensor(file_path)
                
                # normalize to eg3d space
                vertices = (vertices - vertices.mean(dim=0)) * scale_factor + target_mean
                # setattr(self, f'g{i}', GaussianModel(self.sh_degree, copy.deepcopy(vertices), i))
                setattr(self, f'v{i}', copy.deepcopy(vertices))
            self.plane_axes_gs = generate_planes().to(self.verts.device)
            
        
        # all the following things must be executed after "self->keep_only_front_face_UV()"
        
        # gt, init with verts_rgb
        # self.gaussian_debug = GaussianModel(self.sh_degree, copy.deepcopy(self.verts))
        # self.gaussian_debug.update_rgb_textures(self.verts_rgb)
        
        # bg gaussian
        self.bg_resolution = bg_resolution
        bg_verts = torch.rand([self.bg_resolution * self.bg_resolution, 3]).to(self.verts.device) - 0.5
        self.bg_verts = bg_verts
        self.bg_depth = bg_depth
        
        # self.bg_gaussian = GaussianModel_BG(self.sh_degree, bg_verts)
        uv_size = (self.bg_resolution, self.bg_resolution)
        self.bg_uvcoord = self.create_uniform_uv_coordinates(uv_size).to(device='cuda') # in range [-1,1]
        # print(self.bg_uvcoord)
        # st()

        self.decode_before_gridsample = True
        
        self.text_decoder_class = 'TextureDecoder_allAttributes'
        assert self.text_decoder_class in ['TextureDecoder', 'TextureDecoder_noSigmoid', 'TextureDecoder_allAttributes']
        
        # texture decoder: the output dim is aware of whether use rgb or sh to render gaussian, controled by use_colors_precomp
        if ('UV' in self.feature_structure):
            # decode -> UV sample
            
            if self.text_decoder_class == 'TextureDecoder_allAttributes':
                text_decoder_options={'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 0}
                text_decoder_options.update(text_decoder_kwargs)
                bg_decoder_options = {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 0}
                bg_decoder_options.update(bg_decoder_kwargs)
                
                if no_activation_in_decoder:
                    self.text_decoder = TextureDecoder_allAttributes_noActivations(self.planes_channels_uv, text_decoder_options)
                    self.bg_decoder = TextureDecoder_allAttributes_noActivations(self.planes_channels_bg, bg_decoder_options)
                else:
    
                    self.text_decoder = TextureDecoder_allAttributes(self.planes_channels_uv, text_decoder_options)
                    self.bg_decoder = TextureDecoder_allAttributes(self.planes_channels_bg, bg_decoder_options)
                    
            elif self.text_decoder_class == 'TextureDecoder_noSigmoid':   
                self.text_decoder = TextureDecoder_noSigmoid(96, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': (sh_degree + 1) ** 2 * 3})
            elif self.text_decoder_class == 'TextureDecoder':
                self.text_decoder = TextureDecoder(96, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': (sh_degree + 1) ** 2 * 3})
            else:
                self.text_decoder = None
            
        else:
            # triplane sample -> decode
            self.text_decoder = TextureDecoder_triplane(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': (sh_degree + 1) ** 2 * 3})
            
            
    def keep_only_front_face_UV(self):
        # bfm_folder = '/home/ritz/eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/BFM'
        bfm_folder = '/home/xuyimeng/Repo/eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/BFM'
        index_exp = loadmat(osp.join(bfm_folder, 'BFM_front_idx.mat'))
        index_exp = index_exp['idx'].astype(np.int32) - 1  # starts from 0 (to 53215)
        
        assert hasattr(self, 'raw_uvcoords')
        # assert int(index_exp.shape[0]) == int(self.g1._xyz.shape[0])
               
        self.raw_uvcoords = self.raw_uvcoords[0][index_exp].permute(1,0,2)
        self.verts = self.verts[index_exp].squeeze(1)
        self.verts_rgb = self.verts_rgb[index_exp].squeeze(1)
        
        self.feature_structure = f'UV_{self.raw_uvcoords.shape[1]}'

    def record_attributes_to_json(self):
        
        attributes_to_record = {
            # Add other boolean attributes here
            "gaussian_splatting_use_sr": self.gaussian_splatting_use_sr,
            "use_colors_precomp": self.use_colors_precomp,
            # "sh_degree(start)": self.g1.active_sh_degree,
            "sh_degree(max)": self.sh_degree,
            # "update interval": self.g1.update_interval,
            "num_gaussians": self.num_gaussians,
            "gassian render background": 'white' if self.white_background else 'black',
            "gaussian bank init": 'same' if self.init_from_the_same_canonical else f'different_{self.num_gaussians}',
            # "gaussian_model": self.g1.__class__.__name__,
            # Add more attributes as needed
            "feature_structure": self.feature_structure,
            "texture_decoder": self.text_decoder.__class__.__name__,
            "texture_decoder output scale": getattr(self.text_decoder, 'output_scale', None),
            "texture_decoder xyz_offset_scale": getattr(self.text_decoder, 'xyz_offset_scale', None),
            "texture_decoder scale_bias": getattr(self.text_decoder, 'scale_bias', None),
            "texture_decoder scale_factor": getattr(self.text_decoder, 'scale_factor', None),
            "texture_decoder output attriutes": getattr(self.text_decoder, 'options', None),
            "decode_before_gridsample": self.decode_before_gridsample,
            # Rendering 
            "alpha or silhouettes": self.alpha_option,
            "normalize rendered rgb_image": self.normalize_rgb_image,
            "low_res_training": self.low_res_training,
            "gs rendered image_size": self.image_size,
            # For adding BG
            'backbone output planes_channels(All)': self.planes_channels,
            'backbone output planes_channels(Background)': self.planes_channels_bg,
            # "bg_gaussian": self.bg_gaussian.__class__.__name__,
            "bg_resolution": self.bg_resolution, 
            "bg_depth": self.bg_depth,
            "bg_decoder output attriutes": getattr(self.bg_decoder, 'options', None),

        }
        
        return attributes_to_record
    
    import torch

    def create_uniform_uv_coordinates(self, texture_size):
        # Generate 1D coordinates in the range [-1, 1]
        u = torch.linspace(-1, 1, texture_size[1])
        v = torch.linspace(-1, 1, texture_size[0])

        # Create a 2D grid of coordinates
        U, V = torch.meshgrid(u, v)

        # Stack U and V to create the final UV coordinates
        uv_coordinates = torch.stack([U, V], dim=-1)

        return uv_coordinates
            
    def get_a_gaussian(self):
        gs_i = random.randint(1, self.num_gaussians) # upper bound is included
        # print(gs_i)
        return getattr(self, f'g{gs_i}')
    
    def get_a_gaussian_verts(self):
        gs_i = random.randint(1, self.num_gaussians) # upper bound is included
        # print(gs_i)
        return getattr(self, f'v{gs_i}')
        
    
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
        # verts_path = '../dataset_preprocessing/3dmm/gs_colored_vertices_700norm.ply' # aligned with eg3d mesh in both scale and cam coord
        ## align with the actual training space of 3dmm, rather than the saved ply space
        verts_path = 'dataset_preprocessing/3dmm/gs_flipped_uv_textured_vertices_700norm.ply' # aligned with eg3d mesh in both scale and cam coord
                
        plydata = PlyData.read(verts_path)
        verts = np.stack([plydata['vertex'][ax] for ax in ['x', 'y', 'z']], axis=-1) # [V,3]
        # normalize to [-0.5, 0.5] and place the center at origin
        verts_norm = verts / 512.0 - self.rendering_kwargs['box_warp'] / 2
        verts_norm = torch.tensor(verts_norm, dtype=torch.float, device='cuda')
        self.register_buffer('verts', verts_norm)
        
        verts_rgb = np.stack([plydata['vertex'][ax] / 255.0 for ax in ['red', 'green', 'blue']], axis=-1) # [V,3], 0~255 -> 0~1
        self.register_buffer('verts_rgb', torch.tensor(verts_rgb, dtype=torch.float, device='cuda'))
        
        # load uv coords & gt uv map
        uv_h, uv_w = self.uv_resolution, self.uv_resolution
        # uv_coord_path = '../dataset_preprocessing/3dmm/BFM_UV.mat'
        uv_coord_path = 'dataset_preprocessing/3dmm/BFM_UV.mat'
        C = sio.loadmat(uv_coord_path)
        uv_coords = C['UV'].copy(order = 'C') #(53215, 2) = [V, 2]
    
        uv_coords_processed = self.process_uv(uv_coords, uv_h, uv_w) #(53215, 2)
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
        # if not intrinsic_matrix.sum()==0:
        #     st()
        # cv2.imshow('Image', img) 
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 
    
    def getWorld2View_from_eg3d_c(self, c2w):
        c2w[:, :3, 1:3] *= -1 # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        # transpose the R in c2w
        if not torch.all(c2w==0):
            w2c = torch.linalg.inv(c2w)
        else:
            w2c = torch.zeros_like(c2w)
        
        # w2c[:, 1:3, :3] *= -1
        # w2c[:, :3, 3] *= -1
        
        batch_R = w2c[:,:3,:3]
        batch_R_trans = batch_R.transpose(1,2).contiguous()
        w2c[:,:3,:3] = batch_R_trans
        return w2c
        # return w2c.contiguous() ## if the above have bug, try to make it contiguous
        
        return w2c
    
            
    def synthesis(self, ws, c, gs_fname_batch=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
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

            
            ### ----- original eg3d version [END] ----- 
        
        
        else:
            
            ### ----- gaussian splatting -----
            # Reshape output into three 32-channel planes
            if ('UV' in self.feature_structure):
                if self.decode_before_gridsample: # decode before grid sample
                    
                    textures_gen_batch = self.text_decoder(planes[:, :self.planes_channels_uv]) # (4, 96, 256, 256) -> (4, SH, 256, 256), range [0,1]
                    textures_gen_batch = F.grid_sample(textures_gen_batch, self.raw_uvcoords.unsqueeze(0).repeat(textures_gen_batch.shape[0],1,1,1), align_corners=False) # (B, C, 1, N_pts)

                    ## decode bg
                    bg_gen_batch = self.bg_decoder(planes[:, self.planes_channels_uv:]) # (4, C, 256, 256) -> (4, C, 256, 256), range [0,1]
                    bg_gen_batch = F.grid_sample(bg_gen_batch, self.bg_uvcoord.unsqueeze(0).repeat(bg_gen_batch.shape[0],1,1,1), align_corners=False) # (B, C, 1, N_pts)
                    _B, _C, _H, _W = bg_gen_batch.shape
                    bg_gen_batch = bg_gen_batch.reshape(_B, _C, 1, _H*_W)
                        
                else: # decode after grid sample
                    pass
                    st() #TODO: add decode before grid sample

            else:
                textures_gen_batch = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1]) # torch.Size([4, 3, 32, 256, 256])
            

            rgb_image_batch = []
            alpha_image_batch = [] # mask
            
            if self.render_fg_bg_separately:
                rgb_image_batch_fg = []
                alpha_image_batch_fg = [] # mask
                rgb_image_batch_bg = []
                alpha_image_batch_bg = [] # mask
                
            if self.regularize_fg_opacity:
                opacity_image_batch_fg = []

            bg_gaussian = GaussianModel_BG(self.sh_degree, self.bg_verts)

            # to save ply during G_ema eval
            _xyz_offset = None
            # print(f"batch size: {bg_gen_batch.shape}")
            ## TODO: can gaussiam splatting run batch in parallel?
            for _gs_i, _cam2world_matrix, _intrinsics, textures_gen, bg_gen in zip(range(len(textures_gen_batch)), cam2world_matrix, intrinsics, textures_gen_batch, bg_gen_batch): # textures_gen.shape -> torch.Size([3, 32, 256, 256])
                # randomly select a new gaussian for each rendering
                # current_gaussian = self.get_a_gaussian()
                current_gaussian_verts = self.get_a_gaussian_verts()
                current_gaussian = GaussianModel(self.sh_degree, copy.deepcopy(current_gaussian_verts))
                
                self.viewpoint_camera.update_transforms2(intrinsics, _cam2world_matrix)
                
                # update bg gaussian
                ## -- bg update starts -- 
                start_dim = 0
                _C, _H, bg_gen.shape
                textures = bg_gen[None] # [1, C, 256, 256]
                
                if self.bg_decoder.options['gen_rgb']:
                    _rgb = textures[0,start_dim:start_dim+3,0].permute(1,0)
                    start_dim += 3
                    # self.bg_gaussian.update_rgb_textures(_rgb)
                    bg_gaussian.update_rgb_textures(_rgb)
                
                if self.bg_decoder.options['gen_sh']:
                    _sh = textures[0,start_dim:start_dim+3,0].permute(1,0)
                    start_dim += 3
                    # self.bg_gaussian.update_sh_texture(_sh)
                    bg_gaussian.update_sh_texture(_sh)
                
                if self.bg_decoder.options['gen_opacity']:
                    _opacity = textures[0,start_dim:start_dim+1,0].permute(1,0) # should be no adjustment for sigmoid
                    start_dim += 1
                    # self.bg_gaussian.update_opacity(_opacity)
                    bg_gaussian.update_opacity(_opacity)
                
                if self.bg_decoder.options['gen_scaling']:
                    _scaling = textures[0,start_dim:start_dim+3,0].permute(1,0)
                    # self.bg_gaussian.update_scaling(_scaling, max_s = self.bg_decoder.options['max_scaling'], min_s = self.bg_decoder.options['min_scaling'])
                    bg_gaussian.update_scaling(_scaling, max_s = self.bg_decoder.options['max_scaling'], min_s = self.bg_decoder.options['min_scaling'])
                    start_dim += 3
                    
                if self.bg_decoder.options['gen_rotation']:
                    _rotation = textures[0,start_dim:start_dim+4,0].permute(1,0)
                    # self.bg_gaussian.update_rotation(_rotation)
                    bg_gaussian.update_rotation(_rotation)
                    start_dim += 4
                
                ## update bg gaussian _xyz according to c2w
                # current_gaussian._xyz[..., -1].min() -> tensor(0.0047, device='cuda:0'): z_far
                # current_gaussian._xyz[..., -1].max() -> tensor(0.2887, device='cuda:0'): z_near
                # under world coord, smaller (more negative) is behind the face
                # WARN: do not move the position of this line, since the attributes have an order
                ray_origins, ray_directions = self.ray_sampler(_cam2world_matrix[None], _intrinsics[None], self.bg_resolution)
                bg_xyz = (ray_origins + ray_directions * self.bg_depth)[0]
                
                if self.bg_decoder.options['gen_xyz_offset']:
                    _xyz_offset = textures[0,start_dim:start_dim+3,0].permute(1,0)
                    bg_xyz += _xyz_offset
                    start_dim += 3
                    
                # self.bg_gaussian.update_xyz(bg_xyz)
                bg_gaussian.update_xyz(bg_xyz)
            
                assert start_dim==textures.shape[1]
                
                ## -- bg update finish --
                    

                # get UV features
                if ('UV' in self.feature_structure):
                    textures = textures_gen[None] # (1, C, 1, N_pts)
                else: 
                    # do not have existing UV map for regressed 3DMM models. Directly sample feature from triplane, to ensure continuity
                    triplane_textures = sample_from_planes(self.plane_axes_gs, textures_gen[None], current_gaussian._xyz[None], padding_mode='zeros', box_warp=self.rendering_kwargs['box_warp']) # triplane_textures -> [1, 3, 35709, 32]
                    textures = self.text_decoder(triplane_textures) # textures -> (1, C, 1, N_pts), C=3 when use_colors_precomp, C=48 when use SH.
                
                # update gaussian attributes
                if self.use_colors_precomp:
                    override_color = textures[0,:3,0].permute(1,0) # override_color -> [Npts, 3], range in [0,1]
                    current_gaussian.update_rgb_textures(override_color)
                else:
                    override_color = None
                    if self.text_decoder_class == 'TextureDecoder_allAttributes':
                        
                        start_dim = 0
                        
                        if self.text_decoder.options['gen_rgb']:
                            _rgb = textures[0,start_dim:start_dim+3,0].permute(1,0)
                            start_dim += 3
                            current_gaussian.update_rgb_textures(_rgb)
                        
                        if self.text_decoder.options['gen_sh']:
                            _sh = textures[0,start_dim:start_dim+3,0].permute(1,0)
                            start_dim += 3
                            current_gaussian.update_sh_texture(_sh)
                        
                        if self.text_decoder.options['gen_opacity']:
                            _opacity = textures[0,start_dim:start_dim+1,0].permute(1,0) # should be no adjustment for sigmoid
                            start_dim += 1
                            current_gaussian.update_opacity(_opacity)
                            if self.regularize_fg_opacity:
                                opacity_image_batch_fg.append(_opacity[None])
                        
                        if self.text_decoder.options['gen_scaling']:
                            _scaling = textures[0,start_dim:start_dim+3,0].permute(1,0)
                            current_gaussian.update_scaling(_scaling, max_s = self.text_decoder.options['max_scaling'], min_s = self.text_decoder.options['min_scaling'])
                            start_dim += 3
                            
                        if self.text_decoder.options['gen_rotation']:
                            _rotation = textures[0,start_dim:start_dim+4,0].permute(1,0)
                            current_gaussian.update_rotation(_rotation)
                            start_dim += 4
                        
                        if self.text_decoder.options['gen_xyz_offset']:
                            _xyz_offset = textures[0,start_dim:start_dim+3,0].permute(1,0)
                            current_gaussian.update_xyz_offset(_xyz_offset)
                            start_dim += 3
                            
                        assert start_dim==textures.shape[1]
                        
                        
                    else:
                        current_gaussian.update_textures(textures)
                
                # save gs ply during G_ema eval
                if gs_fname_batch is not None and _gs_i==0:
                    gs_fname = f'{gs_fname_batch}_{_gs_i:02d}.ply'
                    # combine bg_gaussian and current gaussian
                    # current_gaussian_with_bg = combine_list_of_gs([current_gaussian, self.bg_gaussian])
                    current_gaussian_with_bg = combine_list_of_gs([current_gaussian, bg_gaussian])
                    current_gaussian_with_bg.save_ply(gs_fname)
                    
                res = gs_render_with_bg(self.viewpoint_camera, [current_gaussian, bg_gaussian], None, self.background, override_color=override_color)
                
                _rgb_image = res["render"]
                _alpha_image = 1 - res["alpha"]
            
           
                rgb_image_batch.append(_rgb_image[None])
                alpha_image_batch.append(_alpha_image[None])
                
                ## render foreground and bg separately
                if self.render_fg_bg_separately:
                    # rgb_image_batch_fg = []
                    # alpha_image_batch_fg = [] # mask
                    # rgb_image_batch_bg = []
                    # alpha_image_batch_bg = [] # mask
                    res_fg = gs_render(self.viewpoint_camera, current_gaussian, None, self.background, override_color=override_color)
                    _rgb_image_fg = res_fg["render"]
                    _alpha_image_fg = 1 - res_fg["alpha"]
                    rgb_image_batch_fg.append(_rgb_image_fg[None])
                    alpha_image_batch_fg.append(_alpha_image_fg[None])
                    
                    res_bg = gs_render(self.viewpoint_camera, bg_gaussian, None, self.background, override_color=override_color)
                    _rgb_image_bg = res_bg["render"]
                    _alpha_image_bg = 1 - res_bg["alpha"]
                    rgb_image_batch_bg.append(_rgb_image_bg[None])
                    alpha_image_batch_bg.append(_alpha_image_bg[None])

            
            rgb_image = torch.cat(rgb_image_batch) # [4, 3, gs_res, gs_res]
            alpha_image = torch.cat(alpha_image_batch)
            real_image = None
            
            if self.render_fg_bg_separately:
                rgb_image_fg = torch.cat(rgb_image_batch_fg) # [4, 3, gs_res, gs_res]
                alpha_image_fg = torch.cat(alpha_image_batch_fg)
                rgb_image_bg = torch.cat(rgb_image_batch_bg) # [4, 3, gs_res, gs_res]
                alpha_image_bg = torch.cat(alpha_image_batch_bg)
            
            if self.regularize_fg_opacity:
                opacity_image_fg = torch.cat(opacity_image_batch_fg) # [4, Npts, 1]
            
            if self.normalize_rgb_image:
                ## FIXME: try different normalization method to normalize rgb image to [-1,1]
                # rgb_image = (rgb_image / rgb_image.max() - 0.5) * 2
                rgb_image = (rgb_image - 0.5) * 2
                # print(f'rgb_image range: {rgb_image.min()}~{rgb_image.max()}')
                if self.render_fg_bg_separately:
                    rgb_image_fg = (rgb_image_fg - 0.5) * 2
                    rgb_image_bg = (rgb_image_bg - 0.5) * 2
            
            
            
            ## TODO: the below superresolution shall be kept?
            ## currently keeping the sr module below. TODO: shall we replace the feature image by texture_uv_map or only the sampled parts?
            if self.low_res_training:
                sr_image = rgb_image
                rgb_image = None
            elif self.gaussian_splatting_use_sr:
                sr_image = self.superresolution(rgb_image, textures_gen_batch, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
            else:
                sr_image = rgb_image
                rgb_image = rgb_image[:,:,::8, ::8] # TODO: FIXME change this downsample to a smoother gaussian filtering
            
            ## TODO: render depth_image. May not explicitly calculated from the face model since its the same for all.
            # depth_image = torch.zeros_like(rgb_image) # (N, 1, H, W)
            ### ----- gaussian splatting [END] -----

        # return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}
        # print(f"alpha range; {alpha_image.min(), alpha_image.max()}")
        synthesis_out = {'image': sr_image, 'image_raw': rgb_image, 'image_mask': alpha_image, 'image_real': real_image}
        
        if self.render_fg_bg_separately:
            # return {'image': sr_image, 'image_raw': rgb_image, 'image_mask': alpha_image, 'image_real': real_image,
            #         'image_fg': rgb_image_fg, 'image_mask_fg': alpha_image_fg, 'image_bg': rgb_image_bg, 'image_mask_bg': alpha_image_bg}
            synthesis_out.update({
                'image_fg': rgb_image_fg, 'image_mask_fg': alpha_image_fg, 'image_bg': rgb_image_bg, 'image_mask_bg': alpha_image_bg
                })
        if self.regularize_fg_opacity:
            synthesis_out.update({
                'opacity_image_fg': opacity_image_fg
                                  })
        
        return synthesis_out
    
    
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
class TextureDecoder_allAttributes_noActivations(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        
        self.options = options
        self.out_dim = 3 * options['gen_rgb'] + 3 * options['gen_sh'] + 1 * options['gen_opacity'] + 3 * options['gen_scaling'] + 4 * options['gen_rotation'] + 3 * options['gen_xyz_offset']
        self.xyz_offset_scale = options['xyz_offset_scale']
        
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.out_dim, lr_multiplier=options['decoder_lr_mul'])
        )
        self.scale_bias = options['scale_bias']
        self.scale_factor = options['scale_factor']
        print(f"scale_bias: {self.scale_bias}, scale_factor:{self.scale_factor}")
            
    def forward(self, sampled_features):
      
        # features (4, 96, 256, 256) -> (4, 16*3, 256, 256)
        # Aggregate features
    
        sampled_features = sampled_features.permute(0,2,3,1)
        x = sampled_features
        N, H, W, C = x.shape 
        x = x.reshape(N*H*W, C)
        
        x = self.net(x)
        # FIXME: do x = x.view(N, M, -1), decode after GS get features
        x = x.reshape(N, H, W, -1)
    
        start_dim = 0
        out = {}
        assert not (self.options['gen_rgb'] and self.options['gen_sh'])
        if self.options['gen_rgb']:
            out['rgb'] = torch.sigmoid(x[..., start_dim:start_dim+3])*(1 + 2*0.001) - 0.001
            start_dim += 3
        
        if self.options['gen_sh']:
            out['sh'] = x[..., start_dim:start_dim+3]
            start_dim += 3
        
        if self.options['gen_opacity']:
            out['opacity'] = x[..., start_dim:start_dim+1] # should be no adjustment for sigmoid
            start_dim += 1
        
        if self.options['gen_scaling']:
            out['scaling'] = self.scale_bias + self.scale_factor * x[..., start_dim:start_dim+3].reshape(-1,3).reshape(N, H, W, 3)
            start_dim += 3
            
        if self.options['gen_rotation']:
            out['rotation'] = x[..., start_dim:start_dim+4].reshape(-1,4).reshape(N, H, W, 4) # check consistency before/after normalize: passed. Use: x[2,2,3,7:11]/out['rotation'][2,:,2,3]
            start_dim += 4
        
        if self.options['gen_xyz_offset']:
            out['xyz_offset'] = self.xyz_offset_scale * torch.nn.functional.normalize(x[..., start_dim:start_dim+3]) # TODO: whether use this normalize? May constrain the offset not deviate too much
            start_dim += 3
            

        # x.permute(0, 3, 1, 2)
        for key, v in out.items():
            # print(f"{key}:{v.shape}")
            out[key] = v.permute(0, 3, 1, 2)
            # print(f"{key} reshape to -> :{out[key].shape}")

        out_planes = torch.cat([v for key, v in out.items()] ,dim=1)
   
        return out_planes

# decode features to SH
class TextureDecoder_allAttributes(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        
        self.options = options
        self.out_dim = 3 * options['gen_rgb'] + 3 * options['gen_sh'] + 1 * options['gen_opacity'] + 3 * options['gen_scaling'] + 4 * options['gen_rotation'] + 3 * options['gen_xyz_offset']
        self.xyz_offset_scale = 6.e-02
        
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.out_dim, lr_multiplier=options['decoder_lr_mul'])
        )
            
    def forward(self, sampled_features):
      
        # features (4, 96, 256, 256) -> (4, 16*3, 256, 256)
        # Aggregate features
    
        sampled_features = sampled_features.permute(0,2,3,1)
        x = sampled_features
        N, H, W, C = x.shape 
        x = x.reshape(N*H*W, C)
        
        x = self.net(x)
        # FIXME: do x = x.view(N, M, -1), decode after GS get features
        x = x.reshape(N, H, W, -1)
    
        start_dim = 0
        out = {}
        assert not (self.options['gen_rgb'] and self.options['gen_sh'])
        if self.options['gen_rgb']:
            out['rgb'] = torch.sigmoid(x[..., start_dim:start_dim+3])*(1 + 2*0.001) - 0.001
            start_dim += 3
        
        if self.options['gen_sh']:
            out['sh'] = x[..., start_dim:start_dim+3]
            start_dim += 3
        
        if self.options['gen_opacity']:
            out['opacity'] = torch.sigmoid(x[..., start_dim:start_dim+1]) # should be no adjustment for sigmoid
            start_dim += 1
        
        if self.options['gen_scaling']:
            out['scaling'] = torch.clamp(torch.exp(x[..., start_dim:start_dim+3].reshape(-1,3)), max=self.options['max_scaling']).reshape(N, H, W, 3)
            start_dim += 3
            
        if self.options['gen_rotation']:
            out['rotation'] = torch.nn.functional.normalize(x[..., start_dim:start_dim+4].reshape(-1,4)).reshape(N, H, W, 4) # check consistency before/after normalize: passed. Use: x[2,2,3,7:11]/out['rotation'][2,:,2,3]
            start_dim += 4
        
        if self.options['gen_xyz_offset']:
            out['xyz_offset'] = self.xyz_offset_scale * torch.nn.functional.normalize(x[..., start_dim:start_dim+3]) # TODO: whether use this normalize? May constrain the offset not deviate too much
            start_dim += 3
            

        # x.permute(0, 3, 1, 2)
        for key, v in out.items():
            # print(f"{key}:{v.shape}")
            out[key] = v.permute(0, 3, 1, 2)
            # print(f"{key} reshape to -> :{out[key].shape}")

        out_planes = torch.cat([v for key, v in out.items()] ,dim=1)
   
        return out_planes   
    
# decode features to SH
class TextureDecoder_noSigmoid(torch.nn.Module):
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
        # st()
        # x = torch.sigmoid(x)*(1 + 2*0.001) - 0.001
    
        x = x.reshape(N, H, W, -1)
        return x.permute(0, 3, 1, 2)


    
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

        self.output_scale = 20
        
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
        
        x = x * self.output_scale - self.output_scale/2
    
        x = x.reshape(N, H, W, -1)
        return x.permute(0, 3, 1, 2)

class TextureDecoder_triplane(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1) # [1, 35709, 49]
        
        x = torch.sigmoid(x)*(1 + 2*0.001) - 0.001 
        return x.permute(0,2,1).unsqueeze(2) # -> [1, 48, 1, 53215]