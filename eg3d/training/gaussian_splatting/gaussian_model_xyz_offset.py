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
import numpy as np
from training.gaussian_splatting.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_covariance_from_scaling_rotation
from torch import nn
import os
from training.gaussian_splatting.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from training.gaussian_splatting.utils.sh_utils import RGB2SH, SH2RGB
# from training.gaussian_splatting.submodules.simple_knn._C import distCUDA2
from simple_knn._C import distCUDA2
from training.gaussian_splatting.utils.graphics_utils import BasicPointCloud
from training.gaussian_splatting.utils.general_utils import strip_symmetric, build_scaling_rotation

from ipdb import set_trace as st

from training.gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser, Namespace
## args to setup training for gaussians
parser = ArgumentParser(description="Training script parameters")
# lp = ModelParams(parser)
op = OptimizationParams(parser)
# pp = PipelineParams(parser)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6009)
parser.add_argument('--debug_from', type=int, default=-1)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default = None)
args = parser.parse_args([])
args.save_iterations.append(args.iterations)

gs_training_args = op.extract(args)
# print(f"gs_training_args: {vars(gs_training_args)}")
fix_opacity_scaling_rotation=False
print(f"You choose to fix_opacity_scaling_rotation:{fix_opacity_scaling_rotation}")
if fix_opacity_scaling_rotation:
    gs_training_args.opacity_lr=0
    gs_training_args.scaling_lr=0
    gs_training_args.rotation_lr=0

def print_grad(name, grad):
    print(f"{name}:")
    if torch.all(grad==0):
        print("grad all 0s")
        return 
    # print(grad)
    print('\t',grad.max(), grad.min(), grad.mean())
    
class GaussianModel_OffsetXYZ:

    def setup_functions(self):
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, verts, index=-1, active_sh_degree=0):
        self.active_sh_degree = min(active_sh_degree, sh_degree)
        
        self.update_iterations = 0 # to record how many times this gaussian has been updated. for the use of oneupSHdegree()
        self.update_interval = 10000
        self.max_sh_degree = sh_degree 
        # self._xyz = torch.empty(0)
        self._xyz_base = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.init_point_cloud(verts)
        self.index = index
        self.training_setup(gs_training_args)
        self.max_s = None
        self.min_s = None
        # print(f"init gs_{self.index} has activeSH={self.active_sh_degree}; maxSH={self.max_sh_degree}")
    

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def get_copy(self):
        # Create a new instance of GaussianModel
        copied_instance = GaussianModel_OffsetXYZ(sh_degree=self.max_sh_degree, verts=self._xyz, index=self.index, active_sh_degree=self.active_sh_degree)

        # Copy the necessary attributes
        copied_instance._xyz = self._xyz.clone().detach()  # Detach if needed
        copied_instance._features_dc = self._features_dc.clone().detach()
        copied_instance._features_rest = self._features_rest.clone().detach()
        copied_instance._scaling = self._scaling.clone().detach()
        copied_instance._rotation = self._rotation.clone().detach()
        copied_instance._opacity = self._opacity.clone().detach()
        copied_instance.max_radii2D = self.max_radii2D.clone().detach()
        copied_instance.xyz_gradient_accum = self.xyz_gradient_accum.clone().detach()
        copied_instance.denom = self.denom.clone().detach()
        
        return copied_instance #TODO: test this function
    
    def parameters(self):
        """
        Returns an iterator over the parameters of the GaussianModel.

        Returns:
            Iterator: Iterator over parameters.
        """
        # Define your logic to return parameters here
        # For example, if your parameters are stored in a list, you can return that list
        return [self._scaling, self._rotation, self._opacity]
        ## FIXME: self._features_dc, self._features_rest: non-leaf tensors, cannot requries grad=False

    
    def requires_grad(self, requires_grad=True):
        """
        Toggles the calculation of gradients for the parameters of the GaussianModel.

        Args:
            requires_grad (bool): If True, gradients will be calculated for the parameters.
                                 If False, gradients will not be calculated.

        Returns:
            None
        """
        for param in self.parameters():
            param.requires_grad_(requires_grad)
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if (self.max_s is not None) or (self.min_s is not None):
        # if (getattr(self, 'max_s', None) is not None) or (self.min_s is not None):
            return torch.clamp(self.scaling_activation(self._scaling), max=self.max_s, min=self.min_s)
        
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # [N, 3, 16]
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # TODO: create gaussian from generated texture
    # this function is corresponding to the "create_from_pcd" in the original gs implementation, but removed the parameterization for color 
    def init_point_cloud(self, verts):
        # TODO: freeze some parameters, like self._xyz?
        self._xyz_base = verts
        self._xyz = verts
        self.spatial_lr_scale = 0 # FIXME: hardcoded from original GS implementation, explained by authors in issue

        dist2 = torch.clamp_min(distCUDA2(self._xyz.to(torch.float32)), 0.0000001) ## TODO:FIXME HOW is this param 0.0000001 determined?
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((self._xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        if gs_training_args.opacity_lr==0:
            # print("when no lr for opacity, init opacity to ones")
            opacities = torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device="cuda")
        else:
            opacities = inverse_sigmoid(0.1 * torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # for test
    def create_from_ply(self, path, spatial_lr_scale : float):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)


        features = torch.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # [N, 3, 16]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc = np.stack((np.asarray(plydata.elements[0]["red"]),
                                np.asarray(plydata.elements[0]["green"]),
                                np.asarray(plydata.elements[0]["blue"])), axis=1)
        features[:, :3, 0 ] = RGB2SH(torch.tensor(features_dc, dtype=torch.float, device="cuda"))
        features[:, 3:, 1:] = 0.0

        # self.active_sh_degree = self.max_sh_degree

        # print("Number of points at initialisation : ", xyz.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        # opacities = torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    # to update gaussian bank
    def update_textures(self, feature_uv):
        # self.update_iterations += 1 # do this before update. start from activeSH=0
        # # print(f"gs_{self.index} at iteration {self.update_iterations}")
        
        # if self.update_iterations % self.update_interval == 0:
        #     self.oneupSHdegree()
        #     print(f"gs{self.index} now has {self.active_sh_degree} active_sh_degree")
            
        # print(f"feature_uv min={feature_uv.min()}, max={feature_uv.max()}")
        
        N, C, _ , V = feature_uv.shape # (1, 48, 1, 5023)
        features = feature_uv.permute(3,1,2,0).reshape(V,3,C//3).contiguous() # [5023, 3, 16] [V, 3, C']
        # print(f"--shs: min={shs.min()}, max={shs.max()}, mean={shs.mean()}, shape={shs.shape}")
        # features = torch.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        ## FIXME: although all features are mapped from RGB[0,1] to SH now, the original GS uses 0 for self._feature_rest

        # print(f"features.requires_grad: {features.requires_grad}") # True when 'G' in phase.name, False when 'D' in phase.name
        # if features.requires_grad:
        #     features.retain_grad()
        #     features.register_hook(lambda grad: print_grad("------features.requires_grad", grad))
        ## not optimizing features_dc, but rather it comes from the generator
        ## thus both the below are set to: requires_grad -> False   
        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()#.requires_grad_(True)
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()#.requires_grad_(True)
        # print(f"self._features_dc.requires_grad: {self._features_dc.requires_grad}, self._features_rest.requires_grad: {self._features_rest.requires_grad}")
        
        # self._xyz.register_hook(lambda grad: print_grad("------_xyz.requires_grad", grad))
        # self._scaling.register_hook(lambda grad: print_grad("------_scaling.requires_grad", grad))
        # self._rotation.register_hook(lambda grad: print_grad("------_rotation.requires_grad", grad))
        # self._opacity.register_hook(lambda grad: print_grad("------_opacity.requires_grad", grad))
    
    # to update gaussian bank
    def update_sh_texture(self, feature_uv):
        V, C = feature_uv.shape # (1, 48, 1, 5023)
        features = feature_uv.reshape(V,3,C//3).contiguous() # [5023, 3, 16] [V, 3, C']
       
        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()#.requires_grad_(True) # [V, 1, 3]
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()#.requires_grad_(True)# [V, sh degree - 1, 3]
    
    def update_xyz_offset(self, xyz_offset):
        self._xyz = self._xyz_base + xyz_offset

    def update_opacity(self, opacity):
        self._opacity = opacity

    def update_scaling(self, scaling, max_s=None, min_s=None):  
        # clamp settings
        self.max_s = max_s
        self.min_s = min_s
        
        # update
        self._scaling = scaling
    
    def update_rotation(self, rotation):
        self._rotation = rotation

        
    ## for assigning rgb texture to G.debug_gaussian. Not for other gaussians
    def update_rgb_textures(self, feature_uv):
        '''
            feature_uv: [Npts, 3]
        '''

        # N, C, _ , V = feature_uv.shape # (1, 48, 1, 5023)
        # features = feature_uv.permute(3,1,2,0).reshape(V,3,C//3).contiguous() # [5023, 3, 16] [V, 3, C']
        # print(f"--shs: min={shs.min()}, max={shs.max()}, mean={shs.mean()}, shape={shs.shape}")
        features = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        ## FIXME: although all features are mapped from RGB[0,1] to SH now, the original GS uses 0 for self._feature_rest
        features[:,:3, 0] = RGB2SH(feature_uv) # (53215, 3): 0~1 -> -1.7~+1.7
        features[:, 3:, 1:] = 0.0

        # self._xyz = nn.Parameter(xyz.clone().detach().to(torch.float32).requires_grad_(False))
        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()#.requires_grad_(True)
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)
        
        # features.requires_grad_(True)
        # features.register_hook(lambda grad: print_grad("------features.requires_grad", grad))
        
        # self._features_dc.requires_grad_(True)
        # self._features_dc.register_hook(lambda grad: print_grad("------self._features_dc.requires_grad", grad))
        # self._features_rest.requires_grad_(True) ## exactly the same as the grad of inside gs_render(): shs.grad
        # self._features_rest.register_hook(lambda grad: print_grad("------self._features_rest.requires_grad", grad))
        
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    # for test
    def create_from_ply2(self, feature_uv):
        print("should Not in create_from_ply2")
        exit(0)
        N, C, _ , V = feature_uv.shape # (1, 48, 1, 5023)
        features = feature_uv.permute(3,1,2,0).reshape(V,3,C//3).contiguous() # [5023, 3, 16] [V, 3, C']
        # features = torch.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features[:, :3, 0 ] = RGB2SH(feature_uv) # (53215, 3)
        # features[:, 3:, 1:] = 0.0

        # self.active_sh_degree = self.max_sh_degree

        # print("Number of points at initialisation : ", xyz.shape[0])

        dist2 = torch.clamp_min(distCUDA2(self._xyz.to(torch.float32)), 0.0000001) ## TODO:FIXME HOW is this param 0.0000001 determined?
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((self._xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        # opacities = torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda")

        # self._xyz = nn.Parameter(xyz.clone().detach().to(torch.float32).requires_grad_(False))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        features.requires_grad_(True)
        features.register_hook(lambda grad: print_grad("------features.requires_grad", grad))
        # self._features_dc.requires_grad_(True)
        # self._features_dc.register_hook(lambda grad: print_grad("------self._features_dc.requires_grad", grad))
        # self._features_rest.requires_grad_(True) ## exactly the same as the grad of inside gs_render(): shs.grad
        # self._features_rest.register_hook(lambda grad: print_grad("------self._features_rest.requires_grad", grad))
        
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
       
        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        # FIXME: not pickle-able. Adjusting the lr foir xyz dynamically.
        # self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.position_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]): # V, 3, 16
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, save_override_color=False):
        if not save_override_color:
            mkdir_p(os.path.dirname(path))

            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)
    
        else:
            mkdir_p(os.path.dirname(path))

            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            
            # FIXME: output 3dmm ply, see if the color is right
            vertex = np.empty(xyz.shape[0], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
            vertex[:] = list(map(tuple, xyz))
            colors = np.empty(f_dc.shape[0], dtype=[('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
            colors[:] = list(map(tuple, SH2RGB(f_dc)*255))
            
            vertex_all = np.empty(xyz.shape[0], vertex.dtype.descr + colors.dtype.descr)
            for prop in vertex.dtype.names:
                vertex_all[prop] = vertex[prop]

            for prop in colors.dtype.names:
                vertex_all[prop] = colors[prop]
            # color = PlyElement.describe(colors, 'color')
            # el = PlyElement.describe(vertex, 'vertex')
            el = PlyElement.describe(vertex_all, 'vertex')
            PlyData([el]).write(path)


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1